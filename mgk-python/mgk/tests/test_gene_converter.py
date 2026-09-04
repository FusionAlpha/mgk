import numpy as np

from tools.convert_gene_geometry import convert_gene_geometry


def test_gene_geometry_converter_uses_periodic_ky_reduction(tmp_path):
    n = 16
    z = np.linspace(-np.pi, np.pi, n, endpoint=False)
    field = 1.0 + 0.1 * np.cos(z)
    gxx = np.full(n, 2.0)
    gxy = np.full(n, 0.2)
    gxz = np.full(n, 0.1)
    gyy = 1.2 + 0.1 * np.sin(z)
    gyz = np.full(n, 0.05)
    gzz = np.full(n, 1.5)
    dbdx = 0.03 * np.cos(z)
    dbdy = np.zeros(n)
    dbdz = -0.1 * np.sin(z)
    jacobian = np.full(n, 2.0)
    table = np.column_stack(
        (gxx, gxy, gxz, gyy, gyz, gzz, field, dbdx, dbdy, dbdz,
         jacobian, np.zeros((n, 5)))
    )
    source = tmp_path / "vmec.dat"
    header = """&parameters
gridpoints = 16
Cxy = 2.0
q0 = -1.25
major_R = 10.0
/
"""
    source.write_text(header + "\n" + "\n".join(
        " ".join(f"{value:.16e}" for value in row) for row in table
    ) + "\n", encoding="utf-8")
    output = tmp_path / "profile.npz"
    convert_gene_geometry(source, output)
    with np.load(output) as profile:
        np.testing.assert_allclose(profile["kperpMetric"], np.sqrt(gyy))
        np.testing.assert_allclose(
            profile["parallelGradient"], -1.25 * 10.0 * 2.0 / (jacobian * field)
        )
        ga1 = gxx * gyy - gxy**2
        ga3 = gxy * gyz - gyy * gxz
        gene_k_y = (dbdx - ga3 / ga1 * dbdz) / 2.0
        np.testing.assert_allclose(profile["driftCurvature"], 10.0 * gene_k_y / field)
        np.testing.assert_array_equal(profile["driftParallelCorrection"], 0.0)
        assert profile["geneQ"] == -1.25
        assert profile["referenceLengthRatio"] == 10.0
        assert profile["geneKyCurvatureCoefficient"].shape == (n,)
        assert np.isfinite(profile["logFieldDerivative"]).all()


def test_gene_converter_accepts_explicit_mgk_reference_length(tmp_path):
    n = 8
    table = np.column_stack((
        np.ones(n), np.zeros(n), np.zeros(n), np.ones(n), np.zeros(n),
        np.ones(n), np.ones(n), np.full(n, 0.1), np.zeros(n), np.zeros(n),
        np.full(n, 2.0), np.zeros((n, 5)),
    ))
    source = tmp_path / "vmec.dat"
    source.write_text(
        "&parameters\ngridpoints = 8\nCxy = 1\nq0 = 2\nmajor_R = 9\n/\n"
        + "\n".join(" ".join(map(str, row)) for row in table),
        encoding="utf-8",
    )
    output = tmp_path / "profile.npz"
    convert_gene_geometry(source, output, reference_length_ratio=11.0)
    with np.load(output) as profile:
        np.testing.assert_allclose(profile["driftCurvature"], 1.1)
        np.testing.assert_allclose(profile["parallelGradient"], 11.0)
        assert profile["referenceLengthRatio"] == 11.0


def test_gene_converter_open_profile_does_not_wrap_derivatives(tmp_path):
    n = 8
    field = np.linspace(1.0, 1.7, n)
    table = np.column_stack((
        np.ones(n), np.zeros(n), np.zeros(n), np.ones(n), np.zeros(n),
        np.ones(n), field, np.zeros(n), np.zeros(n), np.zeros(n),
        np.ones(n), np.zeros((n, 5)),
    ))
    source = tmp_path / "vmec.dat"
    source.write_text(
        "&parameters\ngridpoints = 8\nCxy = 1\nq0 = 1\nmajor_R = 1\n/\n"
        + "\n".join(" ".join(map(str, row)) for row in table),
        encoding="utf-8",
    )
    output = tmp_path / "profile.npz"
    convert_gene_geometry(source, output, periodic=False)
    with np.load(output) as profile:
        assert not bool(profile["periodic"])
        expected = np.gradient(np.log(field), 2 * np.pi / n, edge_order=2)
        np.testing.assert_allclose(profile["logFieldDerivative"][:-1], expected)
        assert profile["z"].size == n + 1
        np.testing.assert_allclose(profile["z"][-1] - profile["z"][0], 2 * np.pi)
