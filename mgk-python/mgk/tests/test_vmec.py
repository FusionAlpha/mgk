import numpy as np
from scipy.io import netcdf_file

import mgk
from mgk.internal.geometry import evaluate_geometry, load_stellarator_profile
from mgk.internal.configuration import build_config


def _scalar(dataset, name, kind, value):
    variable = dataset.createVariable(name, kind, ())
    variable[...] = value


def _minimal_wout(path):
    ns, modes = 3, 2
    with netcdf_file(path, "w") as dataset:
        dataset.createDimension("radius", ns)
        dataset.createDimension("mn_mode", modes)
        dataset.createDimension("mn_mode_nyq", modes)
        for name, value in (
            ("ns", ns), ("nfp", 2), ("mpol", 2), ("ntor", 0),
            ("mnmax", modes), ("mnmax_nyq", modes),
        ):
            _scalar(dataset, name, "i", value)
        for name, value in (("b0", 1.0), ("Aminor_p", 0.5), ("Rmajor_p", 2.0)):
            _scalar(dataset, name, "d", value)

        for name, dimension, values in (
            ("xm", "mn_mode", [0.0, 1.0]),
            ("xn", "mn_mode", [0.0, 0.0]),
            ("xm_nyq", "mn_mode_nyq", [0.0, 1.0]),
            ("xn_nyq", "mn_mode_nyq", [0.0, 0.0]),
            ("iotaf", "radius", [0.5, 0.5, 0.5]),
        ):
            variable = dataset.createVariable(name, "d", (dimension,))
            variable[:] = values

        def surface(name, mode_dimension, row):
            variable = dataset.createVariable(name, "d", ("radius", mode_dimension))
            variable[:] = np.tile(np.asarray(row, dtype=float), (ns, 1))

        surface("rmnc", "mn_mode", [2.0, 0.5])
        surface("zmns", "mn_mode", [0.0, 0.5])
        surface("lmns", "mn_mode", [0.0, 0.0])
        surface("bmnc", "mn_mode_nyq", [1.0, 0.1])
        surface("bsupumnc", "mn_mode_nyq", [0.25, 0.0])
        surface("bsupvmnc", "mn_mode_nyq", [1.0, 0.0])


def test_direct_vmec_reader_and_profile(tmp_path):
    wout = tmp_path / "wout_test.nc"
    profile_path = tmp_path / "direct_vmec_profile.npz"
    _minimal_wout(wout)

    equilibrium = mgk.read_vmec_wout(wout)
    assert equilibrium.ns == 3
    assert equilibrium.nfp == 2
    np.testing.assert_allclose(equilibrium.iotaf, 0.5)

    mgk.write_vmec_profile(
        wout, profile_path, s=0.5, alpha=0.2,
        z_min=-np.pi, z_max=np.pi, num_points=65,
    )
    profile = load_stellarator_profile(profile_path)
    assert profile.periodic is False
    assert profile.z.size == 65
    assert np.all(profile.field > 0)
    assert np.all(profile.kperpMetric > 0)
    assert np.all(profile.parallelGradient > 0)
    assert np.all(np.isfinite(profile.driftCurvature))


def test_nonperiodic_vmec_profile_geometry(tmp_path):
    wout = tmp_path / "wout_test.nc"
    profile_path = tmp_path / "direct_vmec_profile.npz"
    _minimal_wout(wout)
    mgk.write_vmec_profile(
        wout, profile_path, s=0.5, z_min=-np.pi, z_max=np.pi,
        num_points=65,
    )
    config = {
        "geometry": {
            "model": "stellarator", "profileFile": str(profile_path),
        },
        "model": {"parallelBoundary": "open"},
        "grid": {
            "thetaMin": -np.pi, "thetaMax": np.pi, "numTheta": 17,
            "numEnergy": 2, "numPitch": 4, "numBouncePoints": 8,
        },
        "solver": {"useGpu": False},
    }
    _, normalized, _ = build_config(config)
    assert normalized.q == 2.0
    geometry = evaluate_geometry(np.linspace(-np.pi, np.pi, 33), normalized)
    assert np.all(np.isfinite(geometry.fieldStrength))
    assert np.all(np.isfinite(geometry.driftCurvature))
    assert np.all(geometry.parallelGradient > 0)

    with np.testing.assert_raises_regex(ValueError, "only covers"):
        evaluate_geometry(np.asarray([2 * np.pi]), normalized)
