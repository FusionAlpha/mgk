#!/usr/bin/env python3
"""Build an audited MGK1D external candidate without deleting prior output."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 release tooling
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        from pip._vendor import tomli as tomllib  # type: ignore[no-redef]

from release_audit import ROOT, _markdown, audit


DOCUMENTATION_FILES = (
    "README_EXTERNAL.md", "LICENSE", "CONFIGURATION_REFERENCE.md", "GPU_GUIDE.md",
    "BENCHMARK_CASES.md", "CHANGELOG.md", "COMPATIBILITY.md", "SECURITY.md",
    "SUPPORT.md", "THIRD_PARTY_NOTICES.md", "PUBLISHED_VALIDATION_CASES.md",
    "RELEASE_PROCESS.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_source_paths() -> set[str] | None:
    """Return the repository's tracked file list when building from Git."""
    try:
        completed = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return set(completed.stdout.splitlines())


def _sbom(version: str, artifacts: list[Path], classification: str) -> dict:
    serial_seed = hashlib.sha256(
        "\n".join(f"{path.name}:{_sha256(path)}" for path in artifacts).encode()
    ).hexdigest()
    root_ref = f"pkg:pypi/mgk@{version}"
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": (
            f"urn:uuid:{serial_seed[:8]}-{serial_seed[8:12]}-"
            f"{serial_seed[12:16]}-{serial_seed[16:20]}-{serial_seed[20:32]}"
        ),
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tools": {"components": [{
                "type": "application", "name": "MGK1D release tooling",
                "version": version,
            }]},
            "component": {
                "type": "library", "bom-ref": root_ref, "name": "mgk",
                "version": version, "purl": root_ref,
                "properties": [{
                    "name": "release.classification",
                    "value": classification,
                }],
            },
        },
        "components": [
            {
                "type": "library", "bom-ref": "pkg:pypi/numpy",
                "name": "numpy", "version": ">=2.0,<3",
                "licenses": [{"license": {"id": "BSD-3-Clause"}}],
                "properties": [{"name": "dependency.scope", "value": "runtime"}],
            },
            {
                "type": "library", "bom-ref": "pkg:pypi/scipy",
                "name": "scipy", "version": ">=1.13,<2",
                "licenses": [{"license": {"id": "BSD-3-Clause"}}],
                "properties": [{"name": "dependency.scope", "value": "runtime"}],
            },
            {
                "type": "library", "bom-ref": "pkg:pypi/cupy-cuda12x",
                "name": "cupy-cuda12x", "version": ">=13,<15",
                "licenses": [{"license": {"id": "MIT"}}],
                "properties": [{"name": "dependency.scope", "value": "optional-gpu12"}],
            },
            {
                "type": "library", "bom-ref": "pkg:pypi/cupy-cuda13x",
                "name": "cupy-cuda13x", "version": ">=14,<15",
                "licenses": [{"license": {"id": "MIT"}}],
                "properties": [{"name": "dependency.scope", "value": "optional-gpu13"}],
            },
        ],
        "dependencies": [{
            "ref": root_ref,
            "dependsOn": ["pkg:pypi/numpy", "pkg:pypi/scipy"],
        }],
        "properties": [
            {"name": "sbom.note", "value": (
                "Dependency versions are declared ranges, not a customer-environment lock. "
                "NVIDIA driver/runtime components are externally supplied."
            )},
        ],
    }


def _build_documentation_bundle(output: Path, version: str) -> Path:
    destination = output / f"mgk-{version}-documentation.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "README.md",
            "# MGK1D documentation\n\n"
            "Start with [README_EXTERNAL.md](README_EXTERNAL.md). The complete "
            "configuration reference, GPU guide, benchmark contracts, validation "
            "cases, and support documents are included in this archive.\n",
        )
        for name in DOCUMENTATION_FILES:
            source = ROOT / name
            if not source.exists():
                continue
            archive.write(source, name)
        examples = ROOT / "mgk" / "examples" / "validation"
        for source in sorted(examples.glob("*")):
            if source.is_file() and source.suffix in {".py", ".json"}:
                archive.write(source, f"mgk/examples/validation/{source.name}")
        case_studies = ROOT / "research_results" / "case_studies"
        tracked_paths = _tracked_source_paths()
        for source in sorted(case_studies.rglob("*")):
            # The public documentation bundle is Python-only. Local case
            # folders may still contain private comparison exports, so filter
            # them here as a second line of defense beyond MANIFEST.in.
            relative = source.relative_to(ROOT).as_posix()
            if (source.is_file()
                    and (tracked_paths is None or relative in tracked_paths)
                    and source.suffix.lower() not in {".m", ".mat"}
                    and "matlab" not in source.name.lower()):
                archive.write(source, relative)
        for name in (
            "make_stellarator_profile.py",
            "convert_vmec_geometry.py",
            "convert_gene_geometry.py",
        ):
            profile_tool = ROOT / "tools" / name
            if profile_tool.exists():
                archive.write(profile_tool, f"tools/{name}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-license-pending", action="store_true")
    args = parser.parse_args()

    project = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]
    version = project["version"]
    output = args.output_dir or (
        ROOT / "release_build" / "external_candidate" /
        f"mgk-{version}-technical-rc"
    )
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    preflight = audit([], args.allow_license_pending)
    if preflight["summary"]["failed"]:
        print(json.dumps(preflight, indent=2), file=sys.stderr)
        raise SystemExit("release preflight failed")

    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("uv is required to build the release artifacts")
    with tempfile.TemporaryDirectory(prefix="mgk-build-") as temporary:
        staging = Path(temporary)
        subprocess.run(
            [uv, "build", str(ROOT), "--out-dir", str(staging),
             "--no-create-gitignore", "--python", sys.executable],
            cwd=ROOT, check=True,
        )
        built = sorted(path for path in staging.iterdir() if path.is_file())
        if not built:
            raise SystemExit("build produced no artifacts")
        artifacts = []
        for source in built:
            destination = output / source.name
            shutil.copy2(source, destination)
            artifacts.append(destination)

    artifacts.append(_build_documentation_bundle(output, version))

    report = audit(artifacts, args.allow_license_pending)
    (output / "RELEASE_AUDIT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )
    (output / "RELEASE_AUDIT.md").write_text(_markdown(report), "utf-8")
    if report["summary"]["failed"]:
        raise SystemExit("built artifact audit failed")

    hashes = [f"{_sha256(path)}  {path.name}" for path in artifacts]
    (output / "SHA256SUMS").write_text("\n".join(hashes) + "\n", "utf-8")
    (output / "SBOM.cdx.json").write_text(
        json.dumps(_sbom(version, artifacts, report["releaseStatus"]), indent=2)
        + "\n", "utf-8"
    )
    build_info = {
        "project": project["name"], "version": version,
        "classification": report["releaseStatus"],
        "createdUtc": datetime.now(timezone.utc).isoformat(),
        "buildPython": platform.python_version(),
        "buildInterpreter": sys.executable,
        "buildPlatform": platform.platform(),
        "artifacts": [path.name for path in artifacts],
        "licensePendingOverride": args.allow_license_pending,
    }
    (output / "BUILD_INFO.json").write_text(
        json.dumps(build_info, indent=2) + "\n", "utf-8"
    )
    print(f"built and audited {len(artifacts)} artifacts in {output}")


if __name__ == "__main__":
    main()
