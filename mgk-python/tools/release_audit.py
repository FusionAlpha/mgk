#!/usr/bin/env python3
"""Audit MGK1D source and built artifacts for external-release hygiene."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 release tooling
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        from pip._vendor import tomli as tomllib  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DOCS = (
    "README_EXTERNAL.md", "CONFIGURATION_REFERENCE.md", "GPU_GUIDE.md",
    "BENCHMARK_CASES.md", "CHANGELOG.md", "COMPATIBILITY.md", "SECURITY.md",
    "SUPPORT.md", "THIRD_PARTY_NOTICES.md", "PUBLISHED_VALIDATION_CASES.md",
    "RELEASE_PROCESS.md",
)
FORBIDDEN_ARCHIVE_PARTS = {
    "matlab_legacy", "analysis", "docs", "figures", "output", "release_build",
    "tests", "tmp", ".cupy_cache", ".venv", "__pycache__",
}
FORBIDDEN_SUFFIXES = {
    ".pdf", ".mat", ".m", ".cubin", ".npz", ".png", ".jpg", ".jpeg",
    ".pyc", ".pyo",
}
SENSITIVE_PATTERNS = {
    "macOS absolute user path": re.compile(rb"/Users/[A-Za-z0-9._-]+/"),
    "Linux absolute home path": re.compile(rb"/home/[A-Za-z0-9._-]+/"),
    "private IPv4 address": re.compile(
        rb"(?<![0-9])(?:10\.[0-9]{1,3}(?:\.[0-9]{1,3}){2}|"
        rb"192\.168\.[0-9]{1,3}\.[0-9]{1,3}|"
        rb"172\.(?:1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3})(?![0-9])"
    ),
}


@dataclass
class Check:
    name: str
    status: str
    detail: str
    blocking: bool = True


def _add(checks: list[Check], name: str, passed: bool, detail: str,
         *, blocking: bool = True) -> None:
    checks.append(Check(name, "pass" if passed else ("fail" if blocking else "warn"),
                        detail, blocking))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_members(path: Path) -> list[tuple[str, bytes]]:
    members: list[tuple[str, bytes]] = []
    if path.suffix == ".whl" or zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    members.append((info.filename, archive.read(info)))
        return members
    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as archive:
            for info in archive.getmembers():
                if info.isfile():
                    handle = archive.extractfile(info)
                    members.append((info.name, b"" if handle is None else handle.read()))
        return members
    raise ValueError(f"unsupported artifact type: {path}")


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def _audit_artifact(path: Path, checks: list[Check]) -> dict:
    if not path.is_file():
        _add(checks, f"artifact exists: {path.name}", False, str(path))
        return {}
    try:
        members = _archive_members(path)
    except Exception as exc:
        _add(checks, f"artifact readable: {path.name}", False,
             f"{type(exc).__name__}: {exc}")
        return {}
    names = [name for name, _ in members]
    unsafe = [name for name in names if not _safe_member(name)]
    _add(checks, f"archive paths safe: {path.name}", not unsafe,
         "no absolute/traversal paths" if not unsafe else repr(unsafe[:5]))
    forbidden = []
    for name in names:
        pure = PurePosixPath(name)
        parts = set(pure.parts)
        suffix = pure.suffix.lower()
        is_case_study = "research_results" in parts and "case_studies" in parts
        is_python_tests = "mgk" in parts and "tests" in parts
        forbidden_parts = parts & FORBIDDEN_ARCHIVE_PARTS
        if is_case_study:
            forbidden_parts.discard("figures")
        if is_python_tests:
            forbidden_parts.discard("tests")
        # Curated case studies and the Python regression suite are intentionally
        # distributable. Historical analysis, legacy implementations, raw
        # figures, and binary/cache artifacts remain forbidden outside reviewed
        # paths. Case studies contain Python scripts and report figures only.
        allowed_case_suffixes = {".png", ".jpg", ".jpeg"}
        forbidden_suffix = suffix in FORBIDDEN_SUFFIXES and not (
            is_case_study and suffix in allowed_case_suffixes
        )
        if forbidden_parts or forbidden_suffix:
            forbidden.append(name)
        if len(pure.parts) == 2 and pure.name == "README.md" and not is_case_study:
            forbidden.append(name)
    _add(checks, f"archive content allowlist: {path.name}", not forbidden,
         "no forbidden historical/internal/binary-cache assets" if not forbidden
         else repr(forbidden[:10]))
    leaks = []
    for name, data in members:
        if len(data) > 5_000_000:
            continue
        for label, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(data):
                leaks.append(f"{name}: {label}")
    _add(checks, f"archive path/IP scan: {path.name}", not leaks,
         "no local absolute paths or private IPs" if not leaks else repr(leaks[:10]))
    is_wheel = path.suffix == ".whl"
    if is_wheel:
        non_metadata = [n for n in names if ".dist-info/" not in n]
        outside_package = [
            n for n in non_metadata
            if not n.startswith("mgk/") and n != "mgk1d.py"
        ]
        _add(checks, f"wheel package boundary: {path.name}", not outside_package,
             "only mgk package, compatibility shim, plus dist-info" if not outside_package
             else repr(outside_package[:10]))
        license_members = [n for n in names if ".dist-info/licenses/" in n]
        metadata_license_fields = [
            name for name, data in members
            if name.endswith(".dist-info/METADATA") and b"\nLicense-File:" in data
        ]
        unexpected_license = (
            not (ROOT / "LICENSE").is_file()
            and bool(license_members or metadata_license_fields)
        )
        _add(checks, f"wheel license authenticity: {path.name}",
             not unexpected_license,
             "no placeholder is represented as an approved license"
             if not unexpected_license
             else repr(license_members + metadata_license_fields))
    return {
        "path": str(path.resolve()), "sha256": _sha256(path),
        "bytes": path.stat().st_size, "members": len(members),
    }


def audit(artifacts: Iterable[Path], allow_license_pending: bool) -> dict:
    checks: list[Check] = []
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    project = pyproject["project"]
    version = project["version"]

    package_init = (ROOT / "mgk" / "__init__.py").read_text("utf-8")
    _add(checks, "version consistency", f'__version__ = "{version}"' in package_init,
         f"pyproject and source fallback are {version}")
    _add(checks, "runtime dependency bounds",
         project.get("dependencies") == ["numpy>=2.0,<3", "scipy>=1.13,<2"],
         repr(project.get("dependencies")))
    _add(checks, "supported Python declaration",
         project.get("requires-python") == ">=3.10",
         str(project.get("requires-python")))

    missing_docs = [name for name in REQUIRED_DOCS if not (ROOT / name).is_file()]
    _add(checks, "external documentation set", not missing_docs,
         "complete" if not missing_docs else repr(missing_docs))
    manifest = (ROOT / "MANIFEST.in").read_text("utf-8")
    missing_manifest = [name for name in REQUIRED_DOCS if name not in manifest]
    _add(checks, "sdist documentation manifest", not missing_manifest,
         "all external docs included" if not missing_manifest else repr(missing_manifest))

    license_exists = (ROOT / "LICENSE").is_file()
    if license_exists:
        license_text = (ROOT / "LICENSE").read_text("utf-8")
        _add(
            checks,
            "BSD 3-Clause LICENSE",
            license_text.startswith("BSD 3-Clause License\n")
            and "Copyright (c) 2026, FusionAlpha" in license_text,
            "BSD 3-Clause text and FusionAlpha copyright line present",
        )
        _add(checks, "PEP 639 license metadata",
             project.get("license") == "BSD-3-Clause"
             and "LICENSE" in project.get("license-files", []),
             "pyproject declares BSD-3-Clause and LICENSE")
    elif allow_license_pending:
        _add(checks, "BSD 3-Clause LICENSE", False,
             "pending: technical release candidate only", blocking=False)
        _add(checks, "PEP 639 license metadata", False,
             "pending with company license decision", blocking=False)
    else:
        _add(checks, "BSD 3-Clause LICENSE", False,
             "LICENSE is required for external distribution")
        _add(checks, "PEP 639 license metadata", False,
             "add metadata after the company license decision")

    support = (ROOT / "SUPPORT.md").read_text("utf-8")
    has_pending_support = "must be supplied" in support
    _add(checks, "external support identity/contact", not has_pending_support,
         "contractual support identity/contact remain an organization decision",
         blocking=False)

    artifact_records = [_audit_artifact(path, checks) for path in artifacts]
    artifact_records = [record for record in artifact_records if record]
    failures = [check for check in checks if check.status == "fail"]
    warnings = [check for check in checks if check.status == "warn"]
    return {
        "schemaVersion": 1,
        "project": project["name"],
        "version": version,
        "releaseStatus": "blocked" if failures else (
            "technical_release_candidate" if warnings else "release_ready"
        ),
        "checks": [asdict(check) for check in checks],
        "summary": {
            "passed": sum(c.status == "pass" for c in checks),
            "warnings": len(warnings), "failed": len(failures),
        },
        "artifacts": artifact_records,
    }


def _markdown(report: dict) -> str:
    lines = [
        f"# MGK1D {report['version']} external release audit", "",
        f"Status: **{report['releaseStatus']}**", "",
        "| Check | Status | Detail |", "|---|---:|---|",
    ]
    for check in report["checks"]:
        detail = check["detail"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {check['name']} | {check['status']} | {detail} |")
    lines.extend(["", "## Artifact hashes", ""])
    if report["artifacts"]:
        lines.extend(["| Artifact | Bytes | SHA-256 |", "|---|---:|---|"])
        for artifact in report["artifacts"]:
            lines.append(
                f"| `{Path(artifact['path']).name}` | {artifact['bytes']} | "
                f"`{artifact['sha256']}` |"
            )
    else:
        lines.append("No artifacts were supplied to the audit.")
    lines.extend([
        "", "## Required organization actions", "",
        "- Supply the contractual company identity, support contact, and SLA.",
        "- Have legal/compliance approve the dependency license inventory and SBOM.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", type=Path, default=[])
    parser.add_argument("--allow-license-pending", action="store_true")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    report = audit(args.artifact, args.allow_license_pending)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered, "utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(_markdown(report), "utf-8")
    print(rendered, end="")
    if report["summary"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
