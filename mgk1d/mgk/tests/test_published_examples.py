import json
import runpy
from pathlib import Path

from mgk.internal.configuration import build_config


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = ROOT / "mgk" / "examples" / "validation"


def test_published_case_builders_produce_valid_cpu_configs():
    namespace = runpy.run_path(str(EXAMPLE_DIR / "published_cases.py"))
    for name, builder in namespace["BUILDERS"].items():
        supplied, normalized, _ = build_config(builder("cpu"))
        assert normalized.useGpu is False, name
        assert normalized.aparallel is False, name
        assert normalized.bparallel is False, name
        assert supplied.physical.electronBeta == 0, name


def test_published_reference_results_are_machine_readable():
    path = EXAMPLE_DIR / "published_reference_results.json"
    data = json.loads(path.read_text("utf-8"))
    assert data["source"]["arxiv"] == "2608.17418"
    assert "paperCheckpoints" in data
    assert "productReproduction2026_08_20" in data
