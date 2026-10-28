import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import hf_speedtest


def test_fastest_excludes_failed_probe(monkeypatch):
    monkeypatch.setattr(
        hf_speedtest,
        "speedtest",
        lambda endpoints: [
            {"endpoint": "https://failed", "status": "FAILED", "overall_MiB_s": 100},
            {"endpoint": "https://slow", "status": "OK", "overall_MiB_s": 1},
            {"endpoint": "https://fast", "status": "OK", "overall_MiB_s": 3},
        ],
    )
    assert hf_speedtest.fastest_endpoint() == "https://fast"


def test_all_failed_does_not_select_endpoint(monkeypatch):
    monkeypatch.setattr(
        hf_speedtest,
        "speedtest",
        lambda endpoints: [
            {"endpoint": "https://broken", "status": "FAILED", "error": "timeout"},
        ],
    )
    with pytest.raises(RuntimeError, match="timeout"):
        hf_speedtest.fastest_endpoint()
