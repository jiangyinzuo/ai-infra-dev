from pathlib import Path
import sys

import pytest

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2]),
    str(Path(__file__).resolve().parents[1] / "devtools"),
]
from containers.core.bindfs_processes import scan_bindfs
from containers.core import bindfs_limits


def test_scanner_filters_identity_and_strict_decode_errors(tmp_path):
    for pid, args in [
        (
            10,
            [
                "bindfs",
                "--create-for-user=1000",
                "--create-for-group=1001",
                "/src",
                "/view",
            ],
        ),
        (
            11,
            [
                "bindfs",
                "--create-for-user=999",
                "--create-for-group=1001",
                "/src",
                "/view",
            ],
        ),
    ]:
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "cmdline").write_bytes("\0".join(args).encode() + b"\0")
    assert list(scan_bindfs(1000, 1001, proc=tmp_path)) == [
        {"pid": 10, "source": "/src", "target": "/view"}
    ]
    broken = tmp_path / "12"
    broken.mkdir()
    (broken / "cmdline").write_bytes(b"\xff")
    assert len(list(scan_bindfs(1000, 1001, proc=tmp_path))) == 1
    with pytest.raises(UnicodeError):
        list(scan_bindfs(1000, 1001, strict=True, proc=tmp_path))


def test_repair_requires_exact_source_and_target(monkeypatch):
    monkeypatch.setattr(
        bindfs_limits,
        "scan_bindfs",
        lambda uid, gid, strict: iter(
            [
                {"pid": 10, "source": "/other", "target": "/view"},
                {"pid": 11, "source": "/src", "target": "/other"},
                {"pid": 12, "source": "/src", "target": "/view"},
            ]
        ),
    )
    changed = []
    monkeypatch.setattr(bindfs_limits, "raise_limit", changed.append)
    bindfs_limits.repair("/view", "/src", "1000", "1001")
    assert changed == [12]
