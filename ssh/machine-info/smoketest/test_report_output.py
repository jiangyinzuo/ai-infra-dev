import csv
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "machine_info_output", Path(__file__).resolve().parents[1] / "machine-info.py"
)
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def machine(host, os_name):
    return {
        "host": host,
        "group": "lab",
        "arch": "aarch64",
        "os": {"pretty_name": os_name, "errors": []},
        "kubernetes": {
            "status": "not-detected",
            "versions": {},
            "services": {},
            "errors": [],
        },
        "storage": {"directories": [], "errors": []},
    }


def report(machines):
    return {
        "collected_at": "2026-10-14",
        "directory_glob": "/data*",
        "machines": machines,
    }


def test_partial_reports_upsert_hosts_and_preserve_previous_os(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "machine-info"
    output_dir.mkdir()
    monkeypatch.setattr(CLI, "SCRIPT_DIR", output_dir)
    CLI.write_reports(report([machine("one", "Old OS"), machine("two", "Keep OS")]))
    CLI.write_reports(
        report(
            [
                machine("one", 'Linux, "新版本"'),
                machine("three", "New OS"),
                {"host": "two", "group": "lab", "error": "unreachable"},
            ]
        )
    )
    CLI.write_reports(report([machine("three", "Updated OS")]))
    with (output_dir / "machines-output.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["Host", "OS", "Arch"]
        assert list(reader) == [
            {"Host": "one", "OS": 'Linux, "新版本"', "Arch": "aarch64"},
            {"Host": "three", "OS": "Updated OS", "Arch": "aarch64"},
            {"Host": "two", "OS": "Keep OS", "Arch": "aarch64"},
        ]
    assert not (tmp_path / "machines-output.md").exists()
    assert not (tmp_path / "machines-output.csv").exists()
    text = (output_dir / "machines-output.md").read_text()
    assert "Updated OS" in text
    assert "Keep OS" not in text
    assert 'Linux, "新版本"' not in text
