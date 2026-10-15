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
        "npu": {
            "platform": "unknown",
            "models": [],
            "evidence": [],
            "status": "not-found",
            "errors": [],
        },
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
        assert reader.fieldnames == ["Host", "OS", "Arch", "NPUPlatform"]
        assert list(reader) == [
            {
                "Host": "one",
                "OS": 'Linux, "新版本"',
                "Arch": "aarch64",
                "NPUPlatform": "unknown",
            },
            {
                "Host": "three",
                "OS": "Updated OS",
                "Arch": "aarch64",
                "NPUPlatform": "unknown",
            },
            {
                "Host": "two",
                "OS": "Keep OS",
                "Arch": "aarch64",
                "NPUPlatform": "unknown",
            },
        ]
    assert not (tmp_path / "machines-output.md").exists()
    assert not (tmp_path / "machines-output.csv").exists()
    text = (output_dir / "machines-output.md").read_text()
    assert "Updated OS" in text
    assert "Keep OS" not in text
    assert 'Linux, "新版本"' not in text


def test_legacy_csv_upgrade_retains_unqueried_and_failed_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(CLI, "SCRIPT_DIR", tmp_path)
    path = tmp_path / "machines-output.csv"
    path.write_text('Host,OS,Arch\nold,"OS, legacy",x86_64\nfailed,Keep,aarch64\n')
    CLI.write_reports(report([{"host": "failed", "group": "lab", "error": "offline"}]))
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["Host", "OS", "Arch", "NPUPlatform"]
        assert list(reader) == [
            {
                "Host": "failed",
                "OS": "Keep",
                "Arch": "aarch64",
                "NPUPlatform": "unknown",
            },
            {
                "Host": "old",
                "OS": "OS, legacy",
                "Arch": "x86_64",
                "NPUPlatform": "unknown",
            },
        ]


def test_npu_merge_preserves_known_on_failure_and_updates_supported_unknown(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(CLI, "SCRIPT_DIR", tmp_path)
    record = machine("one", "Linux")
    record["npu"] = {
        "platform": "a3",
        "models": ["Ascend 910C"],
        "evidence": ["| 0 Ascend 910C | OK |"],
        "status": "ok",
        "errors": [],
    }
    CLI.write_reports(report([record]))
    text = (tmp_path / "machines-output.md").read_text()
    assert "Ascend 910C" in text
    assert "\\| 0 Ascend 910C \\| OK \\|" in text
    for status in ("failed", "not-found"):
        record["npu"].update(platform="unknown", status=status)
        record["os"]["pretty_name"] = None
        record["arch"] = None
        CLI.write_reports(report([record]))
        with (tmp_path / "machines-output.csv").open(newline="") as stream:
            assert list(csv.DictReader(stream)) == [
                {"Host": "one", "OS": "Linux", "Arch": "aarch64", "NPUPlatform": "a3"}
            ]
    record["npu"].update(platform="unknown", status="ok", models=["910B"])
    CLI.write_reports(report([record]))
    with (tmp_path / "machines-output.csv").open(newline="") as stream:
        assert next(csv.DictReader(stream))["NPUPlatform"] == "unknown"
