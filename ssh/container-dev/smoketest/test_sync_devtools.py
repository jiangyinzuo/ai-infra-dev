import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path):
    ssh = tmp_path / "ssh"
    container = ssh / "container-dev"
    tools = container / "devtools"
    tools.mkdir(parents=True)
    shutil.copy2(ROOT / "sync-devtools.py", container / "sync-devtools.py")
    (tools / "openeuler24.03-lts-aarch64").mkdir()
    (tools / "proxy.env").touch()
    binaries = tmp_path / "bin"
    binaries.mkdir()
    rsync = binaries / "rsync"
    rsync.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\n"
        'with open(os.environ["SYNC_LOG"], "a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
    )
    with rsync.open("a") as stream:
        stream.write('sys.exit(int(os.environ.get("SYNC_EXIT", "0")))\n')
    rsync.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "SYNC_LOG": str(tmp_path / "log"),
    }
    return ssh, container, env


def write_machines(ssh, rows):
    (ssh / "machine-info").mkdir(exist_ok=True)
    with (ssh / "machine-info/machines-output.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Host", "OS", "Arch"])
        writer.writerows(rows)


def run(workspace, *args):
    _, container, env = workspace
    return subprocess.run(
        ["python3", str(container / "sync-devtools.py"), *args],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("mode", ["--apply", "--dry-run"])
def test_host_selects_platform_from_csv(workspace, mode):
    ssh, container, env = workspace
    write_machines(
        ssh,
        [
            ("suzhou-a3-130", "openEuler 24.03 (LTS)", "aarch64"),
            ("lianqiuhu-one", "openEuler 24.03, LTS", "arm64"),
        ],
    )
    result = run(workspace, mode, "suzhou-a3-130", "lianqiuhu-one")
    assert result.returncode == 0, result.stderr
    calls = [
        json.loads(line) for line in Path(env["SYNC_LOG"]).read_text().splitlines()
    ]
    assert len(calls) == 2
    for args, host in zip(calls, ("suzhou-a3-130", "lianqiuhu-one")):
        assert str(container / "devtools/openeuler24.03-lts-aarch64") in args
        assert args[-1] == f"jiangyinzuo@{host}:~/devtools/"
        assert ("--dry-run" in args) == (mode == "--dry-run")


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [("other", "openEuler", "aarch64")],
        [("bad", "Ubuntu", "aarch64")],
        [("bad", "openEuler", "x86_64")],
    ],
)
def test_invalid_host_or_platform_stops_before_any_transfer(workspace, rows):
    ssh, _, env = workspace
    write_machines(ssh, [("good", "openEuler", "aarch64"), *rows])
    result = run(workspace, "--apply", "good", "bad")
    assert result.returncode != 0
    assert "bad" in result.stderr
    assert not Path(env["SYNC_LOG"]).exists()


def test_rsync_failure_stops_remaining_hosts(workspace):
    ssh, _, env = workspace
    write_machines(
        ssh, [("one", "openEuler", "aarch64"), ("two", "openEuler", "arm64")]
    )
    env["SYNC_EXIT"] = "23"
    result = run(workspace, "--apply", "one", "two")
    assert result.returncode == 23
    assert len(Path(env["SYNC_LOG"]).read_text().splitlines()) == 1
