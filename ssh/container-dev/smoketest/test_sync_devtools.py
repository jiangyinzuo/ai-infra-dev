import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path):
    ssh = tmp_path / "ssh"
    container = ssh / "container-dev"
    tools = container / "devtools"
    tools.mkdir(parents=True)
    for source in (ROOT / "devtools").glob("*.py"):
        shutil.copy2(source, tools / source.name)
    for package in ("workflow", "containers"):
        shutil.copytree(
            ROOT / "devtools" / package,
            tools / package,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.tar", "*.tar.gz", "*.tar.zst"
            ),
        )
    shutil.copytree(ROOT.parent / "ssh_common", ssh / "ssh_common")
    data = tmp_path / "resource data"
    (data / "openeuler24.03-lts-aarch64").mkdir(parents=True)
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text(
        f"import sys; sys.path.insert(0, {str(ssh)!r})\n"
        "import ssh_common.remote\nssh_common.remote.local_commit = lambda: 'a' * 40\n"
    )
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
        "DEVTOOLS_DATA_DIR": str(data),
        "PYTHONPATH": str(hooks),
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
        [
            "python3",
            str(container / "devtools/cli.py"),
            "--no-hints",
            "binaries",
            "sync",
            *args,
        ],
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
    result = run(workspace, mode, "--host", "suzhou-a3-130", "--host", "lianqiuhu-one")
    assert result.returncode == 0, result.stderr
    calls = [
        json.loads(line) for line in Path(env["SYNC_LOG"]).read_text().splitlines()
    ]
    assert len(calls) == 2
    for args, host in zip(calls, ("suzhou-a3-130", "lianqiuhu-one")):
        data = Path(env["DEVTOOLS_DATA_DIR"])
        assert str(data / "openeuler24.03-lts-aarch64") in args
        assert (
            args[-1] == f"jiangyinzuo@{host}:~/ai-infra-dev/ssh/container-dev/devtools/"
        )
        assert ("--dry-run" in args) == (mode == "--dry-run")
        assert str(container / "devtools/workflow") not in args
        assert str(container / "devtools/containers") not in args
        assert str(container / "devtools/cli.py") not in args
        assert str(data / "nvim-plugins") in args
        assert "--include=/nvim-plugins/archive-hashes.json" in args
        assert "--include=/openeuler24.03-lts-aarch64/*.gz" in args
        assert "--include=/openeuler24.03-lts-aarch64/fzf-*-support/*.sh" in args
        assert "--include=/openeuler24.03-lts-aarch64/omp" in args
        assert "--exclude=*" in args
        assert not any(arg.startswith("--delete") for arg in args)
        guard = args[args.index("--rsync-path") + 1]
        assert "a" * 40 in guard
        assert "git-repo update" in guard


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
    result = run(workspace, "--apply", "--host", "good", "--host", "bad")
    assert result.returncode != 0
    assert "bad" in result.stderr
    assert not Path(env["SYNC_LOG"]).exists()


def test_rsync_failure_stops_remaining_hosts(workspace):
    ssh, _, env = workspace
    write_machines(
        ssh, [("one", "openEuler", "aarch64"), ("two", "openEuler", "arm64")]
    )
    env["SYNC_EXIT"] = "23"
    result = run(workspace, "--apply", "--host", "one", "--host", "two")
    assert result.returncode == 23
    assert len(Path(env["SYNC_LOG"]).read_text().splitlines()) == 1


@pytest.mark.parametrize("kernel", ["Linux", "microsoft-WSL2"])
@pytest.mark.parametrize("cwd_kind", ["repository", "home", "unrelated"])
def test_default_resources_do_not_depend_on_cwd(workspace, tmp_path, cwd_kind, kernel):
    ssh, container, original_env = workspace
    write_machines(ssh, [("suzhou-one", "openEuler", "aarch64")])
    env = dict(original_env)
    env.pop("DEVTOOLS_DATA_DIR")
    home = tmp_path / "home"
    home.mkdir()
    env["HOME"] = str(home)
    hooks = Path(env["PYTHONPATH"]) / "sitecustomize.py"
    with hooks.open("a") as stream:
        stream.write(f"import platform\nplatform.release = lambda: {kernel!r}\n")
    unrelated = tmp_path / "unrelated directory"
    unrelated.mkdir()
    cwd = {"repository": container, "home": home, "unrelated": unrelated}[cwd_kind]
    source = container / "devtools/openeuler24.03-lts-aarch64"
    (source / "sample.tar.gz").write_bytes(b"archive")
    result = subprocess.run(
        [
            sys.executable,
            str(container / "devtools/cli.py"),
            "--no-hints",
            "binaries",
            "sync",
            "--host",
            "suzhou-one",
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    command = json.loads(Path(env["SYNC_LOG"]).read_text().splitlines()[0])
    assert str(source) in command
    assert str(container / "devtools/nvim-plugins") in command
    assert not (home / "devtools").exists()
