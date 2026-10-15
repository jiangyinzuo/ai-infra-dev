"""Offline smoke tests: python3 -m pytest -q ssh/machine-info/smoketest"""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import shutil
import sys
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLI = load_module("machine_info", "machine-info.py")
PROBE = load_module("machine_probe", "../container-dev/devtools/machine_probe.py")


@pytest.mark.parametrize(
    "device,expected", [("0xD803", "a3"), ("0xD802", "a2"), ("0xFFFF", "unknown")]
)
def test_npu_platform_from_board_pci_device_id(device, expected):
    text = (
        (ROOT / "smoketest/fixtures/npu-board-a3.txt")
        .read_text()
        .replace("0xD803", device)
    )
    with (
        patch.object(PROBE.shutil, "which", return_value="/bin/npu-smi"),
        patch.object(
            PROBE,
            "command",
            return_value={"returncode": 0, "stdout": text, "stderr": ""},
        ),
    ):
        result = PROBE.npu_platform()
    assert result["platform"] == expected
    assert result["models"] == []
    assert result["status"] == "ok"
    assert result["errors"] == []
    assert result["evidence"] == ["PCI Device ID: " + device.lower()]


def test_missing_npu_smi_is_not_collection_failure():
    with (
        patch.object(PROBE.shutil, "which", return_value=None),
        patch.object(PROBE, "command") as command,
    ):
        result = PROBE.npu_platform()
    command.assert_not_called()
    assert result == {
        "platform": "unknown",
        "models": [],
        "evidence": [],
        "status": "not-found",
        "errors": [],
    }


@pytest.mark.parametrize(
    "response",
    [
        {"returncode": 1, "stdout": "| 0 910C | OK |", "stderr": "driver unavailable"},
        {"returncode": None, "stdout": "", "stderr": "timed out"},
        {"returncode": 0, "stdout": "npu-smi version 910C; hostname a3", "stderr": ""},
    ],
)
def test_npu_failure_does_not_infer_platform(response):
    with (
        patch.object(PROBE.shutil, "which", return_value="npu-smi"),
        patch.object(PROBE, "command", return_value=response),
    ):
        result = PROBE.npu_platform()
    assert result["status"] == "failed"
    assert result["platform"] == "unknown"
    assert result["models"] == []
    assert result["errors"]


@pytest.mark.parametrize(
    "value,expected",
    [
        ('"openEuler 22.03 (LTS-SP4)"', "openEuler 22.03 (LTS-SP4)"),
        ("'Ubuntu 24.04 LTS'", "Ubuntu 24.04 LTS"),
        ('"Linux #1 $(not-executed)"', "Linux #1 $(not-executed)"),
        ("Linux", "Linux"),
    ],
)
def test_os_pretty_name_preserves_release_text(tmp_path, value, expected):
    release = tmp_path / "os-release"
    release.write_text(f'NAME="Other"\nPRETTY_NAME={value}\nVERSION_ID="22.03"\n')
    assert PROBE.operating_system(release) == {"pretty_name": expected, "errors": []}


@pytest.mark.parametrize(
    "text", [None, 'NAME="Linux"\n', 'PRETTY_NAME=""\n', 'PRETTY_NAME="unclosed\n']
)
def test_missing_or_invalid_os_release_reports_unknown(tmp_path, text):
    release = tmp_path / "os-release"
    if text is not None:
        release.write_text(text)
    result = PROBE.operating_system(release)
    assert result["pretty_name"] is None
    assert result["errors"]


def test_inventory_filters_concrete_hosts_and_rejects_empty_selection(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".ssh").mkdir()
    config = tmp_path / ".ssh/config"
    config.write_text(
        "Host lab-b lab-a\n User root\n# Host lab-disabled\nHost lab-*\n User root\nHost elsewhere\n User root\n"
    )
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({"groups": {"lab": "wsl", "lianqiuhu": "wsl"}}))
    import ssh_hosts

    monkeypatch.setattr(ssh_hosts, "DEFAULT_PROFILES", profiles)
    args = argparse.Namespace(group=["lab"], host_glob=["*-a"])
    assert [target["host"] for target in CLI.load_targets(args)] == ["lab-a"]
    args.group = None
    args.host_glob = None
    assert [target["host"] for target in CLI.load_targets(args)] == ["lab-a", "lab-b"]
    args.host_glob = ["missing-*"]
    with pytest.raises(ValueError, match="No hosts selected"):
        CLI.load_targets(args)
    args.group = ["unknown"]
    with pytest.raises(ValueError, match="Unknown groups"):
        CLI.load_targets(args)


def test_npu_inventory_cli_preserves_each_profile_connection(tmp_path, monkeypatch):
    config = tmp_path / "ssh-config"
    config.write_text("Host local-one remote-two\n User root\n")
    profiles = tmp_path / "hosts.json"
    profiles.write_text(json.dumps({"groups": {"local": "wsl", "remote": "windows"}}))
    import ssh_hosts
    from dataclasses import replace

    monkeypatch.setattr(ssh_hosts, "DEFAULT_PROFILES", profiles)

    monkeypatch.setitem(
        ssh_hosts.SSH_SOURCES,
        "wsl",
        replace(
            ssh_hosts.SSH_SOURCES["wsl"], config=str(config), ssh_config=str(config)
        ),
    )
    monkeypatch.setattr(
        ssh_hosts,
        "windows_paths",
        lambda: {
            "WINDOWS_SSH_CONFIG": str(config),
            "WINDOWS_SSH_CONFIG_NATIVE": "C:\\Users\\example user\\.ssh\\config",
            "WINDOWS_SSH_HOME": str(tmp_path),
        },
    )
    targets = ssh_hosts.load_targets(argparse.Namespace(group=None, host_glob=None))
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            ssh_hosts.build_inventory(targets, inventory_group_name="npu_servers")
        )
    )
    resolved = subprocess.run(
        ["ansible-inventory", "-i", str(inventory), "--list"],
        capture_output=True,
        text=True,
        check=True,
    )
    hosts = json.loads(resolved.stdout)["_meta"]["hostvars"]
    assert set(hosts) == {"local-one", "remote-two"}
    for host in ("local-one", "remote-two"):
        assert hosts[host]["ansible_ssh_executable"] == str(ROOT.parent / "ssh2")
    import shlex

    assert (
        shlex.split(hosts["remote-two"]["ansible_ssh_args"])[1]
        == "C:\\Users\\example user\\.ssh\\config"
    )


def test_directory_capacity_resolves_links_and_excludes_files(tmp_path):
    directory = tmp_path / "data actual"
    directory.mkdir()
    link = tmp_path / "data-link"
    link.symlink_to(directory, target_is_directory=True)
    (tmp_path / "data-file").write_text("not a directory")
    (tmp_path / "data-broken").symlink_to(tmp_path / "missing")
    result = PROBE.data_directories(str(tmp_path / "data*"))
    entries = {entry["path"]: entry for entry in result["directories"]}
    assert not result["errors"]
    assert set(entries) == {str(directory), str(link)}
    assert entries[str(link)]["symlink"] is True
    assert entries[str(link)]["real_path"] == str(directory)
    df = subprocess.run(
        ["df", "-B1", "--output=size,used", "--", str(directory)],
        capture_output=True,
        text=True,
        check=True,
    )
    total, _ = map(int, df.stdout.splitlines()[1].split())
    assert entries[str(directory)]["total_bytes"] == total
    assert entries[str(link)]["filesystem"] == entries[str(directory)]["filesystem"]
    assert entries[str(directory)]["used_bytes"] <= total
    assert PROBE.data_directories(str(tmp_path / "absent*")) == {
        "directories": [],
        "errors": [],
    }


@pytest.mark.parametrize(
    "failure",
    [
        {"unreachable": True, "msg": "Permission denied"},
        {"failed": True, "msg": "probe timed out"},
        {"rc": 1, "stderr": "Python unavailable"},
        {"rc": 0, "stdout": "login banner only"},
        {"rc": 0, "stdout": "__MACHINE_INFO__{}\n__MACHINE_INFO__{}\n"},
    ],
)
def test_transport_errors_are_host_results_not_batch_exceptions(failure):
    result = CLI.parse_result({"group": "lab", "host": "offline"}, failure)
    assert result["host"] == "offline"
    assert "error" in result
    assert "kubernetes" not in result


def test_service_detection_distinguishes_stopped_nodes_clients_and_unknown():
    def service_result(args):
        state = "loaded" if args[2] == "kubelet.service" else "not-found"
        return {
            "returncode": 0,
            "stdout": f"LoadState={state}\nActiveState=inactive\nSubState=dead",
            "stderr": "",
        }

    with (
        patch.object(PROBE.shutil, "which", return_value=None),
        patch.object(PROBE.Path, "exists", return_value=False),
        patch.object(PROBE.Path, "is_file", return_value=False),
        patch.object(PROBE, "command", side_effect=service_result),
    ):
        node = PROBE.kubernetes()
        assert node["status"] == "installed"
        assert node["services"]["kubelet"]["ActiveState"] == "inactive"
        absent = {
            "returncode": 0,
            "stdout": "LoadState=not-found\nActiveState=inactive\nSubState=dead",
            "stderr": "",
        }
        with patch.object(PROBE, "command", return_value=absent):
            assert PROBE.kubernetes()["status"] == "not-detected"
        with patch.object(
            PROBE,
            "command",
            return_value={
                "returncode": 1,
                "stdout": "",
                "stderr": "systemd unavailable",
            },
        ):
            assert PROBE.kubernetes()["status"] == "unknown"
        with (
            patch.object(
                PROBE.shutil,
                "which",
                side_effect=lambda name: (
                    "/usr/bin/kubectl" if name == "kubectl" else None
                ),
            ),
            patch.object(
                PROBE,
                "command",
                side_effect=lambda args: (
                    {
                        "returncode": 0,
                        "stdout": '{"clientVersion":{"gitVersion":"v1.34.9"}}',
                        "stderr": "",
                    }
                    if args[0] == "/usr/bin/kubectl"
                    else absent
                ),
            ),
        ):
            client = PROBE.kubernetes()
            assert client["status"] == "client-only"
            assert client["versions"]["kubectl"] == "v1.34.9"


def test_playbook_collects_local_capacity_without_ssh(tmp_path):
    """Run the real playbook using Ansible's local transport, no servers or USG."""
    if shutil.which("ansible-playbook") is None:
        pytest.skip("ansible-playbook is required for the local integration smoke")
    data = tmp_path / "data-local"
    data.mkdir()
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "all": {
                    "hosts": {
                        "smoke-local": {
                            "ansible_connection": "local",
                            "machine_python": sys.executable,
                        }
                    }
                }
            }
        )
    )
    output = tmp_path / "results.json"
    variables = tmp_path / "vars.json"
    variables.write_text(
        json.dumps(
            {
                "results_path": str(output),
                "directory_glob": str(tmp_path / "data*"),
                "probe_timeout": 30,
            }
        )
    )
    execution = subprocess.run(
        [
            "ansible-playbook",
            "-i",
            str(inventory),
            str(ROOT / "machine-info.yml"),
            "-e",
            "@" + str(variables),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    assert execution.returncode == 0, execution.stdout + execution.stderr
    result = CLI.parse_result(
        {"host": "smoke-local", "group": "smoke"},
        json.loads(output.read_text())["smoke-local"],
    )
    assert "error" not in result
    directories = result["storage"]["directories"]
    assert [entry["path"] for entry in directories] == [str(data)]
    expected = subprocess.run(
        ["df", "-B1", "--output=size", "--", str(data)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert directories[0]["total_bytes"] == int(expected.stdout.splitlines()[1])
