import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import shlex

import pytest

SSH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SSH))
loader = importlib.machinery.SourceFileLoader("ssh2_test", str(SSH / "ssh2"))
SPEC = importlib.util.spec_from_loader(loader.name, loader)
MODULE = importlib.util.module_from_spec(SPEC)
loader.exec_module(MODULE)


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for key in MODULE.HOST_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("USG0_IP", "192.0.2.10")
    monkeypatch.setenv("USG1_IP", "192.0.2.11")


@pytest.mark.parametrize(
    "host,ip", [("suzhou-one", "192.0.2.10"), ("user@lianqiuhu-one", "192.0.2.11")]
)
def test_gateway_binding_and_all_proxy_values(host, ip, monkeypatch):
    monkeypatch.setenv("GIT_SSH_COMMAND", "do-not-forward-local-command")
    command = MODULE.build_command(["-p", "2222", host, "true"])
    assert command[:5] == [
        "/mnt/c/Windows/System32/OpenSSH/ssh.exe",
        "-b",
        ip,
        "-p",
        "2222",
    ]
    check = subprocess.run(
        ["sh", "-c", command[-1] + "; env"], capture_output=True, text=True, check=True
    )
    values = dict(
        line.split("=", 1) for line in check.stdout.splitlines() if "=" in line
    )
    assert values["USG_IP"] == ip
    assert values["SSH_HOSTNAME_ALIAS"] == host.rsplit("@", 1)[-1]
    assert values["DEVTOOLS_HTTP_PROXY"] == f"http://{ip}:3128"
    for key in MODULE.PROXY_ENV_KEYS:
        assert values[key] == f"http://{ip}:3128"
    git_ssh = shlex.split(values["GIT_SSH_COMMAND"])
    assert git_ssh[:5] == ["ssh", "-o", "HostName=ssh.github.com", "-p", "443"]
    proxy = next(value for value in git_ssh if value.startswith("ProxyCommand="))
    assert shlex.split(proxy.split("=", 1)[1]) == [
        "ncat",
        "--proxy",
        ip + ":3128",
        "--proxy-type",
        "http",
        "%h",
        "%p",
    ]


def test_native_route_does_not_forward_local_git_command(monkeypatch):
    monkeypatch.setenv("GIT_SSH_COMMAND", "local-only")
    command = MODULE.build_command(["xiaoqiaoling-one", "true"])
    assert "GIT_SSH_COMMAND" not in command[-1]


def test_native_route_and_quoted_environment(monkeypatch):
    value = "test ' quoted $HOME; $(false)\nnext"
    monkeypatch.setenv("OPENAI_API_KEY", value)
    monkeypatch.setenv("https_proxy", "http://native-proxy:3128")
    command = MODULE.build_command(["xiaoqiaoling-one", "true"])
    assert command[:2] == ["/usr/bin/ssh", "xiaoqiaoling-one"]
    check = subprocess.run(
        [
            "sh",
            "-c",
            command[-1]
            + '; python3 -c \'import os,json;print(json.dumps([os.environ["OPENAI_API_KEY"],os.environ["https_proxy"]]))\'',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(check.stdout) == [value, "http://native-proxy:3128"]


@pytest.mark.parametrize(
    "options", [["-s"], ["-N"], ["-o", "BatchMode=yes", "-s"], ["-p2222", "-N"]]
)
def test_non_shell_modes_are_passed_unchanged(options):
    assert MODULE.build_command([*options, "xiaoqiaoling-one", "sftp"]) == [
        "/usr/bin/ssh",
        *options,
        "xiaoqiaoling-one",
        "sftp",
    ]


def test_unknown_host_and_missing_bind_rejected(monkeypatch):
    with pytest.raises(ValueError, match="unsupported"):
        MODULE.build_command(["unknown"])
    monkeypatch.delenv("USG0_IP")
    with pytest.raises(ValueError, match="USG0_IP"):
        MODULE.build_command(["suzhou-one"])
