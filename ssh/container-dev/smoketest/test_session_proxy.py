import importlib.machinery
import importlib.util
import os
from pathlib import Path
import subprocess


SSH = Path(__file__).resolve().parents[2]


def test_session_proxy_survives_system_override(monkeypatch):
    loader = importlib.machinery.SourceFileLoader(
        "session_proxy_ssh2", str(SSH / "ssh2")
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    monkeypatch.setenv("http_proxy", "http://other-proxy:8080")
    monkeypatch.setenv("DEVTOOLS_HTTP_PROXY", "http://session-proxy:3128")
    command = module.build_command(["xiaoqiaoling-one", "true"])[-1]
    result = subprocess.run(
        [
            "bash",
            "-c",
            command
            + '; export http_proxy=http://system:8080; source "$1"; printf "%s\\n" "$http_proxy" "$https_proxy" "$HTTP_PROXY" "$HTTPS_PROXY"',
            "bash",
            str(SSH / "createuser/session-proxy.bash"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == ["http://session-proxy:3128"] * 4


def test_missing_session_proxy_preserves_system_configuration():
    environment = dict(os.environ, http_proxy="http://system:8080")
    environment.pop("DEVTOOLS_HTTP_PROXY", None)
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s" "$http_proxy"',
            "bash",
            str(SSH / "createuser/session-proxy.bash"),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "http://system:8080"
