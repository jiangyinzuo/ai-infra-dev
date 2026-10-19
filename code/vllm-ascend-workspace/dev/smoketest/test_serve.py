import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def service(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "serve.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("serve_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "state_directory", lambda: tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "serve_resources",
        SimpleNamespace(
            choose_model=lambda value: "/model", choose_devices=lambda value: "0"
        ),
    )
    binary = tmp_path / "vllm"
    binary.write_text(
        "#!"
        + sys.executable
        + '\nimport os,sys,time\nassert sys.argv[1:] == ["serve", "/model", "--max-model-len", "1024"]\nprint("service-log " + os.environ["ASCEND_RT_VISIBLE_DEVICES"],flush=True)\ntime.sleep(120)\n'
    )
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + ":" + os.environ["PATH"])
    (tmp_path / "launch.py").write_text(
        'from serve_config import ServeConfig\nconfig = ServeConfig(1, ["vllm", "serve", "{model}", "--max-model-len", "1024"])\n'
    )
    monkeypatch.chdir(tmp_path)
    yield module
    state = tmp_path / "service.json"
    if state.exists():
        row = json.loads(state.read_text())
        if module.running(row):
            os.killpg(row["pid"], signal.SIGTERM)


def options(action=None):
    return SimpleNamespace(
        serve_action=action, model=None, devices=None, script="launch.py"
    )


def test_start_log_decline_restart_and_stop(service, monkeypatch, tmp_path):
    assert service.execute(options()) == 0
    state = json.loads((tmp_path / "service.json").read_text())
    assert service.running(state)
    assert "service-log 0" in Path(state["log"]).read_text()
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert service.execute(options()) == 0
    assert json.loads((tmp_path / "service.json").read_text())["pid"] == state["pid"]
    assert service.execute(options("stop")) == 0
    assert not service.running(state)
    assert not (tmp_path / "service.json").exists()
    os.waitpid(state["pid"], 0)


def test_stale_pid_identity_does_not_kill_unrelated_process(service, tmp_path):
    state = {"pid": os.getpid(), "identity": "wrong-start-time"}
    (tmp_path / "service.json").write_text(json.dumps(state))
    assert service.execute(options("stop")) == 0
    assert service.process_identity(os.getpid()) is not None


def test_confirmed_restart_replaces_process(service, monkeypatch, tmp_path):
    assert service.execute(options()) == 0
    first = json.loads((tmp_path / "service.json").read_text())
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert service.execute(options()) == 0
    second = json.loads((tmp_path / "service.json").read_text())
    assert second["pid"] != first["pid"]
    assert not service.running(first)
    assert service.running(second)
    service.execute(options("stop"))
    try:
        os.waitpid(first["pid"], 0)
    except ChildProcessError:
        pass  # Popen may already reap the old child when launching its replacement.
    os.waitpid(second["pid"], 0)


def test_early_launch_failure_reports_log_and_clears_state(service, tmp_path):
    (tmp_path / "vllm").write_text("#!/bin/sh\necho startup-failure >&2\nexit 7\n")
    with pytest.raises(RuntimeError, match="日志"):
        service.execute(options())
    assert not (tmp_path / "service.json").exists()
    assert any("startup-failure" in path.read_text() for path in tmp_path.glob("*.log"))


def test_device_count_mismatch_never_launches(service, monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "serve_resources",
        SimpleNamespace(
            choose_model=lambda value: "/model", choose_devices=lambda value: "0,1"
        ),
    )
    with pytest.raises(ValueError, match="device"):
        service.execute(options())
    assert not (tmp_path / "service.json").exists()
    assert not list(tmp_path.glob("*.log"))


def test_bash_configuration_rejected(service, tmp_path):
    path = tmp_path / "legacy.sh"
    path.write_text("exec vllm serve /model\n")
    with pytest.raises(ValueError, match="Python"):
        service.load_config(path)
