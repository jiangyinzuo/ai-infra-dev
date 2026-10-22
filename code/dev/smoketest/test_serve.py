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
    monkeypatch.setattr(module, "state_directory", lambda resource: tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "serve_resources",
        SimpleNamespace(
            choose_model=lambda value: "/model", choose_devices=lambda value, resource: "0",
            device_visibility_variable=lambda resource: "ASCEND_RT_VISIBLE_DEVICES",
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
    (tmp_path / "serve_config").mkdir()
    monkeypatch.setattr(module, "__file__", str(tmp_path / "serve.py"))
    (tmp_path / "serve_config/launch.py").write_text(
        'from serve_config import ServeConfig\nconfig = ServeConfig(1, ["vllm", "serve", "{model}", "--max-model-len", "1024"])\n'
    )
    monkeypatch.chdir(tmp_path)
    yield module
    state = tmp_path / "service.json"
    if state.exists():
        row = json.loads(state.read_text())
        if module.running(row):
            os.killpg(row["pid"], signal.SIGTERM)


def options(action=None, resource="vllm-ascend"):
    return SimpleNamespace(
        resource=resource, serve_action=action, model=None, devices=None, script="launch.py"
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
            choose_model=lambda value: "/model", choose_devices=lambda value, resource: "0,1",
            device_visibility_variable=lambda resource: "ASCEND_RT_VISIBLE_DEVICES",
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


def test_gpu_launch_uses_config_model_and_cuda_visibility(service, monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "serve_resources",
        SimpleNamespace(
            choose_model=lambda value: pytest.fail("default model prompted"),
            choose_devices=lambda value, resource: "GPU-selected",
            device_visibility_variable=lambda resource: "CUDA_VISIBLE_DEVICES",
        ),
    )
    (tmp_path / "serve_config/launch.py").write_text(
        'from serve_config import ServeConfig\n'
        'config = ServeConfig(1, ["vllm", "serve", "{model}"], model="deepseek-ai/DeepSeek-V3.2")\n'
    )
    (tmp_path / "vllm").write_text(
        "#!" + sys.executable + "\n"
        "import os,sys,time\n"
        'assert sys.argv[1:] == ["serve", "deepseek-ai/DeepSeek-V3.2"]\n'
        'assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-selected"\n'
        'assert "ASCEND_RT_VISIBLE_DEVICES" not in os.environ\n'
        'print("gpu-ready", flush=True)\n'
        'time.sleep(120)\n'
    )
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    assert service.execute(options(resource="vllm")) == 0
    state = json.loads((tmp_path / "service.json").read_text())
    assert "gpu-ready" in Path(state["log"]).read_text()
    service.execute(options("stop", resource="vllm"))
    os.waitpid(state["pid"], 0)


def test_platform_stop_leaves_other_service_running(service, monkeypatch, tmp_path):
    def directory(resource):
        path = tmp_path / "states" / resource
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(service, "state_directory", directory)
    states = {}
    try:
        for resource in ("vllm", "vllm-ascend"):
            service.execute(options(resource=resource))
            states[resource] = json.loads((directory(resource) / "service.json").read_text())
        service.execute(options("stop", resource="vllm"))
        assert not service.running(states["vllm"])
        assert service.running(states["vllm-ascend"])
        assert states["vllm"]["log"] != states["vllm-ascend"]["log"]
    finally:
        for resource, state in states.items():
            service.execute(options("stop", resource=resource))
            os.waitpid(state["pid"], 0)


def test_status_reports_lifecycle_without_mutating_state(service, tmp_path, capsys):
    query = options("status")
    query.script = None
    assert service.execute(query) == 0
    assert "当前没有" in capsys.readouterr().out
    assert not (tmp_path / "service.json").exists()

    service.execute(options())
    state_file = tmp_path / "service.json"
    original = state_file.read_text()
    state = json.loads(original)
    capsys.readouterr()
    assert service.execute(query) == 0
    output = capsys.readouterr().out
    assert "运行中" in output
    assert str(state["pid"]) in output
    assert state["log"] in output
    assert state_file.read_text() == original
    assert service.running(state)

    service.stop(state)
    os.waitpid(state["pid"], 0)
    assert service.execute(query) == 0
    assert "已退出" in capsys.readouterr().out
    assert state_file.read_text() == original


def test_status_rejects_reused_pid_identity(service, tmp_path, capsys):
    state = {"pid": os.getpid(), "identity": "wrong-start-time", "model": "/model", "devices": "0", "script": "launch.py", "log": "/tmp/log"}
    state_file = tmp_path / "service.json"
    state_file.write_text(json.dumps(state))
    original = state_file.read_text()
    assert service.execute(options("status")) == 0
    assert "已退出" in capsys.readouterr().out
    assert state_file.read_text() == original
