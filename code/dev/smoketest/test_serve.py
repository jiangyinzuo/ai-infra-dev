import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
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
            choose_model=lambda value: "/model", choose_devices=lambda value, resource, count: "0",
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
            choose_model=lambda value: "/model", choose_devices=lambda value, resource, count: "0,1",
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
            choose_devices=lambda value, resource, count: "GPU-selected",
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


@pytest.mark.parametrize("child_failure", [False, True], ids=["stop", "child-exit"])
def test_pd_shutdown_reaps_other_services_and_orphan_workers(
    service, tmp_path, child_failure
):
    """停止部署或任一服务退出后，不能遗留独立进程组内的 GPU worker。"""
    launcher = (
        Path(__file__).resolve().parents[1]
        / "serve_config/deepseek-v3.2-hisparse-pd.py"
    )
    child = tmp_path / "child.py"
    child.write_text('''
import os
from pathlib import Path
import subprocess
import sys
import time

role = sys.argv[1]
if role == "prefill":
    worker = subprocess.Popen([sys.executable, "-c", """
import os
from pathlib import Path
import signal
import time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path("worker.ready").write_text(str(os.getpid()))
time.sleep(120)
"""])
    while not Path("worker.ready").exists():
        time.sleep(0.01)
Path(role + ".ready").write_text(str(os.getpid()))
while True:
    if role == "decode" and Path("fail").exists():
        sys.exit(7)
    time.sleep(0.01)
''')
    driver = tmp_path / "driver.py"
    driver.write_text(f'''
from pathlib import Path
import runpy
import sys
module = runpy.run_path({str(launcher)!r})
commands = {{role: [sys.executable, {str(child)!r}, role]
            for role in ("prefill", "decode", "proxy")}}
sys.exit(module["run_services"](commands, ["GPU-P", "GPU-D"], Path.cwd()))
''')
    environment = dict(os.environ, PYTHONPATH=str(launcher.parent.parent))
    ready = [tmp_path / (role + ".ready") for role in ("prefill", "decode", "proxy")]
    worker_ready = tmp_path / "worker.ready"
    with (tmp_path / "supervisor.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(driver)],
            cwd=tmp_path,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        deadline = time.monotonic() + 5
        while not all(path.exists() for path in [*ready, worker_ready]):
            assert process.poll() is None, (tmp_path / "supervisor.log").read_text()
            assert time.monotonic() < deadline, "服务未启动"
            time.sleep(0.01)
        pids = [int(path.read_text()) for path in [*ready, worker_ready]]
        if child_failure:
            (tmp_path / "fail").touch()
        else:
            service.stop({
                "pid": process.pid,
                "identity": service.process_identity(process.pid),
            })
        assert process.wait(timeout=5) == (7 if child_failure else 0)
        deadline = time.monotonic() + 5
        while any(service.process_identity(pid) is not None for pid in pids):
            assert time.monotonic() < deadline, "停止后仍有服务或 worker 存活"
            time.sleep(0.01)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for path in ready:
            if path.exists():
                try:
                    os.killpg(int(path.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_status_checks_child_api_without_restarting_or_mutating_state(
    service, tmp_path, capsys
):
    """子服务进程存活但健康检查失败时，不能报告 API 就绪。"""
    child = tmp_path / "health_server.py"
    child.write_text('''
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import socket
import sys

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if Path("healthy").exists() else 503)
        self.end_headers()
    def log_message(self, *args):
        pass

server = HTTPServer(("127.0.0.1", 0), Handler, bind_and_activate=False)
server.socket.close()
server.socket = socket.socket(fileno=int(sys.argv[1]))
Path("child.ready").touch()
server.serve_forever()
''')
    driver = tmp_path / "supervisor.py"
    driver.write_text(f'''
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen()
port = listener.getsockname()[1]
with Path("prefill.log").open("w") as log:
    child = subprocess.Popen(
        [sys.executable, {str(child)!r}, str(listener.fileno()),
         "--host", "127.0.0.1", "--port", str(port),
         "--kv-transfer-config", json.dumps({{"kv_role": "kv_producer"}})],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-test-prefill"),
        pass_fds=(listener.fileno(),), stdout=log, stderr=subprocess.STDOUT,
    )
listener.close()
Path("child.json").write_text(json.dumps({{"pid": child.pid, "port": port}}))
time.sleep(120)
''')
    process = subprocess.Popen(
        [sys.executable, str(driver)], cwd=tmp_path, start_new_session=True
    )
    state_file = tmp_path / "service.json"
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "child.ready").exists() or not (
            tmp_path / "child.json"
        ).exists():
            assert process.poll() is None
            assert time.monotonic() < deadline, "健康检查服务未启动"
            time.sleep(0.01)
        child_info = json.loads((tmp_path / "child.json").read_text())
        state_file.write_text(json.dumps({
            "pid": process.pid,
            "identity": service.process_identity(process.pid),
            "model": "/model",
            "devices": "GPU-test-prefill,GPU-test-decode",
            "script": "pd.py",
            "log": str(tmp_path / "supervisor.log"),
        }))
        original = state_file.read_bytes()
        assert service.execute(options("status")) == 0
        output = capsys.readouterr().out
        assert "Prefill" in output and "API=未就绪" in output
        assert str(child_info["pid"]) in output
        assert f"http://127.0.0.1:{child_info['port']}" in output
        assert "GPU-test-prefill" in output
        assert str(tmp_path / "prefill.log") in output
        (tmp_path / "healthy").touch()
        assert service.execute(options("status")) == 0
        assert "API=就绪" in capsys.readouterr().out
        assert state_file.read_bytes() == original
        assert process.poll() is None
        assert service.process_identity(child_info["pid"]) is not None
    finally:
        state_file.unlink(missing_ok=True)
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)


def test_show_log_leaves_lifecycle_lock_and_service_untouched(
    service, monkeypatch, tmp_path
):
    """长时间查看日志不能阻塞 stop，也不能改状态或发健康检查请求。"""
    import fcntl

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "ssh"))
    import log_view

    assert service.execute(options()) == 0
    state_file = tmp_path / "service.json"
    original = state_file.read_bytes()
    state = json.loads(original)
    monkeypatch.setattr(
        service, "urlopen", lambda *args, **kwargs: pytest.fail("日志查看不应探测 API")
    )

    def view(logs, resource):
        with (tmp_path / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert service.running(state)
        assert state_file.read_bytes() == original
        assert Path(state["log"]) in [Path(path) for _, path in logs]
        return 0

    monkeypatch.setattr(log_view, "show_logs", view)
    assert service.execute(options("show-log")) == 0
    assert service.running(state)
    assert state_file.read_bytes() == original
