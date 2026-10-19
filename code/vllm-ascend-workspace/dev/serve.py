"""Manage only this tool's container-local vLLM process group."""

import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import shlex
import time


def process_identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2 :].split()
        if fields[0] == "Z":
            return None
        return fields[19]
    except FileNotFoundError:
        return None


def state_directory():
    path = Path("/tmp") / f"vllm-ascend-dev-{os.getuid()}-{socket.gethostname()}"
    path.mkdir(mode=0o700, exist_ok=True)
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise ValueError("服务状态目录不属于当前用户")
    path.chmod(0o700)
    return path


def running(state):
    return process_identity(state["pid"]) == state["identity"]


def stop(state, timeout=30):
    if not running(state):
        return
    # start_new_session makes the recorded PID the process-group leader.
    if os.getpgid(state["pid"]) != state["pid"]:
        raise ValueError("进程组身份不匹配，拒绝停止")
    os.killpg(state["pid"], signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while running(state) and time.monotonic() < deadline:
        time.sleep(0.1)
    if running(state):
        raise RuntimeError("服务在 30 秒内未停止；未强制 SIGKILL，请检查日志")


def execute(args):
    directory = state_directory()
    state_file = directory / "service.json"
    with (directory / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(state_file.read_text()) if state_file.exists() else None
        if state and running(state):
            if args.serve_action != "stop":
                answer = input(
                    f"服务已启动 PID={state['pid']}，是否 stop 并重新启动？[y/N]："
                )
                if answer.strip().lower() != "y":
                    return 0
            stop(state)
            state_file.unlink(missing_ok=True)
            print("服务已停止")
            if args.serve_action == "stop":
                return 0
        elif state:
            state_file.unlink()
        if args.serve_action == "stop":
            print("当前没有本工具运行中的服务")
            return 0
        from serve_resources import choose_model, choose_devices

        model = choose_model(args.model)
        devices = choose_devices(args.devices)
        command = [
            "vllm",
            "serve",
            model,
            "--tensor-parallel-size",
            str(len(devices.split(","))),
        ]
        environment = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=devices)
        print(
            "ASCEND_RT_VISIBLE_DEVICES="
            + shlex.quote(devices)
            + " "
            + shlex.join(command),
            flush=True,
        )
        log = directory / (
            time.strftime("serve-%Y%m%d-%H%M%S-") + str(time.time_ns()) + ".log"
        )
        with log.open("ab") as output:
            process = subprocess.Popen(
                command,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            identity = process_identity(process.pid)
            if identity is None:
                raise RuntimeError("服务进程已退出，请检查日志：" + str(log))
            state = {
                "pid": process.pid,
                "identity": identity,
                "log": str(log),
                "model": model,
                "devices": devices,
            }
            temporary = state_file.with_suffix(".tmp")
            temporary.write_text(json.dumps(state))
            temporary.replace(state_file)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            raise
        time.sleep(0.5)
        if process.poll() is not None:
            state_file.unlink(missing_ok=True)
            raise RuntimeError(
                f"服务启动失败，退出码 {process.returncode}；日志：{log}"
            )
        print(f"后台进程已启动 PID={process.pid}，设备={devices}（不代表 API 已就绪）")
        print(f"日志：{log}\n查看：tail -f {log}")
        return 0
