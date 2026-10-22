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
import importlib.util
from serve_config import ServeConfig


def load_config(path):
    if path.suffix != ".py":
        raise ValueError("仅支持 Python ServeConfig 配置文件")
    spec = importlib.util.spec_from_file_location("user_serve_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = getattr(module, "config", None)
    if not isinstance(config, ServeConfig):
        raise ValueError("配置文件必须定义 config = ServeConfig(...)")
    if type(config.device_count) is not int or config.device_count < 1:
        raise ValueError("device_count 必须为正整数")
    if (
        not isinstance(config.serve_command, list)
        or not config.serve_command
        or not all(
            isinstance(argument, str) and argument for argument in config.serve_command
        )
    ):
        raise ValueError("serve_command 必须为非空字符串参数列表")
    return config


def process_identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2 :].split()
        if fields[0] == "Z":
            return None
        return fields[19]
    except FileNotFoundError:
        return None


def state_directory(resource):
    path = Path("/tmp") / f"{resource}-dev-{os.getuid()}-{socket.gethostname()}"
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


def resolve_script(script_name):
    name = Path(script_name)
    if (
        name.name != script_name
        or name.suffix != ".py"
        or name.name == "__init__.py"
    ):
        raise ValueError("--script 仅接受 serve_config 下的 Python 配置文件名")
    script = Path(__file__).resolve().parent / "serve_config" / name
    if not script.is_file():
        raise ValueError("启动脚本不存在：" + str(script))
    return script


def prepare_existing_service(state_file, stopping):
    """Return whether a new service may start; caller must hold the lifecycle lock."""
    state = json.loads(state_file.read_text()) if state_file.exists() else None
    if state and running(state):
        if not stopping:
            answer = input(
                f"服务已启动 PID={state['pid']}，是否 stop 并重新启动？[y/N]："
            )
            if answer.strip().lower() != "y":
                return False
        stop(state)
        state_file.unlink(missing_ok=True)
        print("服务已停止")
        return not stopping
    if state:
        state_file.unlink()
    if stopping:
        print("当前没有本工具运行中的服务")
        return False
    return True


def select_resources(args, config):
    from serve_resources import (
        choose_model,
        choose_devices,
        device_visibility_variable,
    )

    model = (
        config.model
        if args.model is None and config.model is not None
        else choose_model(args.model)
    )
    visibility = device_visibility_variable(args.resource)
    devices = choose_devices(args.devices, args.resource)
    if len(devices.split(",")) != config.device_count:
        raise ValueError(
            f"配置要求 {config.device_count} 个 device，所选数量不匹配"
        )
    return model, visibility, devices


def spawn_service(command, script, environment, log, visibility, devices):
    with log.open("ab") as output:
        # 将启动参数保存在日志中，进入 less 后也能查看；不改变服务进程的输出行为。
        header = (
            visibility
            + "="
            + shlex.quote(devices)
            + " "
            + shlex.join(command)
        )
        output.write(("执行：" + header + "\n").encode("utf-8"))
        output.flush()
        process = subprocess.Popen(
            command,
            cwd=script.parent,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return process


def record_service(process, state_file, log, model, devices, script):
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
            "script": str(script),
        }
        temporary = state_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(state))
        temporary.replace(state_file)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        raise


def start_service(args, script, config, directory):
    state_file = directory / "service.json"
    model, visibility, devices = select_resources(args, config)
    command = [
        argument.replace("{model}", model) for argument in config.serve_command
    ]
    # 不要为日志跟随添加 PYTHONUNBUFFERED=1：它会改变 vLLM 进程的输出缓冲，
    # 可能增加写系统调用和 I/O 开销、影响性能；宁可接受日志延迟，也要保留 vLLM 进程原有环境。
    environment = dict(os.environ, MODEL_PATH=model)
    environment[visibility] = devices
    print(
        "MODEL_PATH="
        + shlex.quote(model)
        + " "
        + visibility
        + "="
        + shlex.quote(devices),
        flush=True,
    )
    print("执行：" + shlex.join(command), flush=True)
    print(
        "运行时目录："
        + str(script.parent / "runtime")
        + "（配置可在此生成运行时文件；已加入 .gitignore）",
        flush=True,
    )
    log = directory / (
        time.strftime("serve-%Y%m%d-%H%M%S-") + str(time.time_ns()) + ".log"
    )
    process = spawn_service(command, script, environment, log, visibility, devices)
    record_service(process, state_file, log, model, devices, script)
    time.sleep(0.5)
    if process.poll() is not None:
        state_file.unlink(missing_ok=True)
        raise RuntimeError(
            f"服务启动失败，退出码 {process.returncode}；日志：{log}"
        )
    print(f"后台进程已启动 PID={process.pid}，设备={devices}（不代表 API 已就绪）")
    print(f"日志：{log}\n查看：tail -f {log}")
    return 0


def service_status(directory):
    state_file = directory / "service.json"
    if not state_file.exists():
        print("当前没有本工具运行中的服务")
        return 0
    state = json.loads(state_file.read_text())
    status = "运行中（不代表 API 已就绪）" if running(state) else "已退出（状态记录已过期）"
    print(f"状态：{status}")
    for label, key in (("PID", "pid"), ("模型", "model"), ("设备", "devices"), ("配置", "script"), ("日志", "log")):
        print(f"{label}：{state[key]}")
    return 0


def execute(args):
    action = "log" if getattr(args, "print_log", False) else args.serve_action
    launch = None
    match action:
        case None:
            # 先验证配置，再考虑停止旧服务，避免无效配置打断现有服务。
            script = resolve_script(args.script)
            launch = (script, load_config(script))
        case "stop" | "status" | "log":
            pass
        case _:
            raise ValueError("不支持的服务动作：" + str(action))
    directory = state_directory(args.resource)
    with (directory / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        match action:
            case "log":
                state = json.loads((directory / "service.json").read_text())
                print(state["log"])
                return 0
            case "status":
                return service_status(directory)
            case "stop":
                prepare_existing_service(directory / "service.json", stopping=True)
                return 0
            case None:
                if not prepare_existing_service(directory / "service.json", stopping=False):
                    return 0
                script, config = launch
                return start_service(args, script, config, directory)
