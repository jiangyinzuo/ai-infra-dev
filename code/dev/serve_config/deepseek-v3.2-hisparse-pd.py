"""双卡 SM120：tiny random DeepSeek-V3.2，P/D 分离，D 侧开启 HiSparse。

按 --devices 的顺序分配 P、D；API 端口为 8100、8200，代理入口为 8192。
复用单卡配置的模型与参数。仅用于学习执行路径，不用于质量或性能评估。
"""

import importlib.util
import json
import os
import runpy
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve_config import ServeConfig

ROOT = Path(__file__).resolve().parent
BASE = runpy.run_path(str(ROOT / "deepseek-v3.2-hisparse.py"))["config"]
PROXY = (
    ROOT.parents[1]
    / "vllm-workspace/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"
)

config = ServeConfig(
    model=BASE.model,
    device_count=2,
    serve_command=[sys.executable, str(Path(__file__).resolve()), "{model}"],
)


def service_commands(model):
    common = [sys.executable, "-m", "vllm.entrypoints.cli.main", *BASE.serve_command[1:]]
    common = [argument.replace("{model}", model) for argument in common]
    common.extend([
        "--seed", "42",
        "--served-model-name", "tiny-dsa",
        "--host", "127.0.0.1",
        "--disable-extensible-kv-cache",
    ])
    producer = {
        "kv_connector": "NixlConnector",
        "kv_role": "kv_producer",
        "kv_load_failure_policy": "fail",
    }
    consumer = {
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "NixlConnector",
                    "kv_role": "kv_consumer",
                    "kv_load_failure_policy": "fail",
                },
                json.loads(common[common.index("--kv-transfer-config") + 1]),
            ],
        },
    }
    commands = {}
    for role, port, transfer in (
        ("prefill", "8100", producer),
        ("decode", "8200", consumer),
    ):
        command = common.copy()
        command[command.index("--kv-transfer-config") + 1] = json.dumps(transfer)
        if role == "prefill":
            index = command.index("--attention-config") + 1
            attention = json.loads(command[index])
            attention.pop("hisparse_config")
            command[index] = json.dumps(attention)
        commands[role] = [*command, "--port", port]
    commands["proxy"] = [
        sys.executable, str(PROXY),
        "--host", "127.0.0.1", "--port", "8192",
        "--prefiller-hosts", "127.0.0.1", "--prefiller-ports", "8100",
        "--decoder-hosts", "127.0.0.1", "--decoder-ports", "8200",
    ]
    return commands


def signal_group(process, sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def stop_services(processes):
    for process in processes:
        signal_group(process, signal.SIGTERM)
    deadline = time.monotonic() + 20
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    # 组长退出不代表 vLLM worker 已退出；同时回收仍持有 GPU 的后代进程。
    for process in processes:
        signal_group(process, signal.SIGKILL)
    for process in processes:
        process.wait()


def run_services(commands, devices, directory):
    stopping = threading.Event()
    previous = {}
    processes = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, lambda *_: stopping.set())
    try:
        for index, (role, command) in enumerate(commands.items()):
            if stopping.is_set():
                return 0
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = devices[index] if index < 2 else ""
            if role != "proxy":
                environment["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "127.0.0.1"
                environment["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(5600 + index)
                environment["UCX_NET_DEVICES"] = "all"
            log = directory / f"{role}.log"
            print(f"{role}: {shlex.join(command)}\n日志：{log}", flush=True)
            with log.open("ab") as output:
                process = subprocess.Popen(
                    command,
                    env=environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            processes.append(process)
        print(
            "P/D 与代理进程已启动（不代表 API 就绪），"
            "请求入口：http://127.0.0.1:8192",
            flush=True,
        )
        while not stopping.wait(0.2):
            for role, process in zip(commands, processes):
                code = process.poll()
                if code is not None:
                    print(
                        f"{role} 已退出，code={code}；停止其余进程。"
                        f"日志：{directory / (role + '.log')}",
                        flush=True,
                    )
                    return code if code > 0 else 1
        return 0
    finally:
        stop_services(processes)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("请通过 dev vllm serve --script deepseek-v3.2-hisparse-pd.py 启动")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if (
        len(devices) != 2
        or any(not device.strip() for device in devices)
        or devices[0] == devices[1]
    ):
        raise SystemExit("P/D 需要 CUDA_VISIBLE_DEVICES 中两个不同的设备，顺序为 P、D")
    if importlib.util.find_spec("nixl") is None:
        raise SystemExit("缺少 NIXL；先在 vLLM 环境安装 nixl==1.5.2 和 nixl-cu13==1.5.2")
    if not PROXY.is_file():
        raise SystemExit(f"找不到 NIXL 代理：{PROXY}")
    directory = (
        ROOT / "runtime"
        / f"hisparse-pd-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    )
    directory.mkdir(parents=True)
    return run_services(service_commands(sys.argv[1]), devices, directory)


if __name__ == "__main__":
    sys.exit(main())
