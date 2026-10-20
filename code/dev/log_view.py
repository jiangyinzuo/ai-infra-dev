"""Follow remote container logs in a WSL-local pager, independent of the service."""

import os
import shlex
import shutil
import signal
import subprocess
import sys

from ssh_common.remote import SSH_DIR, checked_command, ssh_arguments
from container_command import container_exec


def ssh_args(host, command):
    return ssh_arguments(host, command, tty=False, ssh_dir=SSH_DIR)


def attach(args):
    query = 'exec python3 "$HOME"/ai-infra-dev/code/dev vllm serve --print-log --no-hints'
    command = shlex.join(container_exec(args.container, ["bash", "-lc", query]))
    result = subprocess.run(
        ssh_args(args.host, checked_command(args.host, command)),
        capture_output=True,
        text=True,
        check=True,
    )
    log = result.stdout.strip()
    if not log:
        raise RuntimeError("未获得服务日志路径")
    command = shlex.join(
        container_exec(args.container, ["tail", "-n", "100", "-f", "--", log])
    )
    transport = ssh_args(args.host, command)
    reattach = shlex.join(transport) + " | less -R +F"
    try:
        if (
            not sys.stdin.isatty()
            or not sys.stdout.isatty()
            or not shutil.which("less")
        ):
            print("未打开日志查看：需要交互终端和本地 less。")
            return
        print(
            "日志基于 less +F：Ctrl-C 暂停跟随，F 恢复，q 退出；不停止服务。",
            flush=True,
        )
        stream = subprocess.Popen(
            transport, stdout=subprocess.PIPE, start_new_session=True
        )
        try:
            pager = subprocess.Popen(
                ["less", "-R", "+F"],
                stdin=stream.stdout,
                env={**os.environ, "LESSSECURE": "1"},
            )
            stream.stdout.close()
            while True:
                try:
                    pager.wait()
                    break
                except KeyboardInterrupt:
                    # The foreground pager receives Ctrl-C too: let it enter browse mode.
                    continue
        finally:
            if stream.poll() is None:
                os.killpg(stream.pid, signal.SIGTERM)
                try:
                    stream.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(stream.pid, signal.SIGKILL)
                    stream.wait()
    finally:
        print(
            "已退出日志查看，未向服务发送停止信号。\n重新查看日志：\n" + reattach,
            flush=True,
        )
