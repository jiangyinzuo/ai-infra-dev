"""View service logs without changing the service's environment or lifecycle."""

import os
import shlex
import shutil
import signal
import subprocess
import sys
import time

from ssh_common.remote import SSH_DIR, checked_command, ssh_arguments
from container_command import container_exec


def ssh_args(host, command):
    return ssh_arguments(host, command, tty=False, ssh_dir=SSH_DIR)


def attach(args):
    query = 'exec python3 "$HOME"/ai-infra-dev/code/dev ' + shlex.join(
        [args.resource, "serve", "--print-log", "--no-hints"]
    )
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


def show_logs(logs, resource):
    tmux = shutil.which("tmux")
    tail = shutil.which("tail")
    if not tmux or not tail:
        raise RuntimeError("show-log 需要 tmux 和 tail，请先安装")
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError("show-log 需要交互终端")
    session = f"{resource}-logs-{os.getpid()}-{time.time_ns()}"
    created = False
    try:
        for index in range(0, len(logs), 2):
            pair = logs[index:index + 2]
            name, path = pair[0]
            command = "exec " + shlex.join(
                [tail, "-n", "100", "-F", "-s", "0.2", "--", str(path)]
            )
            window_name = "-".join(label for label, _ in pair)
            create = (
                ["new-session", "-d", "-s", session]
                if not created else ["new-window", "-d", "-t", session + ":"]
            )
            result = subprocess.run(
                [tmux, *create, "-P", "-F", "#{window_id} #{pane_id}",
                 "-n", window_name, command],
                capture_output=True, text=True, check=True,
            )
            created = True
            window, pane = result.stdout.strip().split()
            subprocess.run([tmux, "select-pane", "-t", pane, "-T", name], check=True)
            if len(pair) == 2:
                name, path = pair[1]
                command = "exec " + shlex.join(
                    [tail, "-n", "100", "-F", "-s", "0.2", "--", str(path)]
                )
                result = subprocess.run(
                    [tmux, "split-window", "-d", "-v", "-t", window,
                     "-P", "-F", "#{pane_id}", command],
                    capture_output=True, text=True, check=True,
                )
                subprocess.run(
                    [tmux, "select-pane", "-t", result.stdout.strip(), "-T", name],
                    check=True,
                )
                subprocess.run(
                    [tmux, "select-layout", "-t", window, "even-vertical"],
                    stdout=subprocess.DEVNULL, check=True,
                )
        subprocess.run(
            [tmux, "set-option", "-t", session, "pane-border-status", "top"],
            check=True,
        )
        subprocess.run(
            [tmux, "set-option", "-t", session, "pane-border-format", " #{pane_title} "],
            check=True,
        )
        subprocess.run(
            [tmux, "select-window", "-t", session + ":^"], check=True,
        )
    except BaseException:
        if created:
            subprocess.run(
                [tmux, "kill-session", "-t", session],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
        raise
    print(f"日志会话：{session}；关闭或分离查看器不会停止 vLLM。", flush=True)
    action = "switch-client" if os.environ.get("TMUX") else "attach-session"
    result = subprocess.run([tmux, action, "-t", session], check=False)
    print("重新进入：" + shlex.join([tmux, "attach-session", "-t", session]))
    return result.returncode
