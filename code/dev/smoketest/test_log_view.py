import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import shlex
import shutil
import subprocess
import time
import uuid

import pytest


def test_exit_pager_detaches_only_log_transport(monkeypatch, tmp_path, capsys):
    dev = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(dev))
    monkeypatch.syspath_prepend(str(dev.parents[1] / "ssh"))
    import log_view

    ssh = tmp_path / "ssh2"
    ssh.write_text(
        "#!"
        + sys.executable
        + '\nimport sys,time\nif "--print-log" in sys.argv[-1]:\n print("/tmp/service log")\nelse:\n print("live service log",flush=True)\n time.sleep(120)\n'
    )
    ssh.chmod(0o755)
    pager = tmp_path / "less"
    pager.write_text(
        '#!/bin/sh\nIFS= read -r line\nprintf "%s" "$line" > "$PAGER_CAPTURE"\n'
    )
    pager.chmod(0o755)
    monkeypatch.setattr(log_view, "SSH_DIR", tmp_path)
    monkeypatch.setattr(log_view, "checked_command", lambda host, command: command)
    monkeypatch.setattr(log_view.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(
        log_view.sys,
        "stdout",
        type(
            "TTY",
            (),
            {
                "isatty": lambda self: True,
                "write": lambda self, text: None,
                "flush": lambda self: None,
            },
        )(),
    )
    monkeypatch.setenv("PATH", str(tmp_path) + ":" + os.environ["PATH"])
    monkeypatch.setenv("PAGER_CAPTURE", str(tmp_path / "captured"))
    log_view.attach(SimpleNamespace(resource="vllm-ascend", host="suzhou-one", container="dev1"))
    assert (tmp_path / "captured").read_text() == "live service log"


@pytest.mark.parametrize("count", [1, 3, 4])
def test_tmux_groups_logs_and_follows_new_lines(monkeypatch, tmp_path, count):
    """单日志、奇数和双卡日志均独立跟随，每窗口最多两个 pane。"""
    tmux = shutil.which("tmux")
    if not tmux or not shutil.which("tail"):
        pytest.skip("requires tmux and tail")
    dev = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(dev))
    monkeypatch.syspath_prepend(str(dev.parents[1] / "ssh"))
    import log_view

    server = "dev-log-test-" + uuid.uuid4().hex
    command = [tmux, "-L", server, "-f", "/dev/null"]
    wrapper = tmp_path / "tmux"
    wrapper.write_text('#!/bin/sh\nexec ' + shlex.join(command) + ' "$@"\n')
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + ":" + os.environ["PATH"])
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("TMUX_PANE", raising=False)
    monkeypatch.setattr(log_view.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(log_view.sys.stdout, "isatty", lambda: True)
    run = subprocess.run

    def inspect_viewer(args, **kwargs):
        if args[1] != "attach-session":
            return run(args, **kwargs)
        session = args[-1]
        result = run(
            [*command, "list-panes", "-s", "-t", session,
             "-F", "#{window_id} #{pane_id} #{pane_title}"],
            capture_output=True, text=True, check=True,
        )
        rows = [line.split() for line in result.stdout.splitlines()]
        assert len(rows) == count
        windows = {window for window, _, _ in rows}
        assert len(windows) == (count + 1) // 2
        assert all(sum(row[0] == window for row in rows) <= 2 for window in windows)
        for _, pane, title in rows:
            path = tmp_path / f"{title} log 'quoted'.txt"
            with path.open("a") as output:
                output.write(f"new-line-{title}\n")
            deadline = time.monotonic() + 3
            while True:
                visible = run(
                    [*command, "capture-pane", "-p", "-t", pane],
                    capture_output=True, text=True, check=True,
                ).stdout
                if f"new-line-{title}" in visible:
                    break
                assert time.monotonic() < deadline, visible
                time.sleep(0.02)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(log_view.subprocess, "run", inspect_viewer)
    logs = []
    for index in range(count):
        name = f"log{index}"
        path = tmp_path / f"{name} log 'quoted'.txt"
        path.write_text("initial-line\n")
        logs.append((name, str(path)))
    try:
        assert log_view.show_logs(logs, "vllm") == 0
    finally:
        run([*command, "kill-server"], stderr=subprocess.DEVNULL, check=False)
