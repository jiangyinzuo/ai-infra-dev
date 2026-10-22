import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace


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
