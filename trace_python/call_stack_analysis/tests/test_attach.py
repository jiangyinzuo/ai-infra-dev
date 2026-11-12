"""Public PID-attach behavior against real, uninstrumented CPython subjects."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from trace_python.call_stack_analysis.tests import attach_smoke as smoke

requires_remote_exec = pytest.mark.skipif(
    sys.version_info < (3, 14) or not hasattr(sys, "remote_exec"),
    reason="Real sys.remote_exec requires CPython 3.14+",
)


def real_attach(scenario: Callable[..., dict[str, Any]], *args: Any) -> dict[str, Any]:
    try:
        return scenario(*args)
    except smoke.PermissionUnavailable as exc:
        pytest.skip(f"Host permissions deny real sys.remote_exec: {exc}")


@requires_remote_exec
def test_real_attach_threads_reattach_and_multi_pid(tmp_path: Path) -> None:
    original_cwd = Path.cwd()
    original_environment = dict(os.environ)
    result = real_attach(smoke.successful_scenario, tmp_path / "captures" / "trace", 1.0)
    assert len(result["reports"]) == 3
    assert Path.cwd() == original_cwd
    assert dict(os.environ) == original_environment
    for report in result["reports"]:
        assert all(capture["stopped"] for capture in report["captures"])


@requires_remote_exec
def test_invalid_target_ack_keeps_business_alive(tmp_path: Path) -> None:
    report = real_attach(smoke.invalid_target_scenario, tmp_path / "invalid")
    assert report["errors"]


@requires_remote_exec
def test_timeout_cancels_late_installation(tmp_path: Path) -> None:
    report = real_attach(smoke.timeout_scenario, tmp_path / "timeout")
    assert report["errors"]


@requires_remote_exec
def test_target_can_disable_remote_debugging(tmp_path: Path) -> None:
    report = smoke.disabled_scenario(tmp_path / "disabled")
    assert report["errors"]


@requires_remote_exec
@pytest.mark.parametrize("option", ["--duration", "--timeout"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_cli_rejects_nonpositive_or_nonfinite_times(tmp_path: Path, option: str, value: str) -> None:
    config = smoke.make_config(tmp_path / "config.py")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    result = subprocess.run(
        [sys.executable, "-m", "trace_python.call_stack_analysis.attach", "--pid", str(os.getpid()),
         "--config", str(config), "--output-prefix", str(tmp_path / "invalid-time"), f"{option}={value}"],
        cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=20,
    )
    assert result.returncode != 0, result.stdout
    assert not list(tmp_path.glob("invalid-time.*")), (result.stdout, result.stderr)


def test_python312_rejects_attach_without_touching_target(tmp_path: Path) -> None:
    python312 = shutil.which("python3.12")
    if python312 is None:
        pytest.skip("Python 3.12 executable is not installed")
    config = smoke.make_config(tmp_path / "config.py")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    result = subprocess.run(
        [python312, "-m", "trace_python.call_stack_analysis.attach", "--pid", str(os.getpid()),
         "--config", str(config), "--output-prefix", str(tmp_path / "unsupported")],
        cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=20,
    )
    assert result.returncode != 0, result.stdout
    assert "3.14" in result.stderr, result.stderr
    assert not list(tmp_path.glob("unsupported.*")), (result.stdout, result.stderr)


@requires_remote_exec
def test_partial_multi_pid_failure_uninstalls_successful_capture(tmp_path: Path) -> None:
    from trace_python.call_stack_analysis.attach import AttachError, capture_processes

    first = smoke.Subject(tmp_path / "first")
    disabled = smoke.Subject(tmp_path / "disabled", disabled=True)
    try:
        config = smoke.make_config(tmp_path / "config.py")
        with pytest.raises(AttachError) as caught:
            capture_processes([first.process.pid, disabled.process.pid], config,
                              tmp_path / "partial", duration=0.1, timeout=5)
        if smoke.permission_error(caught.value):
            pytest.skip(f"Host permissions deny real sys.remote_exec: {caught.value}")
        capture, = caught.value.report["captures"]
        assert capture["pid"] == first.process.pid and capture["stopped"]
        first.assert_detached()
        disabled.assert_detached()
    finally:
        failed = sys.exc_info()[0] is not None
        first.close(check=not failed)
        disabled.close(check=not failed)


@requires_remote_exec
def test_duplicate_start_and_wrong_stop_preserve_active_capture(tmp_path: Path) -> None:
    from dataclasses import asdict
    from trace_python.call_stack_analysis.attach import _make_request, _submit
    from trace_python.call_stack_analysis.configuration import load_config

    subject = smoke.Subject(tmp_path / "subject")
    config_path = smoke.make_config(tmp_path / "config.py")
    config = load_config(config_path)
    installed = False

    def request(label: str, action: str, run_id: str) -> dict[str, Any]:
        directory = tmp_path / label
        directory.mkdir()
        message = _make_request(directory, {
            "pid": subject.process.pid, "action": action, "run_id": run_id,
            "root": str(config.root), "config_path": str(config_path),
            "output_prefix": str(tmp_path / "overlap"),
            "targets": [asdict(target) for target in config.targets],
        })
        _submit(message)
        return message.wait(5)

    try:
        try:
            start = request("install", "start", "first")
        except Exception as exc:
            if smoke.permission_error(exc):
                pytest.skip(f"Host permissions deny real sys.remote_exec: {exc}")
            raise
        assert start["status"] == "ok", start
        installed = True
        assert request("duplicate", "start", "second")["status"] == "error"
        assert request("mismatch", "stop", "second")["status"] == "error"
        response = subject.batch("original-still-active")
        assert response["hooks"] == {"current": True, "future": True}
        assert request("uninstall", "stop", "first")["status"] == "ok"
        installed = False
        subject.assert_detached()
        smoke.check_capture({**start, "stopped": True}, {"run_id": "first"}, response)
    finally:
        if installed:
            request("emergency-stop", "stop", "first")
        subject.close(check=sys.exc_info()[0] is None)
