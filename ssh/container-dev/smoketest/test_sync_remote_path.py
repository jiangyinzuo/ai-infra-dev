import importlib.util
import os
from pathlib import Path
import subprocess

import pytest

SPEC = importlib.util.spec_from_file_location(
    "sync_devtools", Path(__file__).resolve().parents[1] / "sync-devtools.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("relative", [True, False])
def test_remote_mkdir_preserves_spaces_quotes_and_metacharacters(tmp_path, relative):
    destination = tmp_path / "tools ' quoted; $literal"
    path = "~/" + destination.name if relative else str(destination)
    command = "mkdir -p -- " + MODULE.remote_path(path)
    subprocess.run(
        ["sh", "-c", command],
        env={**os.environ, "HOME": str(tmp_path)},
        check=True,
        timeout=10,
    )
    assert destination.is_dir()
