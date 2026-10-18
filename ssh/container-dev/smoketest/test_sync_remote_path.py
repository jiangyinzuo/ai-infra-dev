import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
from workflow import sync as MODULE


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
