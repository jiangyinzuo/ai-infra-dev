from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
from containers.commands import create


def test_docker_command_uses_host_ipc():
    command = create.docker_command("example", "/home/user")

    assert "--ipc=host" in command
    assert "--shm-size" not in command
