"""Printed process execution and Docker inventory."""

from ssh_common.config import REMOTE_USER
import json
from ssh_common.terminal import run


def containers() -> list:
    ids = run("docker", "ps", "-aq", capture_output=True, text=True).stdout.split()
    if not ids:
        return []
    return json.loads(
        run("docker", "inspect", *ids, capture_output=True, text=True).stdout
    )


def managed(name: str) -> bool:
    return name.lstrip("/").startswith(REMOTE_USER + "-")
