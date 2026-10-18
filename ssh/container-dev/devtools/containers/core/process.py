"""Printed process execution and Docker inventory."""

import json
from terminal import run


def containers() -> list:
    ids = run("docker", "ps", "-aq", capture_output=True, text=True).stdout.split()
    if not ids:
        return []
    return json.loads(
        run("docker", "inspect", *ids, capture_output=True, text=True).stdout
    )


def managed(name: str) -> bool:
    return name.lstrip("/").startswith("jiangyinzuo-")
