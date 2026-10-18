"""Printed process execution and Docker inventory."""

import json
import shlex
import subprocess
from typing import Any


def run(*args: str, **kwargs: Any) -> subprocess.CompletedProcess:
    print(shlex.join(args), flush=True)
    return subprocess.run(list(args), check=True, **kwargs)


def containers() -> list:
    ids = run("docker", "ps", "-aq", capture_output=True, text=True).stdout.split()
    if not ids:
        return []
    return json.loads(
        run("docker", "inspect", *ids, capture_output=True, text=True).stdout
    )


def managed(name: str) -> bool:
    return name.lstrip("/").startswith("jiangyinzuo-")
