#!/usr/bin/env python3
"""从 SSH config 生成 Ansible JSON 主机清单（依赖 sshconf）。"""

import argparse
from dataclasses import dataclass
import fnmatch
import json
from pathlib import Path
import shlex
import os
import re
import subprocess
from functools import lru_cache

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PROFILES = Path(__file__).resolve().parent / "hosts.json"

from ssh_config import host_passwords


@dataclass(frozen=True)
class SSHSource:
    config: str
    ssh_config: str
    public_key: str
    password_method: str


SSH_SOURCES = {
    "windows": SSHSource(
        "${WINDOWS_SSH_CONFIG}",
        "${WINDOWS_SSH_CONFIG_NATIVE}",
        "${WINDOWS_SSH_HOME}/id_ed25519.pub",
        "windows-askpass",
    ),
    "wsl": SSHSource(
        "~/.ssh/config", "~/.ssh/config", "${WSL_SSH_HOME}/id_ed25519.pub", "sshpass"
    ),
}


def inventory_group(hosts: dict, ssh: str, ssh_config: str) -> dict:
    """Build one inventory group for either NPU or machine-information queries."""
    return {
        "hosts": hosts,
        "vars": {
            "ansible_connection": "ssh",
            "ansible_ssh_executable": ssh,
            "ansible_ssh_args": shlex.join(
                [
                    "-F",
                    ssh_config,
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ControlMaster=no",
                    "-o",
                    "StrictHostKeyChecking=accept-new",
                ]
            ),
        },
    }


def build_inventory(targets, host_variables=None, inventory_group_name=None):
    """Build inventories while leaving tool-specific variables to the caller."""
    children = {}
    for index, target in enumerate(targets):
        variables = host_variables(target) if host_variables else {}
        group = inventory_group(
            {target["host"]: variables}, target["ssh"], target["ssh_config"]
        )
        children[f"target_{index}"] = group
    if inventory_group_name:
        children = {inventory_group_name: {"children": children}}
    return {"all": {"children": children}}


@lru_cache(maxsize=1)
def windows_paths():
    profile = subprocess.check_output(
        ["cmd.exe", "/d", "/c", "chcp 65001 >nul & echo %USERPROFILE%"],
        cwd="/mnt/c",
        text=True,
        encoding="utf-8",
        timeout=15,
    ).strip()
    native = profile + "\\.ssh\\config"
    local = subprocess.check_output(
        ["wslpath", "-u", native], text=True, timeout=10
    ).strip()
    return {
        "WINDOWS_SSH_CONFIG": local,
        "WINDOWS_SSH_CONFIG_NATIVE": native,
        "WINDOWS_SSH_HOME": str(Path(local).parent),
    }


def load_targets(args):
    profiles = json.loads(DEFAULT_PROFILES.read_text())["groups"]
    selected = {
        name: source
        for name, source in profiles.items()
        if not args.group or name in args.group
    }
    missing = set(args.group or []) - selected.keys()
    if missing:
        raise ValueError("Unknown groups: " + ", ".join(sorted(missing)))
    variables = {
        **os.environ,
        "REPO": str(REPO),
        "WSL_SSH_HOME": str(Path.home() / ".ssh"),
    }
    if "windows" in selected.values():
        variables.update(windows_paths())

    def expand(value):
        def replace(match):
            name = match[1]
            if name not in variables:
                raise ValueError("Undefined profile variable: " + name)
            return variables[name]

        return os.path.expanduser(re.sub(r"\$\{([^}]+)\}", replace, value))

    targets = []
    for name, source_name in selected.items():
        if source_name not in SSH_SOURCES:
            raise ValueError(f"Unknown SSH source: {source_name}")
        source = SSH_SOURCES[source_name]
        config = Path(expand(source.config))
        matches = [
            h for h in host_passwords(config) if fnmatch.fnmatchcase(h, name + "*")
        ]
        for host in sorted(matches):
            if args.host_glob and not any(
                fnmatch.fnmatchcase(host, p) for p in args.host_glob
            ):
                continue
            targets.append(
                {
                    "group": name,
                    "host": host,
                    "ssh": str(Path(__file__).resolve().parent / "ssh2"),
                    "ssh_config": expand(source.ssh_config),
                    "python": "python3",
                    "public_key": expand(source.public_key),
                    "config": str(config),
                    "password_method": source.password_method,
                }
            )
    if not targets:
        raise ValueError("No hosts selected")
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group", action="append", help="Profile group to select; repeatable"
    )
    parser.add_argument(
        "--host-glob", action="append", help="Further filter aliases; repeatable"
    )
    parser.add_argument(
        "--inventory-group", required=True, help="Ansible playbook target group"
    )
    args = parser.parse_args()
    try:
        targets = load_targets(args)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(build_inventory(targets, inventory_group_name=args.inventory_group))
    )


if __name__ == "__main__":
    main()
