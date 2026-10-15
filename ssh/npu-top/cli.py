#!/usr/bin/env python3
"""查询 NPU 使用情况；默认所有组，其余参数传给 ansible-playbook。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from ssh_hosts import build_inventory, load_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "-i", "--inventory", type=Path, help="直接使用 Ansible inventory，不按组生成"
    )
    parser.add_argument("--group", action="append", help="选择组，可重复，默认全部")
    parser.add_argument("--host-glob", action="append", help="主机名通配符，可重复")
    parser.add_argument("--containers", action="store_true", help="显示进程所属容器")
    args, extra = parser.parse_known_args()
    try:
        if args.inventory and (args.group or args.host_glob):
            parser.error("--inventory 不能与 --group/--host-glob 同用")
        environment = dict(os.environ)
        plugins = str(ROOT / "callback_plugins")
        existing = environment.get("ANSIBLE_CALLBACK_PLUGINS")
        environment["ANSIBLE_CALLBACK_PLUGINS"] = plugins + (
            os.pathsep + existing if existing else ""
        )
        environment["ANSIBLE_STDOUT_CALLBACK"] = "npu_top"
        with tempfile.TemporaryDirectory(prefix="npu-top-") as directory:
            path = args.inventory
            if path is None:
                targets = load_targets(args)
                inventory = build_inventory(targets, inventory_group_name="npu_servers")
                path = Path(directory) / "inventory.json"
                path.write_text(json.dumps(inventory), encoding="utf-8")
                path.chmod(0o600)
            command = [
                "ansible-playbook",
                "-i",
                str(path),
                str(ROOT / "npu-smi.yml"),
            ]
            if args.containers:
                command += ["-e", "npu_containers=true"]
            return subprocess.run(
                [*command, *extra], env=environment, check=False
            ).returncode
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
