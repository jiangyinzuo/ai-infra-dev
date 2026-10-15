#!/usr/bin/env python3
"""通过 Ansible 批量管理 Linux 用户及 SSH 密钥；默认仅输出 plan，不连接远端。"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from ssh_hosts import build_inventory, load_targets
from bootstrap import ensure_root_access


def build_plan(targets):
    """固定管理 jiangyinzuo；私钥仅记录路径，不读取内容。"""
    private_key = Path.home() / ".ssh/id_ed25519"
    public_key = private_key.with_suffix(".pub")
    for path in (private_key, public_key):
        if not path.is_file():
            raise ValueError(f"SSH key file not found: {path}")
    key = public_key.read_text(encoding="utf-8").strip()
    if (
        len(key.splitlines()) != 1
        or len(key.split()) < 2
        or not key.startswith("ssh-ed25519 ")
    ):
        raise ValueError(f"Expected a single Ed25519 public key in {public_key}")
    user = {
        "name": "jiangyinzuo",
        "authorized_keys": key,
        "ssh_key_files": [
            {"src": str(private_key), "name": "id_ed25519", "mode": "0600"},
            {"src": str(public_key), "name": "id_ed25519.pub", "mode": "0644"},
        ],
    }
    return [{**target, "users": [user]} for target in targets]


def describe_plan(plan, mode):
    """输出账号及密钥源路径，不输出密钥内容。"""
    return {
        "mode": mode,
        "users": [
            {k: v for k, v in user.items() if k != "authorized_keys"}
            for user in (plan[0]["users"] if plan else [])
        ],
        "hosts": [
            {
                "host": target["host"],
                "group": target["group"],
            }
            for target in plan
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--group", action="append", required=True)
    parser.add_argument("--host-glob", action="append")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    return args


def host_variables(target):
    private_key = str(Path(target["public_key"]).with_suffix(""))
    if target["password_method"] == "windows-askpass":
        private_key = subprocess.check_output(
            ["wslpath", "-w", private_key], text=True
        ).strip()
    return {
        "ansible_python_interpreter": target["python"],
        "managed_users": target["users"],
        "ansible_user": "root",
        "ansible_become": False,
        "ansible_private_key_file": private_key,
    }


def run_playbook(plan, args):
    """隔离临时 inventory，并返回 Ansible 退出码。"""
    inventory_data = build_inventory(plan, host_variables)
    with tempfile.TemporaryDirectory(prefix="createuser-") as directory:
        # inventory 只含公钥和私钥路径；临时目录隔离访问，运行后自动清理。
        inventory = Path(directory) / "inventory.json"
        inventory.write_text(json.dumps(inventory_data), encoding="utf-8")
        inventory.chmod(0o600)
        command = [
            "ansible-playbook",
            "-i",
            str(inventory),
            str(SCRIPT_DIR / "createuser.yml"),
            "--forks",
            str(args.workers),
        ]
        if args.check:
            command.append("--check")
        return subprocess.run(command, check=False).returncode


def main():
    args = parse_args()
    try:
        plan = build_plan(load_targets(args))
        mode = "apply" if args.apply else "check" if args.check else "plan"
        print(
            json.dumps(describe_plan(plan, mode), ensure_ascii=False, indent=2),
            flush=True,
        )
        if not args.apply and not args.check:
            return 0
        for target in plan:
            ensure_root_access(target, check=args.check)
        return run_playbook(plan, args)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print("Error: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
