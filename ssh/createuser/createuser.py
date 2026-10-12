#!/usr/bin/env python3
"""Batch-create Linux users through Ansible. Default: print a plan, no SSH.

User configuration (--users, default users.json beside this script):
    {"alice": {"public_keys": "~/.ssh/id_ed25519.pub"},
     "bob": {"comment": "Bob", "shell": "/bin/bash", "groups": ["developers"],
             "public_keys": ["../keys/bob-office.pub", "../keys/bob-dev.pub"]}}
    Each key is a username; its value is an options object ({} is allowed).
    public_keys accepts one file path or a list. Paths are relative to this JSON
    file; ~ is supported. Only public keys, never private keys. Each file must
    contain one key. Additional groups must already exist.

Usage:
    python3 ssh/createuser/createuser.py --group suzhou
    # Check mode connects to targets but does not apply changes:
    python3 ssh/createuser/createuser.py --group suzhou --check
    # Apply to one host first:
    python3 ssh/createuser/createuser.py --group suzhou --host-glob '*-130' --apply

Safety and semantics:
    --group is required and repeatable; --host-glob further restricts targets.
    Shared connection profiles: ssh/hosts.json (--profiles can override).
    Default plan performs local configuration discovery only, no Ansible/SSH.
    Existing users are updated only for explicitly configured shell/comment/groups;
    existing supplementary groups and authorized keys are preserved.
    Missing or empty groups leave supplementary groups untouched; nonempty groups
    are appended, never used to replace existing memberships.
    Creates home directories and grants full passwordless sudo (NOPASSWD: ALL)
    to every configured user, including existing accounts. No password/UID/primary
    group changes. Each root-owned 0440 sudoers drop-in is validated with visudo;
    sudo must already be installed and sudoers must include /etc/sudoers.d.
    Only the managed drop-in is validated; no global visudo -c check is run.
    Existing sudoers issues are not repaired and may affect effective privileges.
    The SSH private key therefore grants root access; protect it accordingly.
    Missing public_keys uses the target group's public_key from ssh/hosts.json;
    an explicit path/list overrides it, and [] disables key installation.
    ${WINDOWS_SSH_HOME} and ${WSL_SSH_HOME} resolve controller SSH directories.
    Existing accounts may retain password access; new accounts use the OS default.
    --check queries accounts/groups, checks requested executable shells and validates
    proposed sudo rules via visudo stdin without installing them. It previews account,
    sudoers and existing-user key changes; new-user key checks are explicitly deferred
    until the real home/account exists. It does not prove effective sudo privileges.
    --ask-become-pass prompts for sudo authentication when necessary.
    Exit code is Ansible's exit code; default plan exits 0, configuration errors 1.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from ssh_hosts import DEFAULT_PROFILES, build_inventory, load_targets


NAME = re.compile(r"[a-z_][a-z0-9_-]*\$?", re.ASCII)
FIELDS = {"comment", "shell", "groups", "public_keys"}


def load_users(path):
    path = path.expanduser().resolve()
    users = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(users, dict) or not users:
        raise ValueError("User configuration must be a nonempty username-to-options object")
    prepared = []
    for name, user in users.items():
        if not isinstance(user, dict) or set(user) - FIELDS:
            raise ValueError("Each username must map to an object with supported fields only")
        if not isinstance(name, str) or not NAME.fullmatch(name) or name == "root":
            raise ValueError("Invalid or reserved username: " + str(name))
        entry = {"name": name, **{key: value for key, value in user.items() if key != "public_keys"}}
        for field in ("comment", "shell"):
            if field in user and (not isinstance(user[field], str) or "\n" in user[field] or "\r" in user[field]):
                raise ValueError(f"{name}: invalid {field}")
        if "shell" in user and not user["shell"].startswith("/"):
            raise ValueError(f"{name}: shell must be an absolute path")
        groups = user.get("groups", [])
        if not isinstance(groups, list) or any(not isinstance(g, str) or not NAME.fullmatch(g) for g in groups):
            raise ValueError(f"{name}: groups must be an array of group names")
        files = user.get("public_keys")
        if isinstance(files, str):
            files = [files]
        if files is not None and (not isinstance(files, list) or any(not isinstance(f, str) for f in files)):
            raise ValueError(f"{name}: public_keys must be a file path or an array of file paths")
        entry["public_keys"] = files
        prepared.append(entry)
    return prepared


def plan_users(users, base_dir, default_public_key, key_cache):
    prepared = []
    for user in users:
        entry = {k: v for k, v in user.items() if k != "public_keys"}
        files = user["public_keys"]
        if files is None:
            files = [default_public_key] if default_public_key else []
        keys = []
        for filename in files:
            path = Path(filename).expanduser()
            if not path.is_absolute():
                path = base_dir / path
            path = path.resolve()
            if path not in key_cache:
                key = path.read_text(encoding="utf-8").strip()
                if "PRIVATE KEY" in key or len(key.splitlines()) != 1 or len(key.split()) < 2:
                    raise ValueError(f"{user['name']}: expected a single SSH public key in {path}")
                if not key.startswith(("ssh-", "ecdsa-", "sk-")):
                    raise ValueError(f"{user['name']}: unsupported SSH public key in {path}")
                key_cache[path] = key
            keys.append(key_cache[path])
        entry["authorized_keys"] = "\n".join(keys)
        prepared.append(entry)
    return prepared


def build_plan(targets, users, base_dir):
    cache = {}
    return [{**target, "users": plan_users(users, base_dir, target.get("public_key"), cache)}
            for target in targets]


def describe_plan(plan, mode):
    return {"mode": mode, "hosts": [
        {"host": target["host"], "group": target["group"], "users": [
            {k: v for k, v in user.items() if k != "authorized_keys"}
            | {"public_key_count": len(user["authorized_keys"].splitlines())}
            for user in target["users"]]} for target in plan]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--group", action="append", required=True)
    parser.add_argument("--host-glob", action="append")
    parser.add_argument("--users", type=Path, default=SCRIPT_DIR / "users.json",
                        help="User JSON configuration (default: users.json beside this script)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--ask-become-pass", action="store_true")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    try:
        users = load_users(args.users)
        targets = load_targets(args)
        plan = build_plan(targets, users, args.users.expanduser().resolve().parent)
        inventory_data = build_inventory(plan, lambda target: {
            "ansible_python_interpreter": target["python"], "managed_users": target["users"],
        })
        mode = "apply" if args.apply else "check" if args.check else "plan"
        print(json.dumps(describe_plan(plan, mode), ensure_ascii=False, indent=2), flush=True)
        if not args.apply and not args.check:
            return 0
        with tempfile.TemporaryDirectory(prefix="createuser-") as directory:
            directory = Path(directory)
            inventory = directory / "inventory.json"
            inventory.write_text(json.dumps(inventory_data), encoding="utf-8")
            inventory.chmod(0o600)
            command = ["ansible-playbook", "-i", str(inventory), str(SCRIPT_DIR / "createuser.yml"),
                       "--forks", str(args.workers)]
            if args.check:
                command.append("--check")
            if args.ask_become_pass:
                command.append("--ask-become-pass")
            return subprocess.run(command, check=False).returncode
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print("Error: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
