#!/usr/bin/env python3
"""从 SSH config 生成 Ansible JSON 主机清单（依赖 sshconf）。"""

import argparse
import fnmatch
import json
from pathlib import Path
import shlex

from ssh_config import host_passwords


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ssh-config", required=True)
    parser.add_argument("--ssh", required=True)
    parser.add_argument("--host-glob", required=True)
    parser.add_argument("--group", required=True)
    args = parser.parse_args()
    hosts = {host: {} for host in host_passwords(args.config)
             if fnmatch.fnmatchcase(host, args.host_glob)}
    if not hosts:
        parser.error(f"未找到匹配 {args.host_glob} 的具体 Host：{args.config}")
    print(json.dumps({"all": {"children": {args.group: {"hosts": hosts, "vars": {
        "ansible_connection": "ssh",
        "ansible_ssh_executable": args.ssh,
        "ansible_ssh_args": shlex.join([
            "-F", args.ssh_config, "-o", "BatchMode=yes", "-o", "ControlMaster=no",
            "-o", "StrictHostKeyChecking=accept-new",
        ]),
    }}}}}))


if __name__ == "__main__":
    main()
