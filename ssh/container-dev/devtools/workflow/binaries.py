"""Download, synchronize, and extract platform tools."""

from pathlib import Path
import subprocess

from ssh_common.remote import add_host_argument, is_remote_host, is_wsl, run_remote_cli
from ssh_common.arguments import add_jobs_argument
from . import sync, binaries_download

from .paths import PLATFORM_DIR


def register(subparsers):
    parser = subparsers.add_parser("binaries", help=__doc__)
    actions = parser.add_subparsers(dest="binaries_action", required=True)
    for action in ("download", "extract"):
        command = actions.add_parser(action)
        add_host_argument(command)
        if action == "download":
            add_jobs_argument(command)
        command.set_defaults(handler=execute, command_parser=command)
    sync.register(actions)


def execute(args):
    parser = args.command_parser
    if args.binaries_action == "extract" and is_wsl() and not is_remote_host(args.host):
        parser.error("WSL requires --host HOST")
    if is_remote_host(args.host):
        remote_args = ["binaries", args.binaries_action]
        if args.binaries_action == "download":
            remote_args.extend(["--jobs", str(args.jobs)])
        return run_remote_cli(args.host, remote_args)
    if args.binaries_action == "download":
        return binaries_download.download(PLATFORM_DIR, args.jobs)
    return subprocess.run(
        ["bash", str(Path(__file__).with_name("extract_tools.sh")), str(PLATFORM_DIR)],
        check=False,
    ).returncode
