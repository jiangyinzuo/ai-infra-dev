"""Download, synchronize, and extract platform tools."""

from pathlib import Path
import subprocess

from remote_exec import add_host_argument, is_wsl, run_remote
from . import sync, binaries_download

DEVTOOLS = Path(__file__).resolve().parents[1]
PLATFORM_DIR = DEVTOOLS / "openeuler24.03-lts-aarch64"


def register(subparsers):
    parser = subparsers.add_parser("binaries", help=__doc__)
    actions = parser.add_subparsers(dest="binaries_action", required=True)
    for action in ("download", "extract"):
        command = actions.add_parser(action)
        add_host_argument(command)
        if action == "download":
            command.add_argument("--jobs", type=int, default=8)
        command.set_defaults(handler=execute, command_parser=command)
    sync.register(actions)


def execute(args):
    parser = args.command_parser
    if args.binaries_action == "download" and args.jobs < 1:
        parser.error("--jobs requires a positive integer")
    if (
        args.binaries_action == "extract"
        and is_wsl()
        and (not args.host or args.host == "localhost")
    ):
        parser.error("WSL requires --host HOST")
    if args.host and args.host != "localhost":
        remote_args = ["--no-hints", "binaries", args.binaries_action]
        if args.binaries_action == "download":
            remote_args.extend(["--jobs", str(args.jobs)])
        return run_remote(args.host, "cli.py", remote_args)
    if args.binaries_action == "download":
        return binaries_download.download(PLATFORM_DIR, args.jobs)
    return subprocess.run(
        ["bash", str(Path(__file__).with_name("extract_tools.sh"))], check=False
    ).returncode
