"""Download, synchronize, and extract platform tools and Neovim plugins."""

from pathlib import Path
import subprocess

from ssh_common.remote import add_host_argument, is_remote_host, is_wsl, run_remote_cli
from ssh_common.arguments import add_jobs_argument
from . import sync, binaries_download, nvim_plugins

from .paths import DATA_ROOT, PLATFORM_DIR


def register(subparsers):
    parser = subparsers.add_parser("binaries", help=__doc__)
    actions = parser.add_subparsers(dest="binaries_action", required=True)
    for action in ("download", "extract"):
        command = actions.add_parser(action)
        add_host_argument(command)
        add_jobs_argument(command)
        if action == "download":
            command.add_argument("--lock", type=Path, help="Neovim plugin lock file")
            command.add_argument(
                "--repos", type=Path, help="JSON plugin repository overrides"
            )
        command.add_argument(
            "--output",
            type=Path,
            help="plugin archive directory (not tool binaries)",
        )
        command.set_defaults(handler=execute, command_parser=command)
    sync.register(actions)


def execute(args):
    parser = args.command_parser
    if args.binaries_action == "extract" and is_wsl() and not is_remote_host(args.host):
        parser.error("WSL requires --host HOST")
    if is_remote_host(args.host):
        remote_args = ["binaries", args.binaries_action]
        remote_args.extend(["--jobs", str(args.jobs)])
        for option in ("lock", "repos", "output"):
            value = getattr(args, option, None)
            if value is not None:
                remote_args.extend(["--" + option, str(value)])
        return run_remote_cli(args.host, remote_args)
    if args.binaries_action == "download":
        args.lock = args.lock or Path.home() / ".config/nvim/lazy-lock.json"
    args.output = args.output or DATA_ROOT / "nvim-plugins"
    failures = []
    result = 0
    try:
        if args.binaries_action == "download":
            result = binaries_download.download(PLATFORM_DIR, args.jobs)
        else:
            result = subprocess.run(
                [
                    "bash",
                    str(Path(__file__).with_name("extract_tools.sh")),
                    str(PLATFORM_DIR),
                ],
                check=False,
            ).returncode
        if result == 130:
            return 130
        if result:
            failures.append(f"platform tools failed (exit {result})")
    except Exception as error:
        failures.append(f"platform tools: {error}")
    try:
        if args.binaries_action == "download":
            nvim_plugins.download_plugins(args)
        else:
            nvim_plugins.extract_plugins(args)
    except Exception as error:
        failures.append(f"Neovim plugins: {error}")
    if failures:
        raise OSError("\n".join(failures))
    return 0
