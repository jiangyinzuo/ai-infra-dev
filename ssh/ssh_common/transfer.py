"""Construct rsync transport options; callers retain transfer policy and execution."""

import shlex

from .remote import checked_command


def rsync_command(host, ssh, remote_command, options):
    """Keep caller flags unchanged, including partial-file and permission policy."""
    return [
        "rsync",
        *options,
        "-e",
        shlex.quote(str(ssh)),
        "--rsync-path",
        checked_command(host, remote_command),
    ]
