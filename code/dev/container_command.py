"""Build container commands without changing caller process or environment policy."""

from ssh_common.config import USER_IN_CONTAINER


def container_exec(container, arguments, *, interactive=False):
    return [
        "docker",
        "exec",
        "--user",
        USER_IN_CONTAINER,
        *(["-it"] if interactive else []),
        container,
        *arguments,
    ]
