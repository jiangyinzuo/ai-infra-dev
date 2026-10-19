"""Raise only the verified user's bindfs daemon limits for a specific mount."""

from pathlib import Path
import resource
import sys

# 编译项目时会并发读写大量文件，bindfs 需为这些访问持有底层文件描述符。
# 将其 RLIMIT_NOFILE 软/硬限制至少设为 524288，避免默认 1024 导致 Too many open files。
NOFILE = 524288


def raise_limit(pid):
    soft, hard = resource.prlimit(pid, resource.RLIMIT_NOFILE)
    target = max(soft, NOFILE)
    resource.prlimit(pid, resource.RLIMIT_NOFILE, (target, max(hard, target)))


def repair(target, source, uid, gid):
    matched = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
            if (
                len(args) >= 3
                and Path(args[0]).name == "bindfs"
                and args[-2:] == [source, target]
                and "--create-for-user=" + uid in args
                and "--create-for-group=" + gid in args
            ):
                raise_limit(int(entry.name))
                matched += 1
        except (FileNotFoundError, ProcessLookupError):
            continue
    if matched != 1:
        raise ValueError(
            f"Expected one matching bindfs daemon, found {matched}: {target}"
        )


if __name__ == "__main__":
    repair(*sys.argv[1:])
