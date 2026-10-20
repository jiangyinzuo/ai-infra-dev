"""Raise only the verified user's bindfs daemon limits for a specific mount."""

import resource
import sys

if __package__:
    from .bindfs_processes import scan_bindfs
else:
    from bindfs_processes import scan_bindfs

# 编译项目时会并发读写大量文件，bindfs 需为这些访问持有底层文件描述符。
# 将其 RLIMIT_NOFILE 软/硬限制至少设为 524288，避免默认 1024 导致 Too many open files。
NOFILE = 524288


def raise_limit(pid):
    soft, hard = resource.prlimit(pid, resource.RLIMIT_NOFILE)
    target = max(soft, NOFILE)
    resource.prlimit(pid, resource.RLIMIT_NOFILE, (target, max(hard, target)))


def repair(target, source, uid, gid):
    matched = 0
    for process in scan_bindfs(uid, gid, strict=True):
        if process["source"] == source and process["target"] == target:
            try:
                raise_limit(process["pid"])
                matched += 1
            except ProcessLookupError:
                continue
    if matched != 1:
        raise ValueError(
            f"Expected one matching bindfs daemon, found {matched}: {target}"
        )


if __name__ == "__main__":
    repair(*sys.argv[1:])
