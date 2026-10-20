"""Read bindfs identities; privileged repairs must not ignore unreadable evidence."""

from pathlib import Path


def scan_bindfs(uid, gid, *, strict=False, proc=Path("/proc")):
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, UnicodeError):
            if strict:
                raise
            continue
        if (
            len(args) >= 3
            and Path(args[0]).name == "bindfs"
            and "--create-for-user=" + str(uid) in args
            and "--create-for-group=" + str(gid) in args
        ):
            yield {"pid": int(entry.name), "source": args[-2], "target": args[-1]}
