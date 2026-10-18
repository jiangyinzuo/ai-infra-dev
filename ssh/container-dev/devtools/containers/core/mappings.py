"""Persistent bindfs mappings shared by create/delete; Python 3.9+."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
from .process import containers, run


class MappingStore:
    def __init__(self):
        self.root = Path("/var/tmp") / ("jiangyinzuo-bindfs-" + str(os.getuid()))

    @contextmanager
    def locked(self):
        self.root.mkdir(mode=0o700, exist_ok=True)
        if self.root.is_symlink() or self.root.stat().st_uid != os.getuid():
            raise ValueError("Invalid mapping registry owner: " + str(self.root))
        with (self.root / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield self

    def acquire(self, source: Path, bindfs: str) -> Path:
        source = source.resolve()
        if source == self.root or source in self.root.parents:
            raise ValueError("源目录不能包含映射注册目录：" + str(source))
        record = {
            "source": str(source),
            "uid": os.getuid(),
            "gid": os.getgid(),
            "policy": "ignore-chown-v1",
        }
        key = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[
            :24
        ]
        directory = self.root / key
        directory.mkdir(exist_ok=True)
        metadata = directory / "mapping.json"
        target = directory / "view"
        if metadata.exists():
            if json.loads(metadata.read_text()) != record:
                raise ValueError("Mapping metadata mismatch: " + str(directory))
        else:
            if target.exists():
                raise ValueError("Unmanaged mapping target: " + str(target))
            metadata.write_text(json.dumps(record, indent=2) + "\n")
        target.mkdir(exist_ok=True)
        if target.is_mount():
            self.validate_mount(target, source)
            print("reuse bindfs: {} -> {}".format(source, target), flush=True)
            return target
        if any(target.iterdir()):
            raise ValueError("Unmounted mapping directory is not empty: " + str(target))
        run(
            "sudo",
            "-n",
            bindfs,
            "--map={}/0:@{}/@0".format(record["uid"], record["gid"]),
            "--create-for-user=" + str(record["uid"]),
            "--create-for-group=" + str(record["gid"]),
            "--chown-ignore",
            "--chgrp-ignore",
            "-o",
            "allow_other,nosuid,nodev",
            str(source),
            str(target),
        )
        return target

    def validate_mount(self, target: Path, source: Path) -> None:
        info = json.loads(
            run(
                "findmnt",
                "-J",
                "-M",
                str(target),
                "-o",
                "SOURCE,FSTYPE",
                capture_output=True,
                text=True,
            ).stdout
        )
        mounts = info.get("filesystems", [])

        if (
            len(mounts) != 1
            or not (
                mounts[0]["fstype"] == "fuse.bindfs"
                or (
                    mounts[0]["fstype"] == "fuse"
                    and bindfs_processes().get(str(target)) == str(source)
                )
            )
            or mounts[0]["source"] != str(source)
        ):
            raise ValueError("Unexpected mount at " + str(target))

    def remove_unused(self, sources: list, remaining: list) -> None:
        references = [
            Path(m["Source"])
            for c in remaining
            for m in c.get("Mounts", [])
            if m.get("Type") == "bind"
        ]
        for target in dict.fromkeys(map(Path, sources)):
            # Only mappings created by this registry; never delete host source directories.
            if target.parent.parent != self.root or target.name != "view":
                continue
            metadata = target.parent / "mapping.json"
            if not metadata.is_file():
                raise ValueError("Missing mapping metadata: " + str(target))
            record = json.loads(metadata.read_text())
            if record["uid"] != os.getuid() or record["gid"] != os.getgid():
                raise ValueError("Mapping ownership mismatch: " + str(target))
            if any(
                p == target or target in p.parents or p in target.parents
                for p in references
            ):
                print("keep bindfs (referenced by another container): " + str(target))
                continue
            if target.is_mount():
                self.validate_mount(target, Path(record["source"]))
                run("sudo", "-n", "umount", str(target))
            run("rmdir", str(target))
            run("rm", "--", str(metadata))
            run("rmdir", str(target.parent))


def mount_table() -> dict:
    def decode(value):
        return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)

    result = {}
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        fields = right.split()
        result[decode(left.split()[4])] = (fields[0], decode(fields[1]))
    return result


def bindfs_processes() -> dict:
    result = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
        except (OSError, UnicodeError):
            continue
        if len(args) >= 3 and Path(args[0]).name == "bindfs":
            if (
                "--create-for-user=" + str(os.getuid()) in args
                and "--create-for-group=" + str(os.getgid()) in args
            ):
                result[args[-1]] = args[-2]
    return result
