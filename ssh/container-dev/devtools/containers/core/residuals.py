"""Read-only discovery first; remove only unreferenced, attributable mappings."""

import json
import os
from pathlib import Path

from .process import containers
from .mappings import (
    mount_table,
    bindfs_processes,
    referencing_containers,
    is_bindfs_mount,
    remove_mapping_entry,
)


def references(target: Path, snapshot: list) -> list:
    return sorted(
        {c["Name"].lstrip("/") for c in referencing_containers(target, snapshot)}
    )


def scan(store, snapshot: list) -> list:
    mounts = mount_table()
    processes = bindfs_processes()
    candidates = {}
    for metadata in store.root.glob("*/mapping.json"):
        record = json.loads(metadata.read_text())
        if record.get("uid") == os.getuid() and record.get("gid") == os.getgid():
            target = metadata.parent / "view"
            candidates[str(target)] = (record["source"], "managed")
    for target, source in processes.items():
        p = Path(target)
        if (
            p.parent.parent == Path("/var/tmp")
            and p.parent.name.startswith("root-mapped-")
            and p.name.isdigit()
        ):
            if not p.parent.is_symlink() and p.parent.stat().st_uid == os.getuid():
                candidates[target] = (source, "legacy")
    for target, (fstype, source) in mounts.items():
        p = Path(target)
        if (
            p.parent.parent == Path("/var/tmp")
            and p.parent.name.startswith("root-mapped-")
            and p.name.isdigit()
            and fstype.startswith("fuse")
            and p.parent.exists()
            and p.parent.stat().st_uid == os.getuid()
        ):
            candidates.setdefault(target, (source, "unverified-legacy"))
    rows = []
    for target, (source, kind) in sorted(candidates.items()):
        mount = mounts.get(target)
        users = references(Path(target), snapshot)
        verified = mount is None or is_bindfs_mount(
            Path(target), Path(source), mount, processes
        )
        rows.append(
            dict(
                target=target,
                source=source,
                kind=kind,
                mounted=mount is not None,
                source_exists=Path(source).exists(),
                containers=users,
                cleanup_allowed=verified and not users and kind != "unverified-legacy",
            )
        )
    return rows


def cleanup(store, rows: list) -> None:
    for old in rows:
        # Recheck Docker references and mount/process identity immediately before removal.
        current = {r["target"]: r for r in scan(store, containers())}.get(old["target"])
        if not current or not current["cleanup_allowed"]:
            print("保留（仍有引用或无法验证）：" + old["target"])
            continue
        target = Path(current["target"])
        remove_mapping_entry(target, current["mounted"], current["kind"] == "managed")
