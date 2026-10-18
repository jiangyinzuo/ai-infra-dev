"""Read-only inventory; listing never creates registry directories or locks."""

from ssh_common.config import REMOTE_USER
import argparse
from contextlib import redirect_stdout
import json
import sys

from ..core.mappings import MappingStore
from ..core.process import containers, managed
from ..core.residuals import scan
from ..core.presentation import print_mappings


def execute(args: argparse.Namespace) -> None:
    # JSON stdout is exclusively machine-readable; actual commands remain visible.
    with redirect_stdout(sys.stderr):
        snapshot = containers()
        mappings = scan(MappingStore(), snapshot)
    rows = []
    for container in snapshot:
        if not managed(container["Name"]):
            continue
        name = container["Name"].lstrip("/")
        rows.append(
            dict(
                name=name,
                status=container["State"]["Status"],
                image=container.get("Config", {}).get(
                    "Image", container.get("Image", "")
                ),
                mappings=[m["target"] for m in mappings if name in m["containers"]],
            )
        )
    if args.json:
        print(
            json.dumps(
                dict(containers=rows, mappings=mappings), ensure_ascii=False, indent=2
            )
        )
        return
    print(f"{REMOTE_USER} 容器：")
    if not rows:
        print("  无")
    for row in rows:
        print("  {name} [{status}] {image}".format(**row))
        for target in row["mappings"]:
            print("    bindfs: " + target)
    print_mappings(mappings)
