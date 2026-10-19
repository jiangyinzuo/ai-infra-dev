"""Inspect upload destinations as the SSH user, without modifying the filesystem."""

import json
import os
from pathlib import Path
import sys
import stat


def public_access(path):
    # Check traditional other-execute bits on every existing ancestor; ACLs may differ.
    try:
        for parent in (path, *path.parents):
            if parent.exists() and not parent.stat().st_mode & stat.S_IXOTH:
                return False
        return True
    except PermissionError:
        return False


def inspect_path(value):
    path = Path(value)
    row = {
        "path": str(path),
        "symlink": path.is_symlink(),
        "exists": False,
        "contents": "未知",
        "read": False,
        "write": False,
        "enter": False,
        "usable": False,
    }
    row["public_access"] = public_access(path)
    try:
        path.stat()
        row["exists"] = True
        row["read"] = os.access(path, os.R_OK)
        row["write"] = os.access(path, os.W_OK)
        row["enter"] = os.access(path, os.X_OK)
        if not path.is_dir():
            row["contents"] = "非目录"
            return row
        try:
            with os.scandir(path) as entries:
                row["contents"] = "非空" if next(entries, None) else "空"
        except PermissionError:
            row["contents"] = "不可读取"
        row["usable"] = row["write"] and row["enter"]
    except FileNotFoundError:
        row["contents"] = "不存在"
        if not row["symlink"]:
            parent = path.parent
            while not parent.exists() and parent != parent.parent:
                parent = parent.parent
            row["usable"] = parent.is_dir() and os.access(parent, os.W_OK | os.X_OK)
    except PermissionError:
        row["contents"] = "无权限"
    return row


def upload_paths(roots, model_name):
    return [
        {**inspect_path(root), "target": inspect_path(Path(root) / model_name)}
        for root in dict.fromkeys(roots)
    ]


if __name__ == "__main__":
    print(json.dumps(upload_paths(sys.argv[2:], sys.argv[1]), ensure_ascii=False))
