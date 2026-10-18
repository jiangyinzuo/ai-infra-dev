"""容器平台选择与远程调度。"""

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT.parent))
from config import NPU_PLATFORMS
from remote_exec import (
    add_host_argument,
    host_alias,
    is_remote_host,
    is_wsl,
    run_remote_cli,
)


def add_target_args(parser):
    add_host_argument(parser)
    parser.add_argument(
        "--npu-platform",
        choices=NPU_PLATFORMS,
        help="覆盖本地 NPU 探测；WSL 本地下载必填",
    )


def detect_local_platform():
    sys.path.insert(0, str(ROOT.parent))
    from machine_probe import npu_platform

    result = npu_platform()
    if result["platform"] not in NPU_PLATFORMS:
        detail = "; ".join(result["errors"] or result["models"]) or result["status"]
        raise ValueError(f"无法识别本地 NPU 平台：{detail}；请指定 --npu-platform a3")
    return result["platform"]


def dispatch(args, script, options):
    wsl = is_wsl()
    if wsl and not args.host:
        raise ValueError("WSL 必须指定 --host（本地使用 localhost）")
    if not is_remote_host(args.host):
        if not args.npu_platform:
            if wsl:
                raise ValueError("WSL 本地下载必须指定 --npu-platform a3")
            args.npu_platform = detect_local_platform()
        return None
    host_alias(args.host)
    ssh_dir = ROOT.parents[2]

    path = ssh_dir / "machine-info/machines-output.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = {row["Host"]: row for row in csv.DictReader(stream)}
    row = rows.get(args.host)
    if row is None:
        raise ValueError(f"{path}: 没有主机 {args.host}")
    npu = row.get("NPUPlatform", "").lower()
    if npu not in NPU_PLATFORMS:
        raise ValueError(f"{args.host}: 不支持或未探测到 NPU 平台 {npu!r}")
    if args.npu_platform and args.npu_platform != npu:
        raise ValueError("指定平台与机器 CSV 不一致")
    return run_remote_cli(
        args.host,
        ["containers", script, "--npu-platform", npu, *options],
    )
