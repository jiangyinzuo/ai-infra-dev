"""容器平台选择与远程调度。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from ssh_common.config import NPU_PLATFORMS
from ssh_common.inventory import read_machines, machine_record
from ssh_common.remote import (
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
    from ssh_common.machine_probe import npu_platform

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
    row = machine_record(read_machines(path, ("NPUPlatform",)), args.host, path)
    npu = row.get("NPUPlatform", "").lower()
    if npu not in NPU_PLATFORMS:
        raise ValueError(f"{args.host}: 不支持或未探测到 NPU 平台 {npu!r}")
    if args.npu_platform and args.npu_platform != npu:
        raise ValueError("指定平台与机器 CSV 不一致")
    return run_remote_cli(
        args.host,
        ["containers", script, "--npu-platform", npu, *options],
    )
