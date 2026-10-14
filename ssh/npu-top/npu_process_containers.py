#!/usr/bin/env python3
"""将 npu-smi 可见宿主机 PID 关联到 Docker 容器；只读，不进入容器。

只覆盖 npu-smi 报告的进程；无 cgroup 容器 ID 时显示未识别，不断言为宿主机进程。
不支持 containerd 元数据查询；只输出容器 ID 及 Docker 未识别状态。
"""

import json
from pathlib import Path
import re
import subprocess
import sys


def parse_processes(text):
    records = []
    active = False
    chip_level = False
    for line in text.splitlines():
        if "Process id" in line and "Process name" in line:
            active = True
            chip_level = "Chip" in line
            continue
        if not active or not line.strip().startswith("|"):
            continue
        fields = [field.strip() for field in line.strip().split("|")[1:-1]]
        if len(fields) < 4:
            continue
        ids = fields[0].split()
        if len(ids) != (2 if chip_level else 1) or not all(x.isdigit() for x in ids):
            continue
        if not fields[1].isdigit():
            continue
        records.append(
            {
                "npu": int(ids[0]),
                "chip": int(ids[1]) if chip_level else None,
                "pid": int(fields[1]),
                "name": fields[2],
                "memory": fields[3],
            }
        )
    return records


def process_owner(pid):
    try:
        cgroup = Path(f"/proc/{pid}/cgroup").read_text()
    except FileNotFoundError:
        return "进程已退出"
    except PermissionError:
        return "权限不足（/proc）"
    except OSError as error:
        return f"未识别（读取 cgroup：{error.strerror}）"
    match = re.search(r"(?:^|[/:-])([0-9a-f]{64})(?=[/.\n]|$)", cgroup)
    if match is None:
        return "未识别（无容器 ID）"
    container_id = match[1]
    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--type",
                "container",
                "--format",
                "{{json .Name}}",
                container_id,
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return f"{container_id[:12]}（Docker 不可用）"
    except subprocess.TimeoutExpired:
        return f"{container_id[:12]}（Docker 查询超时）"
    except PermissionError:
        return f"{container_id[:12]}（权限不足：Docker）"
    if result.returncode:
        reason = (
            "权限不足：Docker"
            if "permission denied" in result.stderr.lower()
            else "Docker 未识别"
        )
        return f"{container_id[:12]}（{reason}）"
    try:
        name = json.loads(result.stdout)
        if not isinstance(name, str) or not name:
            raise ValueError("empty name")
    except ValueError:
        return f"{container_id[:12]}（Docker 返回无效名称）"
    return f"{name.lstrip('/')} [{container_id[:12]}]"


def container_lines(text):
    records = parse_processes(text)
    owners = {
        pid: process_owner(pid) for pid in dict.fromkeys(row["pid"] for row in records)
    }
    lines = ["NPU / Chip / 宿主机PID / 进程 / 容器 / 进程显存(MB)"]
    for row in records:
        chip = row["chip"] if row["chip"] is not None else "—"
        lines.append(
            f"{row['npu']} / {chip} / {row['pid']} / {row['name']} / "
            f"{owners[row['pid']]} / {row['memory']}"
        )
    if not records:
        lines.append("npu-smi 未报告可见进程")
    return lines


if __name__ == "__main__":
    print(json.dumps(container_lines(sys.argv[1]), ensure_ascii=False))
