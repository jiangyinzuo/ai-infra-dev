#!/usr/bin/env python3
"""在 WSL 解析 npu-smi info；从 stdin 读取原表，向 stdout 输出精简行。

独立使用：npu-smi info | python3 npu_smi_summary.py
按表头区分芯片级 AICore 与设备级 NPU Util，不按工具版本或卡型号推断。
ChipProcesses 是每个 (NPU, Chip) 的可见唯一 PID 数，不是 AI Core 数。
设备级表输出 NPUUtil / Processes，不虚构 Chip 编号。
npu-smi 的进程显示上限和容器可见范围可能使计数低于实际值。
"""

import argparse
import json
import re
import sys

from typing import TypedDict


class ChipUsage(TypedDict):
    npu: int
    chip: int | None
    aicore: str
    memory: str
    hbm: str
    processes: int | None


BUS_ID = re.compile(r"[0-9A-Fa-f]+:[0-9A-Fa-f]+:[0-9A-Fa-f]+\.[0-9A-Fa-f]+")
METRICS = re.compile(
    r"(?P<usage>\d+(?:\.\d+)?|N/A|NA)\s+"
    r"(?P<memory>(?:\d+|N/A|NA)\s*/\s*(?:\d+|N/A|NA))\s+"
    r"(?P<hbm>(?:\d+|N/A|NA)\s*/\s*(?:\d+|N/A|NA))"
)


def usage_row(npu: int, chip: int | None, metrics: str) -> ChipUsage:
    match = METRICS.fullmatch(metrics)
    if match is None:
        raise ValueError(f"无法解析利用率和内存：{metrics}")
    return {
        "npu": npu,
        "chip": chip,
        "aicore": match["usage"],
        "memory": re.sub(r"\s*/\s*", " / ", match["memory"]),
        "hbm": re.sub(r"\s*/\s*", " / ", match["hbm"]),
        "processes": None,
    }


class ChipTableParser:
    """
    NPU / Chip / AICore(%) 表，进程按 (NPU, Chip) 计数。
    Ascend 910采用这种格式。
    """

    def __init__(self):
        self.card: int | None = None

    def parse_usage(self, fields: list[str]) -> ChipUsage | None:
        ids = fields[0].split()
        if not ids or not ids[0].isdigit():
            return None
        if len(ids) >= 2 and not ids[1].isdigit():
            self.card = int(ids[0])
            return None
        if not BUS_ID.fullmatch(fields[1]):
            return None
        if self.card is None:
            raise ValueError("Chip 利用率行之前缺少 NPU 编号")
        return usage_row(self.card, int(ids[0]), " ".join(fields[2:]))

    def process_key(self, fields: list[str]) -> tuple[int, int] | None:
        ids = fields[0].split()
        if len(ids) == 2 and all(value.isdigit() for value in ids):
            return int(ids[0]), int(ids[1])
        return None


class DeviceTableParser:
    """
    NPU ID / NPU Util(%) 表，进程按设备计数，无 Chip 列。
    Ascend 950PR采用这种格式。
    """

    def __init__(self):
        self.card: int | None = None

    def parse_usage(self, fields: list[str]) -> ChipUsage | None:
        if fields[0].isdigit():
            self.card = int(fields[0])
            return None
        if fields[0] or len(fields) < 4:
            return None
        if self.card is None:
            raise ValueError("设备利用率行之前缺少 NPU ID")
        return usage_row(self.card, None, " ".join(fields[3:]))

    def process_key(self, fields: list[str]) -> tuple[int, None] | None:
        if fields[0].isdigit():
            return int(fields[0]), None
        return None


def parse_summary(text: str) -> list[ChipUsage]:
    """按表头选择解析器，不依赖版本号或型号。"""
    parser: ChipTableParser | DeviceTableParser
    if "NPU Util(%)" in text:
        parser = DeviceTableParser()
    elif "AICore(%)" in text:
        parser = ChipTableParser()
    else:
        raise ValueError("未识别到 NPU 利用率表，请检查 npu-smi info 输出格式。")
    rows: list[ChipUsage] = []
    pids: dict[tuple[int, int | None], set[int]] = {}
    in_processes = False
    for line in text.splitlines():
        if "Process id" in line and "Process name" in line:
            in_processes = True
        if not line.strip().startswith("|"):
            continue
        fields = [field.strip() for field in line.strip().split("|")[1:-1]]
        if len(fields) < 3 or "Memory-Usage(MB)" in line:
            continue
        if in_processes:
            key = parser.process_key(fields)
            if key is not None and fields[1].isdigit():
                pids.setdefault(key, set()).add(int(fields[1]))
        else:
            row = parser.parse_usage(fields)
            if row is not None:
                rows.append(row)
    if not rows:
        raise ValueError("未识别到 NPU 利用率表，请检查 npu-smi info 输出格式。")
    for row in rows:
        key = (row["npu"], row["chip"])
        row["processes"] = len(pids.get(key, ())) if in_processes else None
    return rows


def format_row(row: ChipUsage) -> str:
    usage = row["aicore"]
    if usage not in ("N/A", "NA"):
        usage += "%"
    processes = row["processes"] if row["processes"] is not None else "N/A"
    device = f"NPU={row['npu']}"
    metric = "NPUUtil" if row["chip"] is None else "AICore"
    process_label = "Processes" if row["chip"] is None else "ChipProcesses"
    if row["chip"] is not None:
        device += f" Chip={row['chip']}"
    return (
        f"{device} {metric}={usage} "
        f"Memory={row['memory']} MB HBM={row['hbm']} MB "
        f"{process_label}={processes}"
    )


def idle_cards(rows: list[ChipUsage]) -> list[int]:
    """同一卡的所有已报告利用率都为零且无可见进程，才判定空闲。"""
    cards: dict[int, bool] = {}
    for row in rows:
        usage = row["aicore"]
        idle = (
            usage not in ("N/A", "NA") and float(usage) == 0 and row["processes"] == 0
        )
        cards[row["npu"]] = cards.get(row["npu"], True) and idle
    return sorted(card for card, idle in cards.items() if idle)


def rank_hosts(results: dict) -> list[str]:
    ranked = []
    failed = []
    for host, result in results.items():
        if result.get("rc") != 0:
            failed.append(host)
            continue
        summary = json.loads(result["stdout"])
        ranked.append((host, summary["idle_cards"], summary["total_cards"]))
    ranked.sort(key=lambda item: (-len(item[1]), item[0]))
    lines = ["空闲卡最多的服务器 TOP10（设备及已报告 Chip 利用率为 0 且无可见进程）"]
    for index, (host, cards, total) in enumerate(ranked[:10], 1):
        ids = ",".join(map(str, cards)) or "-"
        lines.append(f"{index:2}. {host} IdleCards={len(cards)}/{total} NPU=[{ids}]")
    if not ranked:
        lines.append("无成功查询的服务器")
    if failed:
        lines.append("查询失败或未获得有效汇总：" + ", ".join(sorted(failed)))
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--json", action="store_true", help="输出结构化汇总供 Ansible 使用"
    )
    mode.add_argument(
        "--rank", action="store_true", help="读取主机汇总结果 JSON，输出 TOP10"
    )
    args = parser.parse_args()
    try:
        text = sys.stdin.read()
        if args.rank:
            for line in rank_hosts(json.loads(text)):
                print(line)
            return 0
        rows = parse_summary(text)
        lines = [format_row(row) for row in rows]
        if args.json:
            print(
                json.dumps(
                    {
                        "rows": rows,
                        "lines": lines,
                        "idle_cards": idle_cards(rows),
                        "total_cards": len({row["npu"] for row in rows}),
                    }
                )
            )
        else:
            for line in lines:
                print(line)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
