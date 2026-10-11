#!/usr/bin/env python3
"""在控制端解析 npu-smi info；从 stdin 读取原表，向 stdout 输出精简行。

独立使用：npu-smi info | python3 npu_smi_summary.py
ChipProcesses 是每个 (NPU, Chip) 的可见唯一 PID 数，不是 AI Core 数。
npu-smi 的进程显示上限和容器可见范围可能使计数低于实际值。
"""

import argparse
import json
import re
import sys

from typing import TypedDict


class ChipUsage(TypedDict):
    npu: int
    chip: int
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


def parse_summary(text: str) -> list[ChipUsage]:
    """解析设备表和进程表，保留 Chip 顺序，按芯片去重 PID。"""
    rows: list[ChipUsage] = []
    pids: dict[tuple[int, int], set[int]] = {}
    card = None
    in_processes = False
    has_usage_header = False

    for line in text.splitlines():
        if "Process id" in line and "Process name" in line:
            in_processes = True
        if "AICore(%)" in line:
            has_usage_header = True
        if not line.strip().startswith("|"):
            continue
        fields = [field.strip() for field in line.strip().split("|")[1:-1]]
        if len(fields) < 3:
            continue
        ids = fields[0].split()
        if not ids or not ids[0].isdigit():
            continue

        if in_processes:
            if len(ids) == 2 and ids[1].isdigit() and fields[1].isdigit():
                key = (int(ids[0]), int(ids[1]))
                pids.setdefault(key, set()).add(int(fields[1]))
            continue

        if len(ids) >= 2 and not ids[1].isdigit():
            card = int(ids[0])
            continue
        if not BUS_ID.fullmatch(fields[1]):
            continue
        metrics = METRICS.fullmatch(" ".join(fields[2:]))
        if card is None or metrics is None:
            raise ValueError(f"无法解析 Chip 利用率和内存行：{line.strip()}")
        rows.append({
            "npu": card,
            "chip": int(ids[0]),
            "aicore": metrics["usage"],
            "memory": re.sub(r"\s*/\s*", " / ", metrics["memory"]),
            "hbm": re.sub(r"\s*/\s*", " / ", metrics["hbm"]),
            "processes": None,
        })

    if not has_usage_header or not rows:
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
    return (
        f'NPU={row["npu"]} Chip={row["chip"]} AICore={usage} '
        f'Memory={row["memory"]} MB HBM={row["hbm"]} MB '
        f'ChipProcesses={processes}'
    )


def idle_cards(rows: list[ChipUsage]) -> list[int]:
    """同一卡的所有已报告 Chip 都零利用率、零进程，才判定空闲。"""
    cards: dict[int, bool] = {}
    for row in rows:
        usage = row["aicore"]
        idle = usage not in ("N/A", "NA") and float(usage) == 0 and row["processes"] == 0
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
    lines = ["空闲卡最多的服务器 TOP10（所有 Chip 利用率为 0 且无可见进程）"]
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
    mode.add_argument("--json", action="store_true", help="输出结构化汇总供 Ansible 使用")
    mode.add_argument("--rank", action="store_true", help="读取主机汇总结果 JSON，输出 TOP10")
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
            print(json.dumps({"rows": rows, "lines": lines, "idle_cards": idle_cards(rows),
                              "total_cards": len({row["npu"] for row in rows})}))
        else:
            for line in lines:
                print(line)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
