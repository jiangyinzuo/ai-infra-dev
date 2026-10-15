#!/usr/bin/env python3
"""通过 Ansible 只读采集系统、NPU 平台、Kubernetes 状态及 /data* 文件系统容量。

在 ssh/ 目录执行：
  ./machine-info/machine-info.py --group suzhou --host-glob '*-130'
  ./machine-info/machine-info.py --group suzhou

输出到脚本目录：machines-output.md 保存本次报告，machines-output.csv 按 Host 更新 OS、Arch、NPUPlatform。
NPUPlatform 通过 sudo -n npu-smi info -t board -i 0 的 PCI Device ID 判断：D803=a3、D802=a2。
未查询或采集失败的数据保留；旧三列 CSV 自动升级，原有行的平台为 unknown。
缺少 npu-smi 不算采集错误；已知平台在检测失败或命令缺失时保留。
容量是文件系统用量，不是目录大小。
远端需 Python 3、GNU df 和 SSH 密钥登录；USG 连接前需设置 USG0_IP/USG1_IP。
退出码：0 表示采集完整，1 表示有错误（保留已采集结果）。
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from ssh_hosts import build_inventory, load_targets


def parse_result(target, result):
    record = {"group": target["group"], "host": target["host"]}
    try:
        if (
            result.get("unreachable")
            or result.get("failed")
            or result.get("rc", -1) != 0
        ):
            raise ValueError(
                result.get("msg") or result.get("stderr") or "Probe failed"
            )
        lines = [
            line[len("__MACHINE_INFO__") :]
            for line in result.get("stdout", "").splitlines()
            if line.startswith("__MACHINE_INFO__")
        ]
        if len(lines) != 1:
            raise ValueError("Missing or ambiguous probe response")
        record.update(json.loads(lines[0]))
    except ValueError as exc:
        record["error"] = str(exc)
    return record


def collect_all(targets, args):
    inventory_data = build_inventory(
        targets, lambda target: {"machine_python": target["python"]}
    )
    with tempfile.TemporaryDirectory(prefix="machine-info-") as directory:
        directory = Path(directory)
        inventory = directory / "inventory.json"
        results = directory / "results.json"
        variables = directory / "vars.json"
        inventory.write_text(json.dumps(inventory_data))
        variables.write_text(
            json.dumps(
                {
                    "results_path": str(results),
                    "directory_glob": args.directory_glob,
                    "probe_timeout": args.timeout,
                }
            )
        )
        execution = subprocess.run(
            [
                "ansible-playbook",
                "-i",
                str(inventory),
                str(SCRIPT_DIR / "machine-info.yml"),
                "--forks",
                str(args.workers),
                "--timeout",
                "10",
                "-e",
                "@" + str(variables),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "ANSIBLE_STDOUT_CALLBACK": "default"},
        )
        if not results.exists():
            raise ValueError(
                "Ansible did not produce results:\n"
                + execution.stdout
                + execution.stderr
            )
        raw_results = json.loads(results.read_text())
        if execution.returncode:
            raise ValueError(
                "Ansible execution failed:\n" + execution.stdout + execution.stderr
            )
        return [
            parse_result(target, raw_results.get(target["host"], {}))
            for target in targets
        ]


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def capacity(value):
    return f"{value / (1024**3):.2f} GiB"


def markdown(report):
    lines = [
        "# Machine information",
        "",
        "Collected: " + report["collected_at"],
        "",
        "Capacity is for each directory's filesystem, not directory contents. Shared filesystems must not be summed.",
        "Versions are local component versions; running kubelet does not imply node Ready. not-detected is not proof of absence.",
        "",
        "| Group | Host | OS | Arch | NPU platform | NPU model / evidence | Kubernetes | Service states |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for record in report["machines"]:
        if "error" in record:
            lines.append(
                f"| {cell(record['group'])} | {cell(record['host'])} | — | — | — | — | Query failed | {cell(record['error'])} |"
            )
            continue
        k8s = record["kubernetes"]
        versions = "; ".join(f"{k}: {v}" for k, v in k8s["versions"].items())
        kubernetes = k8s["status"] + ("; " + versions if versions else "")
        states = (
            "; ".join(
                f"{k}: {v.get('LoadState', 'unknown')}/{v.get('ActiveState', 'unknown')}/{v.get('SubState', 'unknown')}"
                for k, v in k8s["services"].items()
                if v.get("LoadState") != "not-found"
            )
            or "No detected services"
        )
        npu = record.get("npu", {})
        npu_evidence = "; ".join(npu.get("models", []))
        if npu.get("evidence"):
            npu_evidence += "; " + "; ".join(npu["evidence"])
        if not npu_evidence:
            npu_evidence = npu.get("status", "not-found")
        lines.append(
            "| "
            + " | ".join(
                map(
                    cell,
                    [
                        record["group"],
                        record["host"],
                        record["os"]["pretty_name"] or "Unknown",
                        record["arch"],
                        npu.get("platform", "unknown"),
                        npu_evidence,
                        kubernetes,
                        states,
                    ],
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Data directories",
        "",
        "Pattern: `" + cell(report["directory_glob"]) + "`",
        "",
        "| Host | Directory | Resolved path | Filesystem | Mount | Total | Used | Available | Use% |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for record in report["machines"]:
        if "error" in record:
            continue
        for entry in record["storage"]["directories"]:
            values = [record["host"], entry["path"], entry["real_path"]]
            if "error" in entry:
                values += [entry["error"], "—", "—", "—", "—", "—"]
            else:
                values += [
                    entry["filesystem"],
                    entry["mount"],
                    capacity(entry["total_bytes"]),
                    capacity(entry["used_bytes"]),
                    capacity(entry["available_bytes"]),
                    str(entry["use_percent"]) + "%",
                ]
            lines.append("| " + " | ".join(map(cell, values)) + " |")
        if not record["storage"]["directories"]:
            lines.append(
                f"| {cell(record['host'])} | No matching directories | — | — | — | — | — | — | — |"
            )
    errors = [
        record["host"] + ": " + message
        for record in report["machines"]
        if "error" not in record
        for section in ("os", "npu", "kubernetes", "storage")
        for message in record.get(section, {}).get("errors", [])
    ]
    if errors:
        lines += [
            "",
            "## Collection errors",
            "",
            *("- " + cell(error) for error in errors),
        ]
    return "\n".join(lines) + "\n"


def write_reports(report):
    """写入脚本目录：覆盖 Markdown，按 Host 合并 CSV。"""
    path = SCRIPT_DIR / "machines-output.csv"
    hosts = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            columns = ["Host", "OS", "Arch", "NPUPlatform"]
            if reader.fieldnames not in (columns[:3], columns):
                raise ValueError(f"{path}: expected Host,OS,Arch,NPUPlatform columns")
            hosts = {
                row["Host"]: (
                    row["OS"],
                    row["Arch"],
                    row.get("NPUPlatform") or "unknown",
                )
                for row in reader
            }
    for record in report["machines"]:
        host = record["host"]
        previous_os, previous_arch, previous_npu = hosts.get(
            host, ("Unknown", "Unknown", "unknown")
        )
        npu = record.get("npu", {})
        hosts[host] = (
            record.get("os", {}).get("pretty_name") or previous_os,
            record.get("arch") or previous_arch,
            npu.get("platform", "unknown")
            if npu.get("status") == "ok"
            else previous_npu,
        )
    (SCRIPT_DIR / "machines-output.md").write_text(markdown(report), encoding="utf-8")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Host", "OS", "Arch", "NPUPlatform"])
        writer.writerows((host, *values) for host, values in sorted(hosts.items()))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser._optionals.title = "参数"
    parser.add_argument("-h", "--help", action="help", help="显示帮助并退出")
    parser.add_argument("--group", action="append", help="选择组，可重复；默认全部")
    parser.add_argument(
        "--host-glob", action="append", help="按主机名通配符筛选，可重复"
    )
    parser.add_argument(
        "--directory-glob", default="/data*", help="目录通配符，默认 /data*"
    )
    parser.add_argument("--workers", type=int, default=8, help="并发数，默认 8")
    parser.add_argument(
        "--timeout", type=float, default=120, help="单机采集超时秒数，默认 120"
    )
    args = parser.parse_args()
    if args.workers < 1 or args.timeout <= 0:
        parser.error("并发数和超时必须大于 0")
    try:
        targets = load_targets(args)
        machines = collect_all(targets, args)
        report = {
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "directory_glob": args.directory_glob,
            "machines": machines,
        }
        write_reports(report)
        return int(
            any(
                "error" in r
                or any(
                    r.get(section, {}).get("errors", [])
                    for section in ("os", "npu", "kubernetes", "storage")
                )
                for r in machines
            )
        )
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print("Error: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
