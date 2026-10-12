#!/usr/bin/env python3
"""Collect Kubernetes status and /data* filesystem capacity via Ansible.

Examples:
    python3 ssh/machine-info/machine-info.py
    python3 ssh/machine-info/machine-info.py --group suzhou --host-glob '*-130'
    python3 ssh/machine-info/machine-info.py --format json --output /tmp/machines.json
    python3 ssh/machine-info/machine-info.py --output /tmp/machines.md

Prerequisites:
    Offline smoke: uv run --with pytest --with sshconf==0.2.7 python -m pytest -q ssh/machine-info/smoketest
    Remote: Python 3, GNU df; systemd for service states; SSH key authentication.
    USG: source ssh/ssh-get-usg-ip to export USG0_IP / USG1_IP first.
    No password injection, remote writes, sudo, or Kubernetes API queries.

Profiles (--profiles, default ssh/hosts.json, shared with npu-top):
    groups: list of objects with name, host_glob, config, ssh; optional
    ssh_config (client-native path, defaults to config) and python (default python3).
    config is read locally using the existing SSH parser: concrete Host aliases
    only, no Include or wildcard expansion. Commented hosts are not selected.
    Paths support ~ and ${ENV_VAR}; ${REPO} is this repository. The two
    ${WINDOWS_SSH_CONFIG...} paths are discovered via cmd.exe and wslpath
    only when a selected profile uses them. Profiles contain no passwords.
    machine-info.yml uses raw like npu-top; the shared ssh/ssh_hosts.py builds
    SSH inventory variables. --workers controls Ansible forks; --timeout limits
    each remote probe. Temporary inventory/results live only on the controller.

Output:
    OS: PRETTY_NAME from /etc/os-release (JSON: os.pretty_name).
    Kubernetes: installed / client-only / not-detected / unknown, component
    versions and systemd load/active/sub states. No node Ready claim.
    Storage: matching directories (including links), resolved paths, filesystem,
    mount, total/used/available capacity and use%. Capacity is df filesystem
    usage, NOT du directory usage. Shared filesystem rows must not be summed.
    Markdown uses GiB; JSON retains integer bytes and detailed evidence/errors.
    Exit 0: complete collection (stopped services are not collection errors).
    Exit 1: setup, host, or field collection error; partial results are retained.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from ssh_hosts import DEFAULT_PROFILES, inventory_group, load_targets


def parse_result(target, result):
    record = {"group": target["group"], "host": target["host"]}
    try:
        if result.get("unreachable") or result.get("failed") or result.get("rc", -1) != 0:
            raise ValueError(result.get("msg") or result.get("stderr") or "Probe failed")
        lines = [line[len("__MACHINE_INFO__"):] for line in result.get("stdout", "").splitlines()
                 if line.startswith("__MACHINE_INFO__")]
        if len(lines) != 1:
            raise ValueError("Missing or ambiguous probe response")
        record.update(json.loads(lines[0]))
    except ValueError as exc:
        record["error"] = str(exc)
    return record


def collect_all(targets, args):
    # Per-host groups keep connection settings independent of report group names.
    children = {}
    for index, target in enumerate(targets):
        children[f"machine_{index}"] = inventory_group(
            {target["host"]: {"machine_python": target["python"]}},
            target["ssh"], target["ssh_config"])
    with tempfile.TemporaryDirectory(prefix="machine-info-") as directory:
        directory = Path(directory)
        inventory = directory / "inventory.json"
        results = directory / "results.json"
        variables = directory / "vars.json"
        inventory.write_text(json.dumps({"all": {"children": children}}))
        variables.write_text(json.dumps({"results_path": str(results),
                                         "directory_glob": args.directory_glob,
                                         "probe_timeout": args.timeout}))
        execution = subprocess.run([
            "ansible-playbook", "-i", str(inventory), str(SCRIPT_DIR / "machine-info.yml"),
            "--forks", str(args.workers), "--timeout", "10", "-e", "@" + str(variables),
        ], capture_output=True, text=True,
            env={**os.environ, "ANSIBLE_STDOUT_CALLBACK": "default"})
        if not results.exists():
            raise ValueError("Ansible did not produce results:\n" + execution.stdout + execution.stderr)
        raw_results = json.loads(results.read_text())
        if execution.returncode:
            raise ValueError("Ansible execution failed:\n" + execution.stdout + execution.stderr)
        return [parse_result(target, raw_results.get(target["host"], {})) for target in targets]


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def capacity(value):
    return f"{value / (1024 ** 3):.2f} GiB"


def markdown(report):
    lines = ["# Machine information", "", "Collected: " + report["collected_at"], "",
             "Capacity is for each directory's filesystem, not directory contents. Shared filesystems must not be summed.",
             "Versions are local component versions; running kubelet does not imply node Ready. not-detected is not proof of absence.", "",
             "| Group | Host | OS | Kubernetes | Component versions | Service states |",
             "|---|---|---|---|---|---|"]
    for record in report["machines"]:
        if "error" in record:
            lines.append(f"| {cell(record['group'])} | {cell(record['host'])} | — | Query failed | — | {cell(record['error'])} |")
            continue
        k8s = record["kubernetes"]
        versions = "; ".join(f"{k}: {v}" for k, v in k8s["versions"].items()) or "—"
        states = "; ".join(f"{k}: {v.get('LoadState', 'unknown')}/{v.get('ActiveState', 'unknown')}/{v.get('SubState', 'unknown')}"
                           for k, v in k8s["services"].items() if v.get("LoadState") != "not-found") or "No detected services"
        lines.append("| " + " | ".join(map(cell, [record["group"], record["host"], record["os"]["pretty_name"] or "Unknown", k8s["status"], versions, states])) + " |")
    lines += ["", "## Data directories", "", "Pattern: `" + cell(report["directory_glob"]) + "`", "",
              "| Host | Directory | Resolved path | Filesystem | Mount | Total | Used | Available | Use% |",
              "|---|---|---|---|---|---|---|---|---|"]
    for record in report["machines"]:
        if "error" in record:
            continue
        for entry in record["storage"]["directories"]:
            values = [record["host"], entry["path"], entry["real_path"]]
            if "error" in entry:
                values += [entry["error"], "—", "—", "—", "—", "—"]
            else:
                values += [entry["filesystem"], entry["mount"], capacity(entry["total_bytes"]),
                           capacity(entry["used_bytes"]), capacity(entry["available_bytes"]), str(entry["use_percent"]) + "%"]
            lines.append("| " + " | ".join(map(cell, values)) + " |")
        if not record["storage"]["directories"]:
            lines.append(f"| {cell(record['host'])} | No matching directories | — | — | — | — | — | — | — |")
    errors = [record["host"] + ": " + message for record in report["machines"] if "error" not in record
              for section in ("os", "kubernetes", "storage") for message in record[section]["errors"]]
    if errors:
        lines += ["", "## Collection errors", "", *("- " + cell(error) for error in errors)]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--group", action="append", help="Select group; repeatable, default all")
    parser.add_argument("--host-glob", action="append", help="Further filter aliases; repeatable")
    parser.add_argument("--directory-glob", default="/data*")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path, help="Write report instead of stdout")
    args = parser.parse_args()
    if args.workers < 1 or args.timeout <= 0:
        parser.error("workers and timeout must be positive")
    try:
        targets = load_targets(args)
        machines = collect_all(targets, args)
        report = {"collected_at": datetime.now(timezone.utc).isoformat(),
                  "directory_glob": args.directory_glob, "machines": machines}
        text = json.dumps(report, ensure_ascii=False, indent=2) + "\n" if args.format == "json" else markdown(report)
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return int(any("error" in r or any(r[section]["errors"] for section in ("os", "kubernetes", "storage")) for r in machines))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print("Error: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
