"""读取机器采集结果；平台选择由调用方负责。"""

import csv
from pathlib import Path
from typing import Iterable


def read_machines(path: Path, required: Iterable[str]) -> dict:
    fields = tuple(dict.fromkeys(("Host", *required)))
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(fields).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected {','.join(fields)} columns")
        return {row["Host"]: row for row in reader}


def machine_record(machines: dict, host: str, path: Path) -> dict:
    if host not in machines:
        raise ValueError(f"{host}: 没有主机记录 (missing machine record) in {path}")
    return machines[host]
