"""将每个进程的原始记录写入 JSONL、可读日志与调用路径树。"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

from .call_tree import CallTree, format_duration
from .representation import format_value


class Writer:
    """保留原始记录，并按保存的调用栈生成当前进程的路径树。"""

    def __init__(
        self, machine: TextIO, human: TextIO, tree_json: TextIO, tree_text: TextIO
    ) -> None:
        self.machine, self.human = machine, human
        self.tree_json, self.tree_text = tree_json, tree_text
        self.tree = CallTree()

    def emit(self, record: dict[str, Any]) -> None:
        if record["type"] == "event":
            self.tree.add(record)
        elif record["type"] == "summary":
            self.tree.write_json(self.tree_json, record)
            for name in ("run_id", "process_instance_id", "pid", "ppid", "process_name"):
                if name in record:
                    print(f"{name}={record[name]}", file=self.tree_text)
            self.tree.write_text(self.tree_text, record)
        print(
            json.dumps(record, ensure_ascii=False, allow_nan=False),
            file=self.machine,
            flush=True,
        )
        kind = record["type"]
        if kind == "event":
            identity = "".join(
                f"  {name}={record[name]}"
                for name in ("pid", "process_name", "thread_id")
                if name in record
            )
            print(
                f"\n[{record['sequence']}] {record['target']}  hit={record['hit']}{identity}",
                file=self.human,
            )
            for name, value in record["values"].items():
                print(
                    f"  {name} = {format_value(value)}",
                    file=self.human,
                )
            if record.get("caller_values"):
                stack = record["stack"]
                caller = stack[-2]["function"] if len(stack) > 1 else "<no Python caller>"
                print(f"  👆 caller={caller}", file=self.human)
                for name, value in record["caller_values"].items():
                    print(f"    {name} = {format_value(value)}", file=self.human)
            print("  调用栈（外层 → 内层）：", file=self.human)
            for frame in record["stack"]:
                print(
                    f"    {frame['file']}:{frame['line']}  {frame['function']}",
                    file=self.human,
                )
        elif kind == "summary":
            print(
                f"\n结束：{record['status']}；记录 {record['events']} 个事件",
                file=self.human,
            )
            print(
                f"运行时长：{format_duration(record['elapsed_seconds'])}",
                file=self.human,
            )
            for name, count in record["counts"].items():
                print(f"  {count:6d}  {name}", file=self.human)
            if record.get("error"):
                print(record["error"], file=self.human)
        else:
            print(f"{kind}: {json.dumps(record, ensure_ascii=False)}", file=self.human)
        self.human.flush()


@contextlib.contextmanager
def open_writer(prefix: Path) -> Iterator[Writer]:
    """保存 JSONL、文本、调用树 JSON 和调用树文本。"""
    with contextlib.ExitStack() as stack:
        def open_output(suffix: str) -> TextIO:
            path = Path(str(prefix) + suffix)
            return stack.enter_context(path.open("w", encoding="utf-8"))

        writer = Writer(
            open_output(".jsonl"),
            open_output(".txt"),
            open_output(".tree.json"),
            open_output(".tree.txt"),
        )
        try:
            yield writer
        finally:
            writer.machine.flush()
