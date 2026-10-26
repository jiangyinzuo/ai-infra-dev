"""将已保存的调用栈合并为路径树，不推断未采集的调用。"""
import json
from dataclasses import dataclass, field
from typing import Any, TextIO

def format_duration(seconds: float) -> str:
    """先舍入到毫秒再拆单位，避免出现 60.000 秒的进位显示。"""
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, fraction = divmod(remainder, 1000)
    parts = []
    if hours:
        parts.append(f"{hours}小时")
    if minutes or hours:
        parts.append(f"{minutes}分")
    parts.append(f"{whole_seconds}.{fraction:03d}秒")
    return " ".join(parts)



@dataclass
class Node:
    file: str = ""
    function: str = ""
    caller_line: int | None = None
    samples: int = 0
    first_sequence: int = 0
    last_sequence: int = 0
    first_target_sequence: int = 0
    targets: dict[str, int] = field(default_factory=dict)
    children: dict[tuple[str, str, int | None], "Node"] = field(default_factory=dict)


class CallTree:
    def __init__(self) -> None:
        self.root = Node()

    def add(self, record: dict[str, Any]) -> None:
        node = self.root
        caller_line = None
        sequence = record["sequence"]
        for frame in record["stack"]:
            # 使用调用者行号标识入边；节点自身暂停行会随采集深度变化，不能作为身份。
            key = (frame["file"], frame["function"], caller_line)
            node = node.children.setdefault(key, Node(*key))
            node.samples += 1
            if not node.first_sequence:
                node.first_sequence = sequence
            node.last_sequence = sequence
            caller_line = frame["line"]
        if not node.first_target_sequence:
            node.first_target_sequence = sequence
        node.targets[record["target"]] = node.targets.get(record["target"], 0) + 1

    def write_json(self, output: TextIO, summary: dict[str, Any]) -> None:
        def encode(parent: Node) -> dict[str, Any]:
            result = {}
            for node in sorted(parent.children.values(), key=lambda child: child.first_sequence):
                location = node.file
                if node.caller_line is not None:
                    location += f"; caller:{node.caller_line}"
                marker = f" 🎯[{node.first_target_sequence}]" if node.targets else ""
                label = (f"{node.function}{marker} ({location}) "
                         f"samples={node.samples} events={node.first_sequence}..{node.last_sequence}")
                result[label] = encode(node)
            return result
        record = {
            "tree": encode(self.root),
            "summary": summary,
            "sample_semantics": "samples=已保存栈数，不是调用次数；events=事件序号范围；🎯[n]=该路径首次目标事件序号；兄弟分支按子树最早事件排序；caller=调用者行号",
        }
        json.dump(record, output, ensure_ascii=False, allow_nan=False, indent=2)
        output.write("\n")
        output.flush()

    def write_text(self, output: TextIO, summary: dict[str, Any]) -> None:
        print("调用路径树（已保存事件的路径并集；按首次出现排列）", file=output)
        print(f"trace_started_at={summary['started_at']}", file=output)
        print(f"trace_command={summary.get('command', '')}", file=output)
        print(f"trace_cwd={summary.get('cwd', '')}", file=output)
        print(f"运行时长：{format_duration(summary['elapsed_seconds'])}", file=output)
        print("samples=经过节点的已保存栈数，不是函数调用次数；events=事件序号范围，不表示连续命中。", file=output)
        print("🎯[n]=该路径首次作为采集终点的事件序号；caller=调用者源码行号。循环折叠，递归按深度保留。", file=output)
        print(f"status={summary['status']} events={summary['events']}\n", file=output)
        pending = [(node, "", i == len(self.root.children) - 1)
                   for i, node in enumerate(self.root.children.values())][::-1]
        while pending:
            node, prefix, last = pending.pop()
            branch = "└─" if last else "├─"
            source = f" [caller:{node.caller_line}]" if node.caller_line is not None else ""
            target = f" 🎯[{node.first_target_sequence}]" if node.targets else ""
            print(f"{prefix}{branch}{node.function}{target} ({node.file}){source} "
                  f"samples={node.samples} events={node.first_sequence}..{node.last_sequence}", file=output)
            child_prefix = prefix + ("  " if last else "│ ")
            children = list(node.children.values())
            pending.extend((child, child_prefix, i == len(children) - 1)
                           for i, child in reversed(list(enumerate(children))))
        if not self.root.children:
            print("（没有保存的调用栈；检查 start()、目标匹配与 limit）", file=output)
        print("\n目标完整命中数（包含超过 limit、未保存调用栈的命中）：", file=output)
        for target, count in summary["counts"].items():
            print(f"{count:8d}  {target}", file=output)
        if summary.get("error"):
            print(f"\nerror: {summary['error']}", file=output)
        output.flush()
