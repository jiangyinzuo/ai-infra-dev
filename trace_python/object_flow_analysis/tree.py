"""将对象流事件合并为调用树，保留源码布局和采集证据。"""
import ast
from collections import Counter
from dataclasses import dataclass, field
import linecache
from typing import Any


@dataclass
class TreeNode:
    label: str = ""
    children: dict[tuple[str, str, int | None], "TreeNode"] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    first: int = 0
    last: int = 0
    samples: int = 0


def display_site(event: dict[str, Any]) -> dict[str, Any]:
    site = event["site"]
    is_argument = event["kind"] == "argument" or (
        event["kind"] == "historical_carrier_use" and event.get("operation") == "argument")
    if is_argument and len(event["stack"]) >= 2:
        # 参数绑定在被调函数入口观察，阅读位置应指向调用者，而不是 def 行。
        caller = event["stack"][-2]
        return {**site, "file": caller["file"], "function": caller["function"],
                "line": caller["line"]}
    return site


def source_statements(filename: str) -> dict[int, str]:
    """AST 只定位范围；源码原样截取，保留语句内部缩进和注释。"""
    source = "".join(linecache.getlines(filename))
    source_lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=filename)
    except (SyntaxError, ValueError):
        return {}
    statements: dict[int, str] = {}
    span_widths: dict[int, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Return)):
            continue
        end = node.end_lineno or node.lineno
        if end == node.lineno:
            continue
        block = source_lines[node.lineno - 1:end]
        indent = block[0][:len(block[0]) - len(block[0].lstrip())]
        text = "\n".join(line[len(indent):] if line.startswith(indent) else line
                         for line in block)
        width = end - node.lineno
        for line in range(node.lineno, end + 1):
            # 嵌套语句可能覆盖同一行，取最窄的多行范围；等宽时保留首次遍历结果。
            previous_width = span_widths.get(line)
            if previous_width is None or width < previous_width:
                statements[line] = text
                span_widths[line] = width
    return statements


def statement_label(events: list[dict[str, Any]], sources: dict[str, dict[int, str]]) -> str:
    first = events[0]
    site = display_site(first)
    filename, line = site["file"], site["line"]
    if filename not in sources:
        sources[filename] = source_statements(filename)
    source = sources[filename].get(line) or linecache.getline(filename, line).strip()
    marker = "🚥" if any(event["kind"] == "capture" for event in events) else "🚩"
    kinds = Counter(event["kind"] for event in events)
    objects = dict.fromkeys(event["object_id"] for event in events)
    evidence = " ".join(f"{kind}={count}" for kind, count in kinds.items())
    identities = ",".join(f"O{identity}" for identity in objects)
    source_lines = (source or site.get('detail', '')).split("\n")
    source_lines[0] += (f" {marker}[{first['sequence']}] :{line} "
                        f"{evidence} objects={identities} "
                        f"events={first['sequence']}..{events[-1]['sequence']}")
    return "\n".join(source_lines)


def render_tree(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    root = TreeNode()
    sources: dict[str, dict[int, str]] = {}
    for event in records:
        node = root
        caller: int | None = None
        for frame in event["stack"]:
            key = (frame["file"], frame["function"], caller)
            label = f"{frame['function']} ({frame['file']})"
            if caller is not None:
                label += f" [caller:{caller}]"
            node = node.children.setdefault(key, TreeNode(label))
            node.samples += 1
            node.first = node.first or event["sequence"]
            node.last = event["sequence"]
            caller = frame["line"]
        node.events.append(event)
    lines = [
        "对象流调用树（观察路径并集，不是连续执行路径）",
        "samples=已保存事件栈数；同路径同源码行聚合，kind=N 为该类事件数；编号范围不保证连续；历史载体不保证当前持有目标。",
        f"events={len(records)} incomplete={summary.get('incomplete', False)}",
    ]
    pending = [(root, "", True)]
    while pending:
        node, prefix, last = pending.pop()
        if node is not root:
            suffix = (f" samples={node.samples} events={node.first}..{node.last}"
                      if node.samples else "")
            label_lines = node.label.split("\n")
            lines.append(f"{prefix}{'└─' if last else '├─'}{label_lines[0]}{suffix}")
            continuation = prefix + ("  " if last else "│ ")
            lines.extend(continuation + line for line in label_lines[1:])
            prefix += "  " if last else "│ "
        children = list(node.children.values())
        groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for event in node.events:
            site = display_site(event)
            groups.setdefault((site["file"], site["line"]), []).append(event)
        # 事件先转换为叶节点，再统一排序，避免混合两种条目形状。
        # 只聚合同一调用树路径和源码行；原始事件身份、顺序仍完整保存在 JSONL。
        children.extend(
            TreeNode(statement_label(events, sources), first=events[0]["sequence"],
                     last=events[-1]["sequence"])
            for events in groups.values()
        )
        children.sort(key=lambda child: child.first)
        # 用叶节点保存事件，让统一遍历保持原事件与子调用的首次出现顺序。
        pending.extend((child, prefix, index == len(children) - 1)
                       for index, child in reversed(list(enumerate(children))))
    return "\n".join(lines) + "\n"
