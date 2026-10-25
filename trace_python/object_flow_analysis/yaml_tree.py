"""结构化字段与 literal 源码块，适合编辑器阅读和缩进折叠。"""
from typing import Any

import yaml

from .tree import build_tree, ordered_children


class TreeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        # 输出是阅读文档，不让共享摘要对象变成难读的 anchor/alias。
        return True

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super().increase_indent(flow, False)


def _string(dumper: TreeDumper, value: str):
    return dumper.represent_scalar('tag:yaml.org,2002:str', value,
                                   style='|' if '\n' in value else None)


TreeDumper.add_representer(str, _string)


def yaml_tree(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    roots: list[dict[str, Any]] = []
    sources: dict[str, dict[int, str]] = {}
    pending = [(build_tree(records), roots)]
    while pending:
        node, result = pending.pop()
        for child in ordered_children(node, sources):
            entry = dict(child.metadata)
            entry['event_range'] = [child.first, child.last]
            if child.samples:
                entry['samples'] = child.samples
                children: list[dict[str, Any]] = []
                entry['children'] = children
                pending.append((child, children))
            result.append(entry)
    document = {'tree': roots, 'summary': summary,
                'sample_semantics': '同路径同源码行聚合；samples 为事件栈数，非调用次数；编号范围不保证连续。'}
    return yaml.dump(document, Dumper=TreeDumper, allow_unicode=True,
                     sort_keys=False, default_flow_style=False, width=120)
