"""场景模块必须在 run() 内导入，才能在首次执行前完成插桩。"""
import ast
from pathlib import Path

from trace_python.object_flow_analysis import FlowConfig, Origin

ROOT = Path(__file__).resolve().parent


def run():
    from trace_python.object_flow_analysis.examples.subject import run as scenario
    scenario()


# 示例按语法查找起点，避免示例本身依赖手工维护的固定行号。
source = ast.parse((ROOT / "subject.py").read_text(encoding="utf-8"))
line = next(node.lineno for node in ast.walk(source)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "target"
                    for target in node.targets))
config = FlowConfig(
    root=ROOT,
    origin=Origin("subject.py", "run", line, "target"),
    run=run,
)
