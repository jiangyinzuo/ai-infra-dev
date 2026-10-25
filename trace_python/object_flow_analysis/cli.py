"""运行配置场景并输出可审计的对象流事件和调用树。"""
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
import ast
from dataclasses import replace
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import FunctionType
from typing import Iterator

from . import FlowConfig
from .backend import unsupported
from .loading import instrument_imports
from .runtime import Runtime
from .tree import render_tree


@contextmanager
def scene_output() -> Iterator[None]:
    # 模型和原生扩展可能直接写 fd 1，不能仅 redirect_stdout。
    sys.stdout.flush()
    saved = os.dup(1)
    old_stdout = sys.stdout
    try:
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        yield
    finally:
        sys.stdout.flush()
        sys.stdout = old_stdout
        os.dup2(saved, 1)
        os.close(saved)


def load_config(path: Path) -> FlowConfig:
    spec = importlib.util.spec_from_file_location("_object_flow_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载配置：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.config
    if not isinstance(config, FlowConfig):
        raise TypeError("配置必须导出 FlowConfig 类型的 config")
    if not config.root.is_absolute() or not config.root.is_dir():
        raise ValueError("root 必须是已存在的绝对目录")
    if config.max_events <= 0 or config.max_objects <= 0:
        raise ValueError("资源上限必须大于 0")
    if config.origin.line <= 0 or not config.origin.variable:
        raise ValueError("起点必须指定正行号与变量名")
    origin_path = (config.root / config.origin.file).resolve()
    if not origin_path.is_relative_to(config.root.resolve()) or not origin_path.is_file():
        raise ValueError("起点文件必须位于 root 内")
    tree = ast.parse(origin_path.read_bytes(), filename=str(origin_path))
    matches = [node for node in ast.walk(tree)
               if isinstance(node, (ast.Assign, ast.AnnAssign))
               and node.lineno == config.origin.line
               and any(isinstance(target, ast.Name) and target.id == config.origin.variable
                       for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))]
    if len(matches) != 1:
        raise ValueError("起点必须唯一匹配普通变量赋值语句的起始行")
    return config


def execute(config: FlowConfig, output: Path) -> int:
    run = config.run
    if not isinstance(run, FunctionType):
        raise TypeError("run 必须是普通 Python 函数")
    runtime = Runtime(config.root, config.max_events, config.max_objects, run.__code__)
    error: BaseException | None = None
    old_cwd = Path.cwd()
    unsupported.clear()
    # 先确认可创建输出，避免昂贵场景结束后才发现输出目录不可写。
    with output.open("w", encoding="utf-8") as machine:
        try:
            with scene_output(), instrument_imports(config, runtime.dispatch):
                os.chdir(config.root)
                runtime.start()
                try:
                    run()
                finally:
                    runtime.stop()
        except BaseException as exc:
            error = exc
        finally:
            os.chdir(old_cwd)
        summary = runtime.summary()
        summary["type"] = "summary"
        summary["unsupported_bytecode"] = unsupported[:]
        summary["scope"] = config.include
        summary["root"] = str(config.root)
        summary["python"] = sys.version
        summary["observe_containers"] = config.observe_containers
        if error is not None:
            summary["error"] = {"type": type(error).__name__, "message": str(error)}
        captures = summary.get("captures", [])
        if not captures:
            summary["capture_warning"] = "没有捕获起点；确认函数、赋值起始行、变量名及场景导入发生在 run() 内"
        for record in runtime.records:
            machine.write(json.dumps({"type": "event", **record}, ensure_ascii=False) + "\n")
        machine.write(json.dumps(summary, ensure_ascii=False) + "\n")
    tree = Path(str(output) + ".tree.txt")
    tree.write_text(render_tree(runtime.records, summary), encoding="utf-8")
    print(f"events={len(runtime.records)} incomplete={summary.get('incomplete', False)}")
    print(f"事件：{output}\n调用树：{tree}")
    if error is not None:
        raise error
    return 0 if captures else 2


def parse_args(argv: list[str] | None = None) -> tuple[Namespace, ArgumentParser]:
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="覆盖写入 JSONL；另生成 .tree.txt")
    parser.add_argument("--observe-containers", action=BooleanOptionalAction, default=None,
                        help="观察容器与实例字段读写及浅复制；默认使用配置（开启）")
    return parser.parse_args(argv), parser


def main() -> int:
    if sys.implementation.name != "cpython" or not (3, 12) <= sys.version_info[:2] < (3, 15):
        raise RuntimeError("需要 CPython 3.12–3.14")
    args, _ = parse_args()
    with scene_output():
        config = load_config(args.config.resolve())
    if args.observe_containers is not None:
        config = replace(config, observe_containers=args.observe_containers)
    return execute(config, args.output.resolve())
