"""仅转换配置范围内的源码模块，不写入源码或 pyc 缓存。"""
from contextlib import contextmanager
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder, SourceFileLoader
from pathlib import Path
from types import CodeType, ModuleType
from typing import Callable, Iterator, Sequence
import sys

from .backend import transform
from . import FlowConfig
from .protocol import Site


Dispatch = Callable[[str, Site, tuple[object, ...]], object]


class FlowLoader(SourceFileLoader):
    def __init__(self, name: str, path: str, config: FlowConfig, dispatch: Dispatch) -> None:
        super().__init__(name, path)
        self.config = config
        self.dispatch = dispatch

    def get_code(self, fullname: str) -> CodeType:
        # 忽略已有 pyc，并禁止把转换结果污染正常运行的缓存。
        source = self.get_data(self.path)
        code = compile(source, self.path, "exec", dont_inherit=True)
        origin = self.config.origin
        return transform(code, self.dispatch, (
            str((self.config.root / origin.file).resolve()),
            origin.function, origin.line, origin.variable,
        ), observe_containers=self.config.observe_containers)


class FlowFinder(MetaPathFinder):
    def __init__(self, config: FlowConfig, dispatch: Dispatch) -> None:
        self.config = config
        self.dispatch = dispatch
        self.root = config.root.resolve()
        self.scope = tuple((self.root / item).resolve() for item in config.include or ())

    def accepts(self, filename: str) -> bool:
        path = Path(filename).resolve()
        if not path.is_relative_to(self.root):
            return False
        if any(part in {".venv", "__pycache__", ".git"} for part in path.relative_to(self.root).parts):
            return False
        return not self.scope or any(path == item or path.is_relative_to(item) for item in self.scope)

    def find_spec(self, fullname: str, path: Sequence[str] | None = None,
                  target: ModuleType | None = None) -> ModuleSpec | None:
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is None or not isinstance(spec.loader, SourceFileLoader):
            return None
        if spec.origin and self.accepts(spec.origin):
            spec.loader = FlowLoader(fullname, spec.origin, self.config, self.dispatch)
            return spec
        return None


@contextmanager
def instrument_imports(config: FlowConfig, dispatch: Dispatch) -> Iterator[FlowFinder]:
    finder = FlowFinder(config, dispatch)
    old_path = sys.path[:]
    sys.meta_path.insert(0, finder)
    sys.path.insert(0, str(config.root.resolve()))
    try:
        yield finder
    finally:
        sys.meta_path.remove(finder)
        sys.path[:] = old_path
