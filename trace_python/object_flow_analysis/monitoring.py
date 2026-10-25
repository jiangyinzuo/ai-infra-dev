"""同步调用边界、线程过滤与 monitoring 资源生命周期。"""
from __future__ import annotations

import inspect
import sys
import threading
from pathlib import Path
from types import CodeType, FrameType, FunctionType, MethodType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime import Runtime

_SKIP_FLAGS = inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR
_MISSING = object()


class CallMonitor:
    def __init__(self, runtime: Runtime, entry_code: CodeType | None):
        self.runtime = runtime
        self._entry_code = entry_code
        self.frames: dict[int, int] = {}
        self._baseline: set[int] = set()
        # code 结构哈希可能遍历常量，按身份缓存并保留引用防止复用。
        self._code_project: dict[int, tuple[CodeType, bool]] = {}
        self._layouts: dict[int, tuple[tuple[str, ...], str | None, str | None]] = {}
        self._next_call = 0
        self.thread = 0
        self._tool: int | None = None
        self._native_calls: set[tuple[int, int]] = set()
        self._local_codes: dict[int, CodeType] = {}
        self._internal_files = {str(Path(__file__).with_name(name))
                                for name in ('runtime.py', 'monitoring.py', 'operations.py')}

    def project(self, code: CodeType) -> bool:
        known = self._code_project.get(id(code))
        if known is not None:
            return known[1]
        filename = code.co_filename
        result = (not filename.startswith('<') and
                  Path(filename).resolve().is_relative_to(self.runtime.root) and
                  filename not in self._internal_files)
        self._code_project[id(code)] = (code, result)
        return result


    def _allowed_frame(self, frame: FrameType) -> bool:
        if frame.f_code.co_flags & _SKIP_FLAGS or not self.project(frame.f_code):
            return False
        parent = frame.f_back
        while parent is not None:
            if parent.f_code is self._entry_code:
                return True
            if id(parent) in self.frames or id(parent) in self._baseline:
                return True
            if parent.f_code.co_filename in self._internal_files:
                parent = parent.f_back
                continue
            return False
        return True


    def _on_start(self, code: CodeType, offset: int) -> object:
        if not self.project(code):
            return sys.monitoring.DISABLE
        self._enable_calls(code)
        if not self.runtime._eligible():
            return
        frame = sys._getframe(1)
        if not self._allowed_frame(frame):
            return
        self._next_call += 1
        self.frames[id(frame)] = self._next_call
        if not self.runtime._objects:
            return
        layout = self._layouts.get(id(code))
        if layout is None:
            count = code.co_argcount + code.co_kwonlyargcount
            names = code.co_varnames[:count]
            varargs = code.co_varnames[count] if code.co_flags & inspect.CO_VARARGS else None
            index = count + (1 if varargs else 0)
            varkw = code.co_varnames[index] if code.co_flags & inspect.CO_VARKEYWORDS else None
            layout = (names, varargs, varkw)
            self._layouts[id(code)] = layout
        names, varargs, varkw = layout
        local = frame.f_locals
        if varargs:
            for position, value in enumerate(local.get(varargs, ())):
                if id(value) in self.runtime._objects:
                    self.runtime._emit('argument', value, (code.co_filename, code.co_qualname, frame.f_lineno, frame.f_lineno, 0, 0, f'{varargs}[{position}]'))
        if varkw:
            for key, value in local.get(varkw, {}).items():
                if id(value) in self.runtime._objects:
                    self.runtime._emit('argument', value, (code.co_filename, code.co_qualname, frame.f_lineno, frame.f_lineno, 0, 0, f'{varkw}[{key}]'))
        for name in names:
            value = local.get(name, _MISSING)
            if id(value) in self.runtime._objects:
                self.runtime._emit('argument', value, (code.co_filename, code.co_qualname, frame.f_lineno, frame.f_lineno, 0, 0, name))


    def _on_return(self, code: CodeType, offset: int, value: object) -> object:
        if not self.project(code):
            return sys.monitoring.DISABLE
        frame = sys._getframe(1)
        if id(frame) in self.frames:
            if self.runtime._eligible():
                self.runtime._emit('return', value, (code.co_filename, code.co_qualname, frame.f_lineno, frame.f_lineno, 0, 0, 'return'))
            self.frames.pop(id(frame), None)
        self._baseline.discard(id(frame))


    def _on_unwind(self, code: CodeType, offset: int, exception: BaseException) -> None:
        if not self.project(code):
            return
        frame_id = id(sys._getframe(1))
        self.frames.pop(frame_id, None)
        self._baseline.discard(frame_id)


    def _on_call(self, code: CodeType, offset: int, function: object, arg0: object) -> object:
        if not self.project(code):
            return sys.monitoring.DISABLE
        if not self.runtime._eligible() or type(function) in (FunctionType, MethodType):
            return
        if type(function) is type:
            initializer = inspect.getattr_static(function, '__init__', None)
            if type(initializer) is FunctionType and self.project(initializer.__code__):
                return
        frame = sys._getframe(1)
        if id(frame) in self.frames:
            self._native_calls.add((id(frame), offset))
            self.runtime._external_depth += 1


    def _on_c_exit(self, code: CodeType, offset: int, function: object, arg0: object) -> None:
        if not self.project(code):
            return
        key = (id(sys._getframe(1)), offset)
        if threading.get_ident() == self.thread and key in self._native_calls:
            self._native_calls.remove(key)
            self.runtime._external_depth -= 1


    def _enable_calls(self, code: CodeType) -> None:
        if self._tool is not None and id(code) not in self._local_codes:
            self._local_codes[id(code)] = code
            events = sys.monitoring.events
            sys.monitoring.set_local_events(self._tool, code, events.CALL | events.C_RETURN | events.C_RAISE)


    def start(self, frame: FrameType | None) -> None:
        if self.runtime.active:
            return
        monitoring = sys.monitoring
        tool = None
        for candidate in range(6):
            try:
                monitoring.use_tool_id(candidate, 'object_flow_analysis')
            except ValueError:
                continue
            tool = candidate
            break
        if tool is None:
            raise RuntimeError('No free sys.monitoring tool ID')
        self._tool = tool
        self.thread = threading.get_ident()
        while frame is not None:
            self._baseline.add(id(frame))
            if self.project(frame.f_code) and not frame.f_code.co_flags & _SKIP_FLAGS:
                self._next_call += 1
                self.frames[id(frame)] = self._next_call
                self._enable_calls(frame.f_code)
            frame = frame.f_back
        self.runtime.active = True
        events = monitoring.events
        try:
            monitoring.register_callback(tool, events.PY_START, self._on_start)
            monitoring.register_callback(tool, events.PY_RETURN, self._on_return)
            monitoring.register_callback(tool, events.PY_UNWIND, self._on_unwind)
            monitoring.register_callback(tool, events.CALL, self._on_call)
            monitoring.register_callback(tool, events.C_RETURN, self._on_c_exit)
            monitoring.register_callback(tool, events.C_RAISE, self._on_c_exit)
            monitoring.set_events(tool, events.PY_START | events.PY_RETURN | events.PY_UNWIND)
        except BaseException:
            self.stop()
            raise


    def stop(self) -> None:
        self.runtime.active = False
        if self._tool is not None:
            monitoring = sys.monitoring
            tool = self._tool
            monitoring.set_events(tool, 0)
            for code in self._local_codes.values():
                monitoring.set_local_events(tool, code, 0)
            self._local_codes.clear()
            for event in (monitoring.events.PY_START, monitoring.events.PY_RETURN, monitoring.events.PY_UNWIND,
                          monitoring.events.CALL, monitoring.events.C_RETURN, monitoring.events.C_RAISE):
                monitoring.register_callback(tool, event, None)
            monitoring.free_tool_id(tool)
            self._tool = None
        self.frames.clear()
        self._baseline.clear()
        self._native_calls.clear()
        self.runtime._external_depth = 0

