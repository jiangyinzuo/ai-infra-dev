"""有界对象身份与历史证据；协调探针、操作适配和调用观察。"""
from __future__ import annotations

import inspect
import sys
import threading
from pathlib import Path
from types import CodeType
from typing import Any

from .monitoring import CallMonitor
from .operations import OperationObserver
from .protocol import Site

_SKIP_FLAGS = inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR


class Runtime:
    def __init__(self, root: Path, max_events: int = 10000, max_objects: int = 10000,
                 entry_code: CodeType | None = None):
        self.root = root.resolve()
        self.max_events = max_events
        self.max_objects = max_objects
        self.records: list[dict[str, Any]] = []
        self.captures: list[dict[str, int]] = []
        self.active = False
        self.incomplete = False
        self.reasons: list[str] = []
        self._objects: dict[int, dict[str, Any]] = {}
        self._refs: dict[int, object] = {}
        self._edges: set[tuple[int, int, str]] = set()
        self._disabled = False
        self._external_depth = 0
        self.probes = 0
        self.unknown_operations = 0
        self.monitor = CallMonitor(self, entry_code)
        self.operations = OperationObserver(self)

    def dispatch(self, kind: str, site: Site, operands: tuple[Any, ...]) -> object:
        self.probes += 1
        observe = self._eligible() and (kind == 'capture' or bool(self._objects))
        # 必须在直接探针入口取业务帧，不能移进额外包装层。
        if observe:
            frame = sys._getframe(1)
            if frame.f_code.co_flags & _SKIP_FLAGS or id(frame) not in self.monitor.frames:
                observe = False
        if kind in ('capture', 'bind'):
            value = operands[0]
            if observe:
                if kind == 'capture':
                    self._capture(value, site)
                else:
                    self._emit(kind, value, site)
            return value
        if kind in ('call', 'call_ex'):
            return self.operations.call(kind, site, operands, observe)
        return self.operations.execute(kind, site, operands, observe)

    def _capture(self, value: object, site: Site) -> None:
        node = self._register(value, 'target')
        if node is None:
            return
        capture_id = len(self.captures) + 1
        node['capture_ids'].append(capture_id)
        self.captures.append({'capture_id': capture_id, 'object_id': node['object_id']})
        self._emit('capture', value, site, capture_id=capture_id)

    def wants_call(self, function: object) -> bool:
        return self.operations.wants_call(function)

    def start(self) -> Runtime:
        # baseline 必须来自原调用者；monitor.start 只负责注册与资源管理。
        self.monitor.start(sys._getframe(1))
        return self

    def stop(self) -> None:
        self.monitor.stop()

    def _limit(self, reason: str) -> None:
        self.incomplete = True
        self._disabled = True
        if reason not in self.reasons:
            self.reasons.append(reason)


    def _eligible(self) -> bool:
        return (self.active and not self._disabled and not self._external_depth and
                threading.get_ident() == self.monitor.thread)


    def _register(self, value: object, role: str, captures: list[int] | None = None) -> dict[str, Any] | None:
        key = id(value)
        found = self._objects.get(key)
        if found is not None:
            if role == 'target' or (role == 'historical_carrier' and found['role'] == 'carrier_relation'):
                found['role'] = role
            return found
        if len(self._objects) >= self.max_objects:
            self._limit('max_objects')
            return None
        cls = type(value)
        found = {'object_id': len(self._objects) + 1, 'type': type.__getattribute__(cls, '__qualname__'),
                 'module': type.__getattribute__(cls, '__module__'), 'role': role, 'capture_ids': list(captures or ())}
        self._refs[key] = value  # 强引用阻止 id 重用；不调用用户 repr 或相等比较。
        self._objects[key] = found
        return found


    def _stack(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        frame = sys._getframe(1)
        while frame is not None:
            call_id = self.monitor.frames.get(id(frame))
            if call_id is not None and self.monitor.project(frame.f_code):
                result.append({'file': frame.f_code.co_filename,
                               'function': frame.f_code.co_qualname,
                               'line': frame.f_lineno, 'call_id': call_id})
            frame = frame.f_back
        result.reverse()
        return result


    @staticmethod
    def _site(site: Site) -> dict[str, Any]:
        return dict(zip(('file', 'function', 'line', 'end_line', 'column', 'end_column', 'detail'), site))


    def _emit(self, kind: str, value: object, site: Site, **related: Any) -> None:
        node = self._objects.get(id(value))
        if node is None or not self._eligible():
            return
        if len(self.records) >= self.max_events:
            self._limit('max_events')
            return
        self.records.append({'sequence': len(self.records) + 1, 'kind': kind,
                             'object_id': node['object_id'], 'role': node['role'],
                             'site': self._site(site), 'stack': self._stack(), **related})
        if node['role'] in ('historical_carrier', 'carrier_relation') and kind in ('argument', 'return', 'bind', 'read_subscript', 'read_attribute', 'builtin_result'):
            if len(self.records) >= self.max_events:
                self._limit('max_events')
                return
            self.records.append({'sequence': len(self.records) + 1, 'kind': 'historical_carrier_use',
                                 'object_id': node['object_id'], 'role': node['role'],
                                 'site': self._site(site), 'stack': self.records[-1]['stack'],
                                 'operation': kind, **related})


    def _hold(self, container: object, value: object, site: Site, kind: str) -> None:
        source = self._objects.get(id(value))
        if source is None or not self._eligible():
            return
        role = 'historical_carrier' if source['role'] in ('target', 'copy') else 'carrier_relation'
        dest = self._register(container, role)
        if dest is None:
            return
        edge = (dest['object_id'], source['object_id'], kind)
        self._edges.add(edge)
        self._emit('observed_write', value, site, carrier_object_id=dest['object_id'], relation=role, operation=kind)


    def _copy(self, source: object, result: object, site: Site) -> None:
        node = self._objects.get(id(source))
        if node is None:
            return
        role = 'copy' if node['role'] in ('target', 'copy') else 'carrier_relation'
        dest = self._register(result, role, node['capture_ids'] if role == 'copy' else None)
        if dest is not None:
            kind = 'shallow_copy' if node['role'] in ('target', 'copy') else 'carrier_copy'
            self._emit(kind, result, site, source_object_id=node['object_id'], same_object=result is source)


    def _related(self, value: object) -> dict[str, int]:
        node = self._objects.get(id(value))
        return {'container_object_id': node['object_id']} if node is not None else {}


    def summary(self) -> dict[str, Any]:
        return {'events': len(self.records), 'objects': [dict(node, capture_ids=list(node['capture_ids'])) for node in self._objects.values()],
                'captures': list(self.captures), 'incomplete': self.incomplete,
                'reasons': list(self.reasons), 'probes': self.probes,
                'unknown_operations': self.unknown_operations, 'relations': len(self._edges)}


    def __enter__(self) -> Runtime:
        return self.start()


    def __exit__(self, *exception: object) -> None:
        self.stop()

