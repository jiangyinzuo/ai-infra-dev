"""执行原操作一次；成功后解释精确内置容器和普通字段的证据。"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import FunctionType, GetSetDescriptorType
from typing import Any, TYPE_CHECKING, cast

from .protocol import Site

if TYPE_CHECKING:
    from .runtime import Runtime

_CONTAINERS = (list, dict, set, tuple, frozenset)
_MISSING = object()
_METHODS: dict[type, frozenset[str]] = {
    list: frozenset(('copy', 'append', 'insert', 'extend', 'pop')),
    dict: frozenset(('copy', 'update', 'get', 'pop', 'setdefault')),
    set: frozenset(('copy', 'add', 'update', 'pop')),
}


class OperationObserver:
    def __init__(self, runtime: Runtime):
        self.runtime = runtime

    def execute(self, kind: str, site: Site, operands: tuple[Any, ...], observe: bool) -> object:
        if kind in ('list_append', 'set_add', 'map_add'):
            if observe:
                if kind == 'list_append':
                    container, value = operands
                    if type(container) is list:
                        self.runtime._hold(container, value, site, kind)
                elif kind == 'set_add':
                    container, value = operands
                    if type(container) is set and any(actual is value for actual in container):
                        self.runtime._hold(container, value, site, kind)
                else:
                    container, key, value = operands
                    if type(container) is dict:
                        self.runtime._hold(container, value, site, kind)
                        if any(actual is key for actual in container):
                            self.runtime._hold(container, key, site, kind)
            return None
        if kind == 'subscript':
            container, key = operands
            result = container[key]
            if observe and type(container) in _CONTAINERS:
                if type(key) is slice and key.start is None and key.stop is None and key.step is None and type(container) in (list, tuple):
                    self.runtime._copy(container, result, site)
                self.runtime._emit('read_subscript', result, site, **self.runtime._related(container))
            return result
        if kind == 'store_subscript':
            value, container, key = operands
            container[key] = value
            if observe and type(container) in (list, dict):
                if type(container) is list:
                    if type(key) is not slice:
                        self.runtime._hold(container, value, site, kind)
                    elif type(value) in _CONTAINERS:
                        # 切片赋值写入成员而非 iterable 自身，只匹配实际保留的身份。
                        candidates = {id(member) for member in (value.keys() if type(value) is dict else value)}
                        for member in container:
                            if id(member) in candidates:
                                self.runtime._hold(container, member, site, kind)
                elif type(container) is dict:
                    # __hash__/__eq__ 可能重入修改容器；成功返回不保证最终仍保存 value。
                    if id(value) in self.runtime._objects and any(actual is value for actual in container.values()):
                        self.runtime._hold(container, value, site, kind)
                    if id(key) in self.runtime._objects and any(actual is key for actual in container):
                        self.runtime._hold(container, key, site, kind)
            return None
        if kind == 'attribute':
            receiver = operands[0]
            result = getattr(receiver, site[6])
            if observe and id(result) in self.runtime._objects and self._ordinary_attribute(receiver, site[6], result):
                self.runtime._emit('read_attribute', result, site, **self.runtime._related(receiver))
            return result
        if kind == 'store_attribute':
            value, receiver = operands
            setattr(receiver, site[6], value)
            if observe and id(value) in self.runtime._objects and self._ordinary_attribute(receiver, site[6], value):
                self.runtime._hold(receiver, value, site, kind)
            return None
        if kind == 'build_list':
            result = list(operands)
        elif kind == 'build_tuple':
            result = tuple(operands)
        elif kind == 'build_set':
            result = set(operands)
        elif kind == 'build_map':
            result = dict(zip(operands[::2], operands[1::2]))
        elif kind == 'build_const_map':
            result = dict(zip(operands[-1], operands[:-1]))
        else:
            self.runtime.unknown_operations += 1
            raise ValueError(f'Unknown object-flow operation: {kind}')
        if observe:
            # 只看构造后直接成员，字典重复键和集合去重不能虚报被丢弃的对象。
            for value in self._members(result):
                self.runtime._hold(result, value, site, kind)
        return result


    @staticmethod
    def _members(value: object) -> tuple[object, ...]:
        if type(value) is dict:
            return tuple(part for pair in value.items() for part in pair)
        if type(value) in _CONTAINERS:
            return tuple(cast(list | set | tuple | frozenset, value))
        return ()


    def _ordinary_attribute(self, receiver: object, name: str, value: object) -> bool:
        cls = type(receiver)
        if inspect.getattr_static(cls, '__setattr__') is not object.__setattr__ or inspect.getattr_static(cls, '__getattribute__') is not object.__getattribute__:
            return False
        module = sys.modules.get(type.__getattribute__(cls, '__module__'))
        filename = vars(module).get('__file__') if module is not None else None
        project_class = bool(filename and Path(filename).resolve().is_relative_to(self.runtime.root))
        if not project_class:
            # exec 创建的类未必注册模块；其自身方法的代码仍可证明项目来源。
            prefix = type.__getattribute__(cls, '__qualname__') + '.'
            for member in type.__getattribute__(cls, '__dict__').values():
                if type(member) in (staticmethod, classmethod):
                    member = member.__func__
                if type(member) is FunctionType and member.__qualname__.startswith(prefix) and self.runtime.monitor.project(member.__code__):
                    project_class = True
                    break
        if not project_class:
            return False
        for base in type.__getattribute__(cls, '__mro__'):
            namespace = type.__getattribute__(base, '__dict__')
            if '__slots__' in namespace:
                return False
            dict_descriptor = namespace.get('__dict__', _MISSING)
            if dict_descriptor is not _MISSING and type(dict_descriptor) is not GetSetDescriptorType:
                return False
            attribute = namespace.get(name, _MISSING)
            if attribute is not _MISSING and (inspect.getattr_static(type(attribute), '__get__', _MISSING) is not _MISSING or inspect.getattr_static(type(attribute), '__set__', _MISSING) is not _MISSING):
                return False
        try:
            namespace = object.__getattribute__(receiver, '__dict__')
        except AttributeError:
            return False
        return type(namespace) is dict and namespace.get(name, _MISSING) is value


    def wants_call(self, function: object) -> bool:
        """仅代理已知容器操作，普通调用保留原 CALL 和调用帧。"""
        if not self.runtime._eligible() or not self.runtime._objects:
            return False
        if any(function is cls for cls in _CONTAINERS):
            return True
        if not inspect.isbuiltin(function):
            return False
        receiver = function.__self__
        cls = type(receiver)
        name = function.__name__
        return name in _METHODS.get(cls, ())


    def call(self, kind: str, site: Site, operands: tuple[Any, ...], observe: bool) -> object:
        function = operands[0]
        if kind == 'call':
            args, names = operands[1:]
            split = len(args) - len(names)
            positional, kwargs = args[:split], dict(zip(names, args[split:]))
        else:
            positional, kwargs = operands[1:]
            if kwargs is None:
                kwargs = {}
        receiver = function.__self__ if inspect.isbuiltin(function) else None
        # 不做第二次 key 查找，避免额外触发 __hash__/__eq__。仅对可能新增
        # 追踪关系的 setdefault 保存旧 key 身份；旧 key 的值不会被该操作覆盖。
        before_keys = None
        if (observe and type(receiver) is dict and function.__name__ == 'setdefault'
                and len(positional) in (1, 2) and not kwargs
                and any(id(value) in self.runtime._objects for value in positional)):
            before_keys = tuple(receiver)
        # 外部函数中的项目回调也被隔离；已知容器操作仅在返回后解释其直接效果。
        self.runtime._external_depth += 1
        try:
            result = function(*positional, **kwargs)
        finally:
            self.runtime._external_depth -= 1
        if before_keys is not None and self.runtime._eligible():
            old_ids = {id(key) for key in before_keys}
            default = positional[1] if len(positional) == 2 else None
            for key, value in cast(dict, receiver).items():
                if key is positional[0] and id(key) not in old_ids:
                    if value is default:
                        self.runtime._hold(receiver, value, site, 'setdefault')
                    self.runtime._hold(receiver, key, site, 'setdefault')
                    break
        if observe:
            self._builtin_call(function, positional, kwargs, result, site)
        return result


    def _builtin_call(self, function: object, args: tuple[Any, ...], kwargs: dict[str, Any], result: object, site: Site) -> None:
        if any(function is cls for cls in _CONTAINERS):
            if len(args) == 1 and not kwargs and type(args[0]) in _CONTAINERS:
                source = args[0]
                if function is not dict or type(source) is dict:
                    self.runtime._copy(source, result, site)
                # 复制历史载体仅建立复制关系，不沿历史关系推断成员。
                if id(source) not in self.runtime._objects or (function is dict and type(source) is not dict):
                    for value in self._members(result):
                        self.runtime._hold(result, value, site, 'construct')
            elif not args and function is dict:
                for value in kwargs.values():
                    self.runtime._hold(result, value, site, 'construct')
            return
        # 不接受用户伪造同名方法；精确 builtin 方法才有已知副作用语义。
        if not inspect.isbuiltin(function):
            return
        receiver = getattr(function, '__self__', None)
        cls = type(receiver)
        if cls not in (list, dict, set):
            return
        name = getattr(function, '__name__', '')
        if (cls is dict and name in ('get', 'pop', 'setdefault')) or (cls in (list, set) and name == 'pop'):
            # 默认值也可能就是目标，故只称“方法结果”，不伪造字典读取来源。
            self.runtime._emit('builtin_result', result, site, operation=name, **self.runtime._related(receiver))
        elif name == 'copy' and not args and not kwargs:
            self.runtime._copy(receiver, result, site)
        elif cls is list and name in ('append', 'insert') and not kwargs:
            if len(args) == (1 if name == 'append' else 2):
                self.runtime._hold(receiver, args[-1], site, name)
        elif cls is set and name == 'add' and len(args) == 1 and not kwargs:
            # add(equal_but_not_identical) 不意味着新对象实际进入集合。
            if any(value is args[0] for value in cast(set, receiver)):
                self.runtime._hold(receiver, args[0], site, name)
        elif cls is list and name == 'extend' and len(args) == 1 and not kwargs:
            if type(args[0]) in _CONTAINERS:
                source = args[0]
                values = source.keys() if type(source) is dict else source
                for value in values:
                    self.runtime._hold(receiver, value, site, name)
        elif cls is set and name == 'update' and not kwargs:
            if all(type(value) in _CONTAINERS for value in args):
                candidates = {id(value) for source in args for value in (source.keys() if type(source) is dict else source)}
                for value in cast(set, receiver):
                    if id(value) in candidates:
                        self.runtime._hold(receiver, value, site, name)
        elif cls is dict and name == 'update':
            if not args or (len(args) == 1 and type(args[0]) is dict):
                candidates = {id(value) for value in kwargs.values()}
                if args:
                    candidates.update(id(value) for value in self._members(args[0]))
                for value in self._members(receiver):
                    if id(value) in candidates:
                        self.runtime._hold(receiver, value, site, name)

