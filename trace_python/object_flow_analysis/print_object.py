"""捕获时的有界快照，不调用用户 repr、property 或自定义迭代协议。"""
from itertools import islice
from types import GetSetDescriptorType
from typing import Any, cast


def snapshot(value: object, *, depth: int = 2, members: int = 8,
             text_limit: int = 160, node_limit: int = 64) -> Any:
    remaining = node_limit
    active: set[int] = set()

    def visit(item: object, level: int) -> Any:
        nonlocal remaining
        cls = type(item)
        typename = type.__getattribute__(cls, '__qualname__')
        if remaining <= 0:
            return {"type": typename, "truncated": True}
        remaining -= 1
        if item is None or cls in (bool, float):
            # JSON 不接受非有限浮点；保留字符串而非产生非法 JSON。
            if cls is float and (item != item or item in (float('inf'), float('-inf'))):
                return str(item)
            return item
        if cls is int:
            item = cast(int, item)
            return item if item.bit_length() <= 256 else {"type": "int", "bits": item.bit_length()}
        if cls is str:
            item = cast(str, item)
            return item if len(item) <= text_limit else {"type": "str", "value": item[:text_limit], "truncated": True}
        if cls is bytes:
            item = cast(bytes, item)
            return {"type": "bytes", "hex": item[:text_limit // 2].hex(), "length": len(item)}
        if id(item) in active:
            return {"type": typename, "cycle": True}
        if level < 0:
            return {"type": typename, "truncated": True}
        active.add(id(item))
        try:
            if cls is dict:
                item = cast(dict, item)
                return {"type": typename, "length": len(item),
                        "items": [[visit(key, level - 1), visit(val, level - 1)]
                                  for key, val in islice(item.items(), members)],
                        "truncated": len(item) > members}
            if cls in (list, tuple, set, frozenset):
                item = cast(list | tuple | set | frozenset, item)
                return {"type": typename, "length": len(item),
                        "items": [visit(child, level - 1) for child in islice(item, members)],
                        "truncated": len(item) > members}
            # 只读取 CPython 普通实例字典 descriptor；不经过对象的 __getattribute__。
            descriptor = next((type.__getattribute__(base, '__dict__')['__dict__']
                               for base in type.__getattribute__(cls, '__mro__')
                               if '__dict__' in type.__getattribute__(base, '__dict__')), None)
            if type(descriptor) is not GetSetDescriptorType:
                return {"type": typename}
            namespace = descriptor.__get__(item, cls)
            if type(namespace) is not dict:
                return {"type": typename}
            fields = {key: visit(val, level - 1) for key, val in islice(namespace.items(), members)
                      if type(key) is str}
            return {"type": typename, "fields": fields, "truncated": len(namespace) > members}
        finally:
            active.remove(id(item))

    return visit(value, depth)
