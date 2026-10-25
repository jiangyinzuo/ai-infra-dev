"""不依赖模型的对象流场景，同时覆盖历史载体与直接复制。"""


class Box:
    value: object


def consume(value=None):
    return value


def inspect_mapping(mapping, request_id):
    value = mapping[request_id]
    return consume(value=value)


def run():
    target = []
    alias = target
    mapping = {}
    mapping["request"] = alias
    inspect_mapping(mapping, "request")
    consume(mapping.get("request"))
    consume(mapping.pop("request"))
    consume(mapping.setdefault("request", target))
    items = []
    items.append(target)
    consume(items.pop())
    copied = target.copy()
    consume(copied)
    box = Box()
    box.value = target
    consume(box.value)
    inner = [target]
    outer = {"inner": inner}
    consume(outer)
    mapping.clear()
    consume(mapping)
    consume(*[target], **{})
    return target
