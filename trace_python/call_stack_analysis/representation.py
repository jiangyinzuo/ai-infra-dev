"""将结构化快照格式化为面向人类的 Python 风格文本。"""

from typing import Any


def format_value(value: Any) -> str:
    """只处理已采集快照，不访问业务对象或 GPU 数据。"""
    if not isinstance(value, dict):
        return repr(value)
    if "unavailable" in value:
        return f"<unavailable: {value['unavailable']}>"
    if "shape" in value:
        return (
            f"Tensor(shape={tuple(value['shape'])!r}, "
            f"dtype={value['dtype']!r}, device={value['device']!r})"
        )
    kind = value["type"]
    if "length" not in value:
        return f"<{kind}>"
    items = value.get("items", [])
    truncated = value.get("truncated", False) or len(items) < value["length"]
    if kind == "dict":
        parts = [f"{format_value(key)}: {format_value(item)}" for key, item in items]
    else:
        parts = [format_value(item) for item in items]
    if truncated:
        parts.append("...")
    body = ", ".join(parts)
    match kind:
        case "list":
            text = f"[{body}]"
        case "tuple":
            comma = "," if len(parts) == 1 else ""
            text = f"({body}{comma})"
        case "dict":
            text = "{" + body + "}"
        case "set":
            text = "{" + body + "}" if parts else "set()"
        case "frozenset":
            text = "frozenset({" + body + "})" if parts else "frozenset()"
        case _:
            return f"<{kind} length={value['length']}>"
    return f"{text} (length={value['length']})" if truncated else text
