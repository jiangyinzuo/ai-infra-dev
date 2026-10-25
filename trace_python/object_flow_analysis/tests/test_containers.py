"""支持的精确内置容器复制、构建与方法调用。"""
from trace_python.object_flow_analysis.tests.test_flow import trace_source


def test_supported_shallow_copy_operations(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def consume(value):
    return value

def run():
    target = [1, 2]
    copies = [target.copy(), list(target), tuple(target), set(target), frozenset(target), target[:]]
    for value in copies:
        consume(value)
    return [type(value).__name__ for value in copies]
''')
    assert result == ["list", "list", "tuple", "set", "frozenset", "list"]
    copies = [event for event in runtime.records if event["kind"] == "shallow_copy"]
    assert len(copies) == 6
    assert len({event["object_id"] for event in copies}) == 6


def test_empty_kwargs_method_alias_and_dict_copy(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def consume(value):
    return value

def run():
    target = {"k": 1}
    copied = target.copy(**{})
    items = []
    append = items.append
    append(target)
    consume(items)
    return copied == target and copied is not target and items[0] is target
''')
    assert result is True
    assert any(event["kind"] == "shallow_copy" for event in runtime.records)
    assert any(event["kind"] == "observed_write" for event in runtime.records)


def test_literal_containers_and_ordinary_field(tmp_path):
    runtime, result = trace_source(tmp_path, '''
class Box:
    pass

def run():
    target = object()
    items = [target]
    mapping = {"x": target}
    pair = (target,)
    members = {target}
    box = Box()
    box.value = target
    return items[0] is mapping["x"] is pair[0] is box.value and target in members
''')
    assert result is True
    assert sum(event["kind"] == "observed_write" for event in runtime.records) >= 5


def test_subclass_copy_not_followed(tmp_path):
    runtime, result = trace_source(tmp_path, '''
class Custom(list):
    def copy(self):
        return ["custom"]

def run():
    target = Custom([1])
    copied = target.copy()
    return copied
''')
    assert result == ["custom"]
    assert not any(event["kind"] == "shallow_copy" for event in runtime.records)
