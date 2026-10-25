"""异常区间和循环保持业务结果，复杂表达式不重算。"""
from trace_python.object_flow_analysis.tests.test_flow import trace_source


def test_loop_comprehension_and_nested_exception_regions(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def run():
    target = object()
    values = [target for _ in range(3)]
    mapping = {str(i): target for i in range(2)}
    members = {target for _ in range(2)}
    try:
        try:
            raise ValueError("expected")
        except ValueError:
            mapping["new"] = target
    finally:
        values.append(target)
    return len(values), len(mapping), len(members), all(value is target for value in values)
''')
    assert result == (4, 3, 1, True)
    assert any(event["kind"] == "observed_write" for event in runtime.records)


def test_chained_assignment_and_attribute_receiver_once(tmp_path):
    runtime, result = trace_source(tmp_path, '''
class Box:
    pass

def run():
    target = object()
    calls = []
    box = Box()
    def receiver():
        calls.append("receiver")
        return box
    alias = another = target
    receiver().value = alias
    return calls, box.value is another
''')
    assert result == (["receiver"], True)
