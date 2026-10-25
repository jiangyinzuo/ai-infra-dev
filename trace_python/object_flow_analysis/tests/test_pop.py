"""pop 观察实际结果，保留索引、异常与精确类型边界。"""
from trace_python.object_flow_analysis.tests.test_flow import trace_source


def test_list_and_set_pop_observe_discarded_results_and_aliases(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def run():
    target = object()
    items = [target, None]
    pop = items.pop
    pop(*(0,), **{})
    members = {target}
    members.pop()
    return items, len(members)
''')
    assert result == ([None], 0)
    reads = [event for event in runtime.records if event["kind"] == "builtin_result"]
    assert len(reads) == 2
    assert all(event["operation"] == "pop" for event in reads)
    assert len({event["object_id"] for event in reads}) == 1
    assert all("container_object_id" in event for event in reads)


def test_pop_failure_and_unrelated_value_do_not_emit_target_result(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def run():
    target = object()
    items = [target, None]
    first = items.pop()
    errors = []
    try:
        items.pop(5)
    except IndexError:
        errors.append("index")
    try:
        set().pop()
    except KeyError:
        errors.append("empty")
    return first, errors, items[0] is target
''')
    assert result == (None, ["index", "empty"], True)
    assert not any(event["kind"] == "builtin_result" for event in runtime.records)


def test_custom_pop_is_not_classified_as_builtin_result(tmp_path):
    runtime, result = trace_source(tmp_path, '''
class Custom(list):
    def pop(self, index=-1):
        return self[0]

def run():
    target = object()
    items = Custom([target])
    return items.pop() is target, len(items)
''')
    assert result == (True, 1)
    assert not any(event["kind"] == "builtin_result" for event in runtime.records)


def test_popped_carrier_keeps_historical_role(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def run():
    target = object()
    inner = [target]
    items = [inner]
    value = items.pop()
    return value is inner
''')
    assert result is True
    event = next(event for event in runtime.records if event["kind"] == "builtin_result")
    assert event["role"] == "historical_carrier"
    assert any(item["kind"] == "historical_carrier_use"
               and item["object_id"] == event["object_id"]
               and item["site"] == event["site"] for item in runtime.records)
