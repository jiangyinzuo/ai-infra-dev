"""字典方法的实际返回值与新增持有必须分别证明。"""
from trace_python.object_flow_analysis.tests.test_flow import trace_source


def test_get_pop_setdefault_results_include_alias_and_unpacking(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def run():
    target = object()
    mapping = {"get": target, "pop": target, "existing": target}
    getter = mapping.get
    first = getter(*("get",), **{})
    second = mapping.pop("pop")
    third = mapping.setdefault("existing", object())
    return first is second is third is target, "pop" not in mapping
''')
    assert result == (True, True)
    reads = [event for event in runtime.records if event["kind"] == "builtin_result"]
    assert [event["operation"] for event in reads] == ["get", "pop", "setdefault"]
    assert len({event["object_id"] for event in reads}) == 1


def test_defaults_do_not_claim_dictionary_holding(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def run():
    target = object()
    mapping = {}
    first = mapping.get("absent", target)
    second = mapping.pop("absent", target)
    return first is second is target, mapping
''')
    assert result == (True, {})
    reads = [event for event in runtime.records if event["kind"] == "builtin_result"]
    assert [event["operation"] for event in reads] == ["get", "pop"]
    assert all("container_object_id" not in event for event in reads)
    assert not any(event["kind"] == "observed_write" for event in runtime.records)


def test_setdefault_new_key_but_not_existing_key_records_insertion(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def run():
    target = object()
    mapping = {"existing": None}
    first = mapping.setdefault("new", target)
    second = mapping.setdefault("existing", target)
    return first is target, second is None, mapping["new"] is target
''')
    assert result == (True, True, True)
    writes = [event for event in runtime.records if event["kind"] == "observed_write"]
    assert len(writes) == 1
    assert writes[0]["operation"] == "setdefault"
    assert sum(event["kind"] == "builtin_result" for event in runtime.records) == 1


def test_dict_errors_do_not_record_success_or_repeat_hash(tmp_path):
    runtime, result = trace_source(tmp_path, '''
class Key:
    calls = 0
    def __hash__(self):
        type(self).calls += 1
        return 42

def run():
    target = object()
    mapping = {}
    key = Key()
    mapping.setdefault(key, target)
    for operation in (mapping.get, mapping.pop, mapping.setdefault):
        try:
            operation([], target)
        except TypeError:
            pass
    try:
        mapping.pop("missing")
    except KeyError:
        pass
    return Key.calls
''')
    assert result == 1
    assert sum(event["kind"] == "builtin_result" for event in runtime.records) == 1
    assert sum(event["kind"] == "observed_write" for event in runtime.records) == 1


def test_dict_subclass_not_promoted_and_same_named_method_not_intercepted(tmp_path):
    runtime, result = trace_source(tmp_path, '''
class Custom(dict):
    def get(self, key, default=None):
        return default

def run():
    target = object()
    mapping = Custom()
    return mapping.get("key", target) is target
''')
    assert result is True
    assert not any(event["kind"] == "builtin_result" for event in runtime.records)
