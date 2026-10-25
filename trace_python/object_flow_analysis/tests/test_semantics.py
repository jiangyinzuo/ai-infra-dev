"""执行字节码转换不能改变用户可观察的求值次序和错误。"""
from trace_python.object_flow_analysis.tests.test_flow import trace_source


def test_call_expansion_preserves_argument_order(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def run():
    target = object()
    order = []
    def args():
        order.append("args")
        return [target]
    def kwargs():
        order.append("kwargs")
        return {"flag": True}
    def consume(value, *, flag):
        return value is target and flag
    answer = consume(*args(), **kwargs())
    return order, answer
''')
    assert result == (["args", "kwargs"], True)


def test_duplicate_keyword_error_preserved(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def run():
    target = object()
    calls = []
    def consume(**kwargs):
        calls.append(kwargs)
    try:
        consume(value=target, **{"value": target})
    except TypeError:
        return calls
''')
    assert result == []


def test_equal_nonidentical_set_member_not_claimed(tmp_path):
    runtime, result = trace_source(tmp_path, '''
class Equal:
    def __hash__(self):
        return 1
    def __eq__(self, other):
        return True

def run():
    target = Equal()
    existing = Equal()
    values = {existing}
    values.add(target)
    return len(values), next(iter(values)) is existing
''')
    assert result == (1, True)
    assert not any(event["kind"] == "observed_write" for event in runtime.records)
