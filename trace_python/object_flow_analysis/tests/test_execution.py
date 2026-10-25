"""插桩必须保留调用展开、异常和表达式的求值行为。"""
from trace_python.object_flow_analysis.tests.test_flow import trace_source


def test_keyword_call_and_bound_self(tmp_path):
    runtime, result = trace_source(tmp_path, '''
class Target:
    def consume(self, *, value):
        return self is value

def run():
    target = Target()
    return target.consume(value=target)
''')
    assert result is True
    arguments = [event for event in runtime.records
                 if event["kind"] == "argument" and event["site"]["function"] == "Target.consume"]
    assert len(arguments) >= 2
    assert len({event["object_id"] for event in arguments}) == 1


def test_failed_call_does_not_run_twice(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def run():
    target = object()
    log = []
    def fail(value):
        log.append(value)
        raise ValueError("expected")
    try:
        fail(target)
    except ValueError as error:
        return len(log), str(error), log[0] is target
''')
    assert result == (1, "expected", True)


def test_return_from_finally_and_exception_cleanup(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def consume(value):
    try:
        raise KeyError("expected")
    finally:
        return value

def run():
    target = object()
    return consume(target) is target
''')
    assert result is True
    assert any(event["kind"] == "return" and event["site"]["function"] == "consume"
               for event in runtime.records)


def test_full_slice_not_partial_copy(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def run():
    target = [1, 2]
    partial = target[:1]
    complete = target[:]
    return partial, complete
''')
    assert result == ([1], [1, 2])
    assert sum(event["kind"] == "shallow_copy" for event in runtime.records) == 1
