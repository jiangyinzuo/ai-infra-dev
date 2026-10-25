"""不同版本的 CALL_FUNCTION_EX 空 kwargs 语义。"""
from trace_python.object_flow_analysis.tests.test_flow import trace_source


def test_unpack_constructor_with_and_without_keywords(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def run():
    target = [1, 2]
    first = list(*(target,))
    second = tuple(*(target,), **{})
    return first, second
''')
    assert result == ([1, 2], (1, 2))
    assert sum(event["kind"] == "shallow_copy" for event in runtime.records) == 2
