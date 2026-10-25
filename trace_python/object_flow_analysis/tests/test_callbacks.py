"""外部 C 函数中的项目回调不能重新打开已截断的路径。"""
from trace_python.object_flow_analysis.tests.test_flow import trace_source


def test_sorted_callback_is_not_followed(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def callback(value):
    return 0

def consume(value):
    return value

def run():
    target = object()
    sorted([target], key=callback)
    return consume(target) is target
''')
    assert result is True
    assert not any(event["site"]["function"] == "callback" for event in runtime.records)
    assert any(event["kind"] == "argument" and event["site"]["function"] == "consume"
               for event in runtime.records)
