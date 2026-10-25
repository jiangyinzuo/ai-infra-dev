"""运行时筛选必须让别名调用仍在原调用帧中执行。"""
from trace_python.object_flow_analysis.tests.test_flow import trace_source


def test_locals_eval_aliases_keep_original_frame(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def run():
    target = object()
    local_namespace = locals
    evaluate = eval
    return local_namespace()["target"] is target, evaluate("target") is target
''')
    assert result == (True, True)
