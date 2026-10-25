"""对象内容来自捕获时，不执行用户展示或属性逻辑。"""
import json
from trace_python.object_flow_analysis.print_object import snapshot
from trace_python.object_flow_analysis.tests.test_flow import trace_source
from trace_python.object_flow_analysis.tree import render_tree, json_tree


def test_print_object_does_not_call_user_hooks():
    class Value:
        def __getattribute__(self, name):
            raise AssertionError("attribute hook")
        def __repr__(self):
            raise AssertionError("repr hook")
    value = Value()
    object.__setattr__(value, "answer", 42)
    result = snapshot(value)
    assert result["fields"] == {"answer": 42}


def test_cycle_large_values_and_limits_are_json_safe():
    value = []
    value.append(value)
    result = snapshot(value)
    assert result["items"][0]["cycle"] is True
    data = snapshot({"long": "x" * 1000, "integer": 10 ** 10000, "nan": float('nan')})
    encoded = json.dumps(data, allow_nan=False)
    assert len(encoded) < 1000
    assert snapshot(list(range(100)))["truncated"] is True


def test_capture_content_is_not_final_mutated_object(tmp_path):
    runtime, _ = trace_source(tmp_path, '''
def run():
    target = [42]
    target.append(99)
''')
    capture = next(event for event in runtime.records if event['kind'] == 'capture')
    assert capture['captured_value']['items'] == [42]
    summary = runtime.summary()
    assert summary['objects'][0]['captured_value']['items'] == [42]
    assert '对象内容' in render_tree(runtime.records, summary)
    assert json_tree(runtime.records, summary)['summary']['objects'][0]['captured_value']['items'] == [42]
