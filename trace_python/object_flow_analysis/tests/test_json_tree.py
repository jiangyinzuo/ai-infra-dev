"""JSONL 树保留可展开调用层次、源码聚合与运行摘要。"""
import json
from trace_python.object_flow_analysis.tests.test_flow import trace_source
from trace_python.object_flow_analysis.tree import json_tree


def test_nested_tree_preserves_calls_and_aggregated_multiline_source(tmp_path):
    runtime, _ = trace_source(tmp_path, '''
def consume(value):
    return value

def run():
    target = object()
    for _ in range(2):
        consume(
            target,  # 保留源码
        )
''')
    value = json.loads(json.dumps(json_tree(runtime.records, runtime.summary())))
    tree = value["tree"]
    run_children = next(children for label, children in tree.items() if label.startswith("run ("))
    consume_children = next(children for label, children in run_children.items()
                            if label.startswith("consume ("))
    argument_labels = [label for label in consume_children if "argument=" in label]
    assert len(argument_labels) == 1
    assert "argument=2" in argument_labels[0]
    assert "\n    target,  # 保留源码\n)" in argument_labels[0]
    assert consume_children[argument_labels[0]] == {}
    assert value["summary"]["captures"] == runtime.summary()["captures"]


def test_empty_tree_retains_resource_summary():
    summary = {"events": 0, "incomplete": True, "reasons": ["max_objects"]}
    value = json_tree([], summary)
    assert value["tree"] == {}
    assert value["summary"] == summary
