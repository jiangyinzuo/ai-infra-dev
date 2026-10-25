"""YAML 源码块可原样读取，结构无需解析长标签。"""
import yaml

from trace_python.object_flow_analysis.tests.test_flow import trace_source
from trace_python.object_flow_analysis.yaml_tree import yaml_tree


def test_yaml_keeps_multiline_source_and_structured_evidence(tmp_path):
    runtime, _ = trace_source(tmp_path, '''
def consume(value):
    return value

def run():
    target = object()
    for _ in range(2):
        consume(
            target,  # 保留说明
        )
''')
    text = yaml_tree(runtime.records, runtime.summary())
    document = yaml.safe_load(text)
    run = next(node for node in document['tree'] if node['function'] == 'run')
    consume = next(node for node in run['children'] if node.get('function') == 'consume')
    argument = next(node for node in consume['children'] if 'argument' in node.get('events', {}))
    assert argument['source'] == 'consume(\n    target,  # 保留说明\n)'
    assert argument['events'] == {'argument': 2}
    assert argument['objects'] == ['O1']
    assert argument['line'] == 8
    assert 'source: |-' in text
    assert '&id' not in text and '*id' not in text
    assert document['summary']['captures'] == runtime.summary()['captures']
