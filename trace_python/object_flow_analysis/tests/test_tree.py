"""调用参数显示真实调用位置，重复执行聚合但不同位置不能混合。"""
from trace_python.object_flow_analysis.tests.test_flow import trace_source
from trace_python.object_flow_analysis.tree import render_tree

SOURCE = '''
class Blocks:
    def get_block_ids(self, allow_none=False):
        return None

def run():
    target = Blocks()
    mapping = {"request": target}
    for i in range(3):
        mapping["request"].get_block_ids(allow_none=True)
    mapping["request"].get_block_ids(allow_none=False)
'''


def test_argument_leaf_shows_call_statement_not_definition(tmp_path):
    runtime, _ = trace_source(tmp_path, SOURCE)
    tree = render_tree(runtime.records, runtime.summary())
    leaves = [line for line in tree.splitlines() if "argument" in line]
    assert leaves
    assert all("def get_block_ids" not in line for line in leaves)
    call_line = next(index for index, source in enumerate(SOURCE.splitlines(), 1)
                     if 'get_block_ids(allow_none=True)' in source)
    assert any('mapping["request"].get_block_ids(allow_none=True)' in line
               and f":{call_line} " in line for line in leaves)


def test_same_source_line_aggregates_across_executions_and_event_kinds(tmp_path):
    runtime, _ = trace_source(tmp_path, SOURCE)
    tree = render_tree(runtime.records, runtime.summary())
    arguments = [line for line in tree.splitlines() if "argument" in line]
    assert len(arguments) == 2  # 三次循环调用和另一个调用位置，旧版产生四个叶子。
    repeated = [line for line in tree.splitlines()
                if 'mapping["request"].get_block_ids(allow_none=True)' in line
                and "argument" in line]
    assert len(repeated) == 1
    assert "argument=3" in repeated[0]
    other = [line for line in tree.splitlines()
             if 'mapping["request"].get_block_ids(allow_none=False)' in line
             and "argument" in line]
    assert len(other) == 1
    assert "argument=1" in other[0]
