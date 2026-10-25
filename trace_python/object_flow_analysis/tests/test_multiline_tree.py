"""多行源码保留换行、注释、原引号和相对缩进。"""
from trace_python.object_flow_analysis.tests.test_flow import trace_source
from trace_python.object_flow_analysis.tree import render_tree


def test_multiline_call_preserves_source_and_tree_continuation(tmp_path):
    runtime, _ = trace_source(tmp_path, '''
def consume(value, label):
    return value

def run():
    target = object()
    result = consume(
        value=target,  # 保留参数说明
        label="two  spaces",
    )
    return result
''')
    lines = render_tree(runtime.records, runtime.summary()).splitlines()
    indices = [index for index, line in enumerate(lines) if "argument=" in line]
    assert len(indices) == 1
    index = indices[0]
    header = lines[index]
    assert "result = consume(" in header
    assert header.index(":7 ") < header.index("argument=") < header.index("objects=")
    tree_prefix = header[:header.index("├─") if "├─" in header else header.index("└─")]
    continuation = tree_prefix + ("│ " if "├─" in header else "  ")
    assert lines[index + 1] == continuation + "    value=target,  # 保留参数说明"
    assert lines[index + 2] == continuation + '    label="two  spaces",'
    assert lines[index + 3] == continuation + ")"
