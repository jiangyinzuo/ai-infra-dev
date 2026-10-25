"""依赖调用帧的内置函数不能被通用调用代理改变语义。"""
from trace_python.object_flow_analysis.tests.test_flow import trace_source


def test_zero_argument_super_and_local_namespace(tmp_path):
    runtime, result = trace_source(tmp_path, '''
class Parent:
    def get(self):
        return self

class Child(Parent):
    def get(self):
        return super().get()

def run():
    target = Child()
    alias = target
    namespace = locals()
    return target.get() is target, namespace["alias"] is target
''')
    assert result == (True, True)


def test_eval_uses_original_namespace(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def run():
    target = object()
    return eval("target") is target, globals()["__name__"]
''')
    assert result == (True, "flow_subject")
