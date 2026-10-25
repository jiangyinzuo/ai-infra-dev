"""不能以用户自定义操作或外部返回推断对象来源。"""
from trace_python.object_flow_analysis.tests.test_flow import trace_source


def test_property_and_custom_container_are_not_holding_evidence(tmp_path):
    runtime, result = trace_source(tmp_path, '''
class Box:
    @property
    def value(self):
        return None
    @value.setter
    def value(self, value):
        self.accepted = True

class Sink:
    def __setitem__(self, key, value):
        pass

def run():
    target = object()
    box = Box()
    box.value = target
    Sink()["key"] = target
    return box.accepted
''')
    assert result is True
    assert not any(event["kind"] == "observed_write" for event in runtime.records)


def test_external_identity_can_be_observed_again_without_return_edge(tmp_path):
    # 标准库调用不建立返回边；随后参数仍然按实际身份识别。
    runtime, result = trace_source(tmp_path, '''
import operator

def consume(value):
    return value

def run():
    target = []
    value = operator.itemgetter(0)((target,))
    return consume(value) is target
''')
    assert result is True
    captured = next(event["object_id"] for event in runtime.records if event["kind"] == "capture")
    assert any(event["kind"] == "argument" and event["object_id"] == captured
               and event["site"]["function"] == "consume" for event in runtime.records)
    assert all("operator" not in event["site"]["file"] for event in runtime.records)


def test_same_object_multiple_captures_share_identity(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def run():
    shared = []
    for i in range(2):
        target = shared
    return shared
''')
    captures = [event for event in runtime.records if event["kind"] == "capture"]
    assert len(captures) == 2
    assert captures[0]["object_id"] == captures[1]["object_id"]
    assert len(runtime.summary()["captures"]) == 2


def test_cycle_and_carrier_copy_do_not_become_target_copies(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def run():
    target = object()
    first = [target]
    second = [first]
    first.append(second)
    copied = first.copy()
    return first[0] is target and second[0] is first and copied is not first
''')
    assert result is True
    assert not any(event["kind"] == "shallow_copy" for event in runtime.records)
