"""只显示标记的结果任务；失败和不可达沿用 Ansible 默认输出。"""

from ansible.plugins.callback.default import CallbackModule as DefaultCallback

DOCUMENTATION = """
name: npu_top
type: stdout
short_description: Display NPU summaries and errors only
description:
  - Successful tasks are displayed only when tagged npu_top_output.
extends_documentation_fragment:
  - default_callback
  - result_format_callback
"""


class CallbackModule(DefaultCallback):
    CALLBACK_NAME = "npu_top"

    def v2_playbook_on_start(self, playbook):
        pass

    def v2_playbook_on_play_start(self, play):
        self._play = play

    def v2_playbook_on_task_start(self, task, is_conditional):
        # 保留失败输出所需的标题状态，但不提前显示任务标题。
        self._task_type_cache[task._uuid] = "TASK"
        self._last_task_name = task.get_name().strip()

    def v2_runner_on_ok(self, result):
        if "npu_top_output" not in result.task.tags:
            return
        self._display.display(f"[{result.host.get_name()}] {result.task.get_name().strip()}")
        message = result.result.get("msg", "")
        lines = message if isinstance(message, list) else [message]
        for line in lines:
            self._display.display(str(line))

    def v2_runner_on_skipped(self, result):
        pass

    def v2_playbook_on_stats(self, stats):
        pass
