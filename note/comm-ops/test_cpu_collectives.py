import ast
import re
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path


class CollectiveOutputTest(unittest.TestCase):
    def test_four_ranks_emit_complete_lines(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                "--nproc-per-node=4",
                str(Path(__file__).with_name("cpu_collectives.py")),
            ],
            capture_output=True,
            encoding="utf-8",
            timeout=90,
            check=True,
        )
        events = Counter()
        operation_groups = []
        ranks_by_operation = {}
        for line in result.stdout.splitlines():
            match = re.fullmatch(r"\[(\w+)\] rank=([0-3]): (.+)", line)
            self.assertIsNotNone(match, line)
            operation, rank, payload = match.groups()
            ranks_by_operation.setdefault(operation, []).append(int(rank))
            if not operation_groups or operation_groups[-1] != operation:
                operation_groups.append(operation)
            self.assertNotRegex(payload, r"\[\w+\] rank=", line)
            events[operation, int(rank), payload.split("=", 1)[0]] += 1
            if operation in ("all_to_all", "reduce", "all_reduce", "reduce_scatter"):
                input_text, output_text = payload.removeprefix("input=").split(
                    ", output=", 1
                )
                rank_value = int(rank)
                if operation == "all_to_all":
                    expected_input = list(range(rank_value * 4, (rank_value + 1) * 4))
                    expected_output = [source * 4 + rank_value for source in range(4)]
                else:
                    expected_input = [
                        value
                        for destination in range(1, 5)
                        for value in ((rank_value + 1) * destination, 10 * (rank_value + 1) * destination)
                    ]
                    expected_output = (
                        [10 * (rank_value + 1), 100 * (rank_value + 1)]
                        if operation == "reduce_scatter"
                        else [10, 100, 20, 200, 30, 300, 40, 400]
                    )
                self.assertEqual(ast.literal_eval(input_text), expected_input)
                if operation == "reduce" and rank_value != 0:
                    self.assertEqual(output_text, "仅 rank=0 接收结果")
                else:
                    self.assertEqual(ast.literal_eval(output_text), expected_output)

        expected = Counter()
        for rank in range(4):
            for operation in (
                "broadcast", "reduce", "all_reduce", "reduce_scatter", "gather", "all_gather", "all_to_all"
            ):
                expected[operation, rank, "input"] = 1
            expected["scatter", rank, "output"] = 1
        expected["scatter", 0, "chunks"] = 1
        self.assertEqual(events, expected)
        self.assertEqual(
            operation_groups,
            ["broadcast", "reduce", "all_reduce", "reduce_scatter", "gather", "all_gather", "scatter", "all_to_all"],
        )
        for operation in operation_groups:
            self.assertEqual(
                ranks_by_operation[operation],
                [0, 0, 1, 2, 3] if operation == "scatter" else [0, 1, 2, 3],
                operation,
            )


if __name__ == "__main__":
    unittest.main()
