"""共享 CLI 参数，与具体业务无关。"""

import argparse


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def add_jobs_argument(parser: argparse.ArgumentParser, default: int = 8) -> None:
    parser.add_argument(
        "--jobs",
        type=positive_int,
        default=default,
        help=f"并发任务数（正整数，默认 {default}）",
    )
