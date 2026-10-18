#!/usr/bin/env python3
"""删除 jiangyinzuo 容器；可保留 bindfs 供重建复用。"""

import argparse
import os
import sys
import subprocess
from mappings import MappingStore, containers, run


def managed(name: str) -> bool:
    return name.startswith("jiangyinzuo-")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="容器名称；省略则交互选择")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--keep-bindfs", action="store_true", help="删除容器和匿名 volume，保留映射"
    )
    mode.add_argument(
        "--remove-bindfs", action="store_true", help="另清理未被其他容器引用的托管映射"
    )
    args = parser.parse_args()
    if os.getuid() == 0:
        parser.error("请以创建容器的普通用户运行，不要 sudo")
    store = MappingStore()
    with store.locked():
        choices = [c for c in containers() if managed(c["Name"].lstrip("/"))]
        if not choices:
            if args.name:
                parser.error("未找到指定的 jiangyinzuo 容器")
            print("没有可管理的 jiangyinzuo 容器")
            return
        if args.name:
            matches = [c for c in choices if c["Name"].lstrip("/") == args.name]
            if not matches:
                parser.error("未找到指定的 jiangyinzuo 容器")
            selected = matches[0]
        else:
            for index, c in enumerate(choices, 1):
                print(
                    "{}）{} [{}]".format(
                        index, c["Name"].lstrip("/"), c["State"]["Status"]
                    )
                )
            while True:
                answer = input("选择要删除的容器编号（q 取消）：").strip()
                if answer.lower() == "q":
                    return
                if answer.isdigit() and 1 <= int(answer) <= len(choices):
                    selected = choices[int(answer) - 1]
                    break
        if not args.keep_bindfs and not args.remove_bindfs:
            while True:
                answer = (
                    input(
                        "1）仅删除容器及匿名 volume，保留映射 [默认]；2）同时清理 bindfs；q）取消："
                    ).strip()
                    or "1"
                )
                if answer.lower() == "q":
                    return
                if answer in ("1", "2"):
                    args.remove_bindfs = answer == "2"
                    break
        name = selected["Name"].lstrip("/")
        sources = [
            m["Source"] for m in selected.get("Mounts", []) if m.get("Type") == "bind"
        ]
        print(
            "删除 {}；{}".format(
                name, "清理空闲托管映射" if args.remove_bindfs else "保留 bindfs 映射"
            )
        )
        if selected["State"]["Running"]:
            run("docker", "stop", selected["Id"])
        run("docker", "rm", "-v", selected["Id"])
        if args.remove_bindfs:
            store.remove_unused(sources, containers())
            for source in sources:
                if "root-mapped-" in source:
                    print("旧版映射无托管记录，保留，请核对后手动卸载：" + source)
        print("完成：" + name)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, EOFError, subprocess.SubprocessError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        sys.exit(1)
