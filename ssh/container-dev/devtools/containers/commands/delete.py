"""Interactive container deletion and explicit orphan cleanup."""

import argparse
import os
from typing import Optional

from core.mappings import MappingStore
from core.process import containers, managed, run
from core.residuals import scan, cleanup
from .ls import print_mappings


def select_container(choices: list, name: Optional[str]) -> Optional[dict]:
    if name:
        for container in choices:
            if container["Name"].lstrip("/") == name:
                return container
        raise ValueError("未找到指定的 jiangyinzuo 容器")
    if not choices:
        print("没有可管理的容器；可用 cli.py ls 检查映射")
        return None
    for index, container in enumerate(choices, 1):
        print(
            "{}）{} [{}]".format(
                index, container["Name"].lstrip("/"), container["State"]["Status"]
            )
        )
    while True:
        answer = input("选择要删除的容器编号（q 取消）：").strip()
        if answer.lower() == "q":
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]


def select_removal(args: argparse.Namespace) -> Optional[bool]:
    if args.keep_bindfs or args.remove_bindfs:
        return args.remove_bindfs
    while True:
        answer = (
            input(
                "1）删除容器及匿名 volume，保留映射 [默认]；2）同时清理 bindfs；q）取消："
            ).strip()
            or "1"
        )
        if answer.lower() == "q":
            return None
        if answer in ("1", "2"):
            return answer == "2"


def cleanup_mappings(store: MappingStore, yes: bool) -> None:
    rows = scan(store, containers())
    print_mappings(rows)
    unused = [row for row in rows if row["cleanup_allowed"]]
    if unused and (yes or input("清理以上未引用映射？[y/N]：").strip().lower() == "y"):
        cleanup(store, unused)


def execute(args: argparse.Namespace) -> None:
    if args.cleanup_bindfs and (args.name or args.keep_bindfs or args.remove_bindfs):
        raise ValueError("--cleanup-bindfs 不能与容器删除参数混用")
    if args.yes and not args.cleanup_bindfs:
        raise ValueError("--yes 仅用于 --cleanup-bindfs")
    if os.getuid() == 0:
        raise ValueError("请以创建容器的普通用户运行，不要 sudo")
    store = MappingStore()
    with store.locked():
        if args.cleanup_bindfs:
            cleanup_mappings(store, args.yes)
            return
        selected = select_container(
            [c for c in containers() if managed(c["Name"])], args.name
        )
        if selected is None:
            return
        remove = select_removal(args)
        if remove is None:
            return
        name = selected["Name"].lstrip("/")
        sources = [
            m["Source"] for m in selected.get("Mounts", []) if m.get("Type") == "bind"
        ]
        print(
            "删除 {}；{}".format(
                name, "清理空闲托管映射" if remove else "保留 bindfs 映射"
            )
        )
        if selected["State"]["Running"]:
            run("docker", "stop", selected["Id"])
        run("docker", "rm", "-v", selected["Id"])
        if remove:
            store.remove_unused(sources, containers())
            for source in sources:
                if "root-mapped-" in source:
                    print(
                        "旧版映射请使用 cli.py delete --cleanup-bindfs 检查清理："
                        + source
                    )
        print("完成：" + name)
