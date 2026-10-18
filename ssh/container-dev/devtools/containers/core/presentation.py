"""供列表与清理流程共用的映射展示。"""


def print_mappings(rows: list) -> None:
    print("bindfs 映射：")
    if not rows:
        print("  无")
    for row in rows:
        status = (
            "可清理"
            if row["cleanup_allowed"]
            else "保留：" + (",".join(row["containers"]) or "无法验证")
        )
        print(
            "  {} -> {} | {} | 源目录{} | {}".format(
                row["source"],
                row["target"],
                "已挂载" if row["mounted"] else "未挂载",
                "存在" if row["source_exists"] else "已删除",
                status,
            )
        )
