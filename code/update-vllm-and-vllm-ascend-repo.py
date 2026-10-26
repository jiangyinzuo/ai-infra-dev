#!/usr/bin/env python3
"""更新 vLLM/Ascend 主仓库并对齐开发 worktree（只依赖 Python 3 和 Git）。

用法：python3 code/update-vllm-and-vllm-ascend-repo.py，可从任意目录调用。
首次运行克隆 main，后续只做 fast-forward，不重置本地提交或覆盖本地修改。
本脚本不安装依赖；失败可能保留已完成的更新，修复原因后可再次运行。
"""

import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Literal, overload


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE = CODE_DIR / "vllm-ascend-workspace"
VLLM = CODE_DIR / "vllm-workspace/vllm"
ASCEND = WORKSPACE / "vllm-ascend"
WORKTREE = WORKSPACE / "vllm"
VLLM_URL = "https://github.com/vllm-project/vllm.git"
ASCEND_URL = "https://github.com/vllm-project/vllm-ascend.git"
COMMIT_FILE = ASCEND / ".github/vllm-main-verified.commit"


@overload
def git(repo: Path | None, *args: str, capture: Literal[True]) -> str: ...


@overload
def git(repo: Path | None, *args: str, capture: Literal[False] = False) -> None: ...


def git(repo: Path | None, *args: str, capture: bool = False) -> str | None:
    """所有 Git 调用使用参数数组，避免路径或参数被 shell 解释。"""
    command = ["git"]
    if repo is not None:
        command += ["-C", str(repo)]
    command += list(args)
    print(f"{shlex.join(command)}", flush=True)
    result = subprocess.run(
        command, check=True, text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else None


def check_clean(repo: Path) -> None:
    # 检查 .git，防止 Git 将普通目录误识别为外层仓库的一部分。
    if not (repo / ".git").exists():
        raise ValueError(f"不是 Git 仓库或 worktree：{repo}")
    git(repo, "rev-parse", "--is-inside-work-tree", capture=True)
    if git(repo, "status", "--porcelain", "--untracked-files=all", capture=True):
        raise ValueError(f"存在本地修改（包括未跟踪文件），请先提交或 stash：{repo}")


def check_main_repo(repo: Path, url: str) -> None:
    if not repo.exists():
        return
    check_clean(repo)
    # 读取原始 URL，允许 Git 的 insteadOf 配置将官方地址重写为镜像。
    if git(repo, "config", "--get", "remote.origin.url", capture=True) != url:
        raise ValueError(f"origin 与预期不符：{repo}，预期 {url}")
    if git(repo, "symbolic-ref", "--quiet", "--short", "HEAD", capture=True) != "main":
        raise ValueError(f"主仓库必须位于 main 分支：{repo}")


def common_dir(repo: Path) -> Path:
    return Path(git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir", capture=True)).resolve()


def check_worktree() -> None:
    if not WORKTREE.exists():
        return
    check_clean(WORKTREE)
    if not VLLM.exists() or common_dir(WORKTREE) != common_dir(VLLM):
        raise ValueError(f"{WORKTREE} 不是 {VLLM} 的 worktree")
    # 拒绝切走用户创建的分支；本脚本只维护 detached HEAD。
    if git(WORKTREE, "rev-parse", "--abbrev-ref", "HEAD", capture=True) != "HEAD":
        raise ValueError(f"对齐 worktree 必须处于 detached HEAD：{WORKTREE}")


def update_main(repo: Path, url: str) -> None:
    if not repo.exists():
        git(None, "clone", "--branch", "main", url, str(repo))
    else:
        git(repo, "fetch", "origin", "main")
        git(repo, "merge", "--ff-only", "origin/main")


def main() -> None:
    if len(sys.argv) != 1:
        raise ValueError("用法：python3 code/update-vllm-and-vllm-ascend-repo.py（不接受参数）")
    # 先检查所有现有目录，避免明知另一目录有修改时仍更新第一个仓库。
    check_main_repo(VLLM, VLLM_URL)
    check_main_repo(ASCEND, ASCEND_URL)
    check_worktree()
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    update_main(VLLM, VLLM_URL)
    update_main(ASCEND, ASCEND_URL)

    # 使用 Ascend 仓库记录的验证版本，而不是猜测当前 vLLM main 是否兼容。
    commit = COMMIT_FILE.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError(f"验证版本必须是完整的 40 位 Git SHA：{COMMIT_FILE}")
    commit = commit.lower()
    command = ["git", "-C", str(VLLM), "cat-file", "-e", f"{commit}^{{commit}}"]
    print(f"$ {shlex.join(command)}", flush=True)
    probe = subprocess.run(
        command,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if probe.returncode:
        git(VLLM, "fetch", "--no-tags", "origin", commit)
    if WORKTREE.exists():
        git(WORKTREE, "checkout", "--detach", commit)
    else:
        git(VLLM, "worktree", "add", "--detach", str(WORKTREE), commit)
    print(f"vLLM main：{git(VLLM, 'rev-parse', 'HEAD', capture=True)}")
    print(f"Ascend main：{git(ASCEND, 'rev-parse', 'HEAD', capture=True)}")
    print(f"已对齐 {WORKTREE} 到 {commit}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(1)
