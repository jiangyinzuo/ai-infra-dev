## 更新 vLLM 与 vLLM-Ascend 源码

从任意目录执行（只依赖 Python 3 和 Git，不安装依赖）：

```bash
python3 ~/ai-infra-dev/code/update-vllm-and-vllm-ascend-repo.py
```

以下路径相对本 README 所在的 `code/` 目录：

| 路径 | 操作 |
|---|---|
| `vllm-workspace/vllm` | 首次克隆 vLLM，后续仅 fast-forward 更新 `main` |
| `vllm-ascend-workspace/vllm-ascend` | 首次克隆 vLLM-Ascend，后续仅 fast-forward 更新 `main` |
| `vllm-ascend-workspace/vllm` | 创建或对齐 vLLM worktree，保持 detached HEAD，版本取自 `vllm-ascend-workspace/vllm-ascend/.github/vllm-main-verified.commit` |

每条 Git 命令在执行前打印，包含检查、拉取和 worktree 对齐命令。

已有主仓库必须位于 `main` 且 origin 为官方 HTTPS URL（可用 Git `insteadOf` 重写为镜像）。所有受管理仓库和 worktree 必须无本地修改，包括未跟踪文件。失败可能保留已完成的更新，修复原因后可重新运行，不会重置本地提交或覆盖本地修改。

## 安装vLLM或vLLM Ascend

见 ~/ai-infra-dev/code/dev/
