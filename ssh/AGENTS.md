## Python 格式化

`ssh/ruff.toml` 统一配置本目录及子目录的 Python 格式。修改 Python 文件后，从仓库根目录执行：

```bash
uvx ruff format ssh
uvx ruff format --check ssh
```

仅统一格式，不自动修复 lint 问题。
