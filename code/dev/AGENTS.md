开发脚本要求 Python >=3.11；共享 SSH、环境检测逻辑复用仓库 ssh/ssh_common。

smoketest/ 使用 pytest，必须快速、离线、隔离；不调用真实 Docker、SSH 或 pip 安装。可用临时 pip 替身验证子进程行为。

在本目录运行：
```bash
uv run --with pytest python -m pytest smoketest -q
```

README 面向使用者，不记录冒烟测试操作。
