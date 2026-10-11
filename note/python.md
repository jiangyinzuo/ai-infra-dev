
```bash
# 允许同用户ptrace
sudo sysctl kernel.yama.ptrace_scope=0
py-spy dump --pid 3008106 --native
```

uv安装命令行工具
```bash
uv tool install pylint pyrefly ansible ansible-core
```

ansible: 批量执行ssh命令的工具
