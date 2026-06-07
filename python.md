
```bash
# 允许同用户ptrace
sudo sysctl kernel.yama.ptrace_scope=0
py-spy dump --pid 3008106 --native
```

