Python 脚本须兼容 Python 3.9；不要使用 match/case 等更高版本语法。Ruff 目标版本在上级 ruff.toml 中配置。
共享配置统一维护在 devtools/config.py；WSL 远程调用通过 devtools/remote_exec.py 和 ssh2 执行。

smoketest仅适合存放不需要网络、外部文件依赖、运行速度非常快的测试。

在本目录运行全部 smoketest：

```bash
uv run --with pytest python -m pytest smoketest -q
```
