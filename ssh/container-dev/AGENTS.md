Python 脚本须兼容 Python 3.9；不要使用 match/case 等更高版本语法。Ruff 目标版本在上级 ruff.toml 中配置。
跨子系统共享配置、SSH 路由及机器信息接口统一维护在 ../ssh_common/；远程调用通过 ../ssh_common/remote.py 和 ssh2 执行。两端代码及默认资源路径均基于 ai-infra-dev 仓库定位，不依赖当前工作目录；sync 仅传资源。

smoketest仅适合存放不需要网络、外部文件依赖、运行速度非常快的测试。

在本目录运行全部 smoketest：

```bash
uv run --with pytest python -m pytest smoketest -q
```
