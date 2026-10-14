smoketest仅适合存放不需要网络、外部文件依赖、运行速度非常快的测试。

在本目录运行全部 smoketest：

```bash
uv run --with pytest python -m pytest smoketest -q
```
