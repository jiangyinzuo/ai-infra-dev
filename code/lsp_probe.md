# LSP 诊断脚本

`lsp_probe.py` 绕过 Neovim，直接查询 Pyrefly。所有命令在 vLLM 目录执行，使用项目的 uv venv：

```bash
cd ~/ai-infra-dev/code/vllm-workspace/vllm
```

## 查找引用

```bash
uv run --no-sync --offline --python .venv/bin/python python ../../lsp_probe.py \
  --file vllm/v1/engine/output_processor.py \
  --line 551 --symbol OutputProcessor \
  --open vllm/v1/engine/async_llm.py
```

默认查询引用；`--open` 额外打开一个文件，可省略或重复指定。

## 跳转定义

```bash
uv run --no-sync --offline --python .venv/bin/python python ../../lsp_probe.py \
  --file vllm/v1/engine/async_llm.py \
  --line 180 --symbol OutputProcessor --action definition
```

## 查看类型与文档

```bash
uv run --no-sync --offline --python .venv/bin/python python ../../lsp_probe.py \
  --file vllm/v1/engine/output_processor.py \
  --line 551 --symbol OutputProcessor --action hover
```

## 常用选项

- `--line`：从 1 开始的行号，源码变化后需调整。
- `--symbol`：该行中唯一出现的符号文本；也可用 `--column` 指定从 1 开始的列号。
- `--wait 15`：查询前等待 15 秒；默认 5 秒，不保证索引已完成。
- `--root ~/ai-infra-dev`：指定 LSP 工作区根目录，默认是 `code/vllm-workspace/vllm`；不强制覆盖 Pyrefly 逐文件选择的配置。
- `--exclude-declaration`：引用结果不包含声明。
- `--help`：查看全部选项。

查询结果以 JSON 输出，位置中的行、列从 0 开始。服务端日志路径会打印到终端；脚本不修改源码或配置。
JSON 中的 `result_count` 表示结果条数：列表取长度，单个对象计 1，`null` 计 0；包含声明时，声明也计入总数。

每次运行都会启动全新的 LSP 会话，不继承 Neovim 已打开的文件和索引状态。`code/vllm-workspace/pyrefly.toml` 的 `project-includes = ["vllm/**/*.py"]` 覆盖 vLLM 源码和测试；`use-ignore-files = false` 避免父仓库的 Git 忽略规则将独立 checkout 排除出索引。Git 自身的忽略行为不受影响。也可显式打开文件：

```bash
--open tests/detokenizer/test_stop_string_while_stop_model_terminates.py
```

等待期间脚本会持续处理服务端配置等请求，让后台索引正常推进，而不是单纯 sleep。

## 代码结构

- `parse_arguments()`：解析参数，校验等待时间和超时。
- `prepare_query()`：读取文件、校验光标位置、构造 UTF-16 LSP 查询参数。
- `initialize_client()` / `open_documents()`：初始化协议会话、通知服务端打开文档。
- `run_query()`：串联初始化、打开文档、等待与查询，返回结果。
- `open_log()` / `main()`：创建日志，管理进程和文件生命周期，输出 JSON。