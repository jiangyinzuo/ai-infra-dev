## 不下载权重打印 Transformers 类

脚本 `deepseek_v32_class.py` 只读取 `deepseek-ai/DeepSeek-V3.2` 的配置，在 PyTorch `meta` 设备上构建 `DeepseekV32ForCausalLM`，然后打印 Transformers 模型结构。它不下载 checkpoint 权重，不需要 GPU，也不能执行推理。

```bash
# 首次运行：只下载配置
HF_ENDPOINT=https://hf-mirror.com python deepseek_v32_class.py

# 配置已缓存后：完全离线
HF_HUB_OFFLINE=1 python deepseek_v32_class.py
```

当前 Transformers 实现构建 61 个主干 decoder 层；官方配置还声明 1 个 MTP 层，但该类不把 `model.layers.61.*` 构建为模型模块。打印结果是 Transformers 类的结构，不是 checkpoint 原始张量布局。

`deepseek_v31_class.py` 以相同方式查看 `deepseek-ai/DeepSeek-V3.1`，实际构建的类为 `DeepseekV3ForCausalLM`；同样只读取配置、不下载权重。

```bash
HF_ENDPOINT=https://hf-mirror.com python deepseek_v31_class.py
HF_HUB_OFFLINE=1 python deepseek_v31_class.py
```

## HTML 实时预览

```bash
typst watch --features html --port 3000 deepseek-v3-v32-glm5-2.typ deepseek-v3-v32-glm5-2.html
```

服务启动后访问 `http://localhost:3000`。

## 导出 SVG

```bash
typst compile diagrams/deepseek-mla.typ diagrams/deepseek-mla.svg
for path in qk value; do
  for stage in before after; do
    typst compile --input path="$path" --input stage="$stage" diagrams/deepseek-absorption.typ "diagrams/deepseek-absorption-$path-$stage.svg"
  done
done
typst compile diagrams/deepseek-dsa.typ diagrams/deepseek-dsa.svg
typst compile diagrams/deepseek-dsa-absorption.typ diagrams/deepseek-dsa-absorption.svg
```

