# GLM MoE DSA

## 运行小模型

在当前目录、已激活 `view_model/.venv` 的环境中运行：

```bash
uv pip install -r ../requirements.txt
HF_ENDPOINT=https://hf-mirror.com python run_tiny_model.py
```

`HF_ENDPOINT` 只对这条命令生效，不修改共享的 `../gitee-proxy.sh`。当前 Gitee 端点对该模型的仓库 API 返回 401，`tokenizer_config.json` 请求返回 404，可能最终表现为 tokenizer 初始化失败；不应据此判断缺少 `sentencepiece`。可以直连 Hugging Face 时，改用 `HF_ENDPOINT=https://huggingface.co`。

首次运行下载约 1.24 GB 权重及 tokenizer；缓存位置遵循现有的 `HF_HOME` / `HF_HUB_CACHE`。下载完成后可以离线运行：

```bash
HF_HUB_OFFLINE=1 python run_tiny_model.py
```

脚本保留权重配置的 BF16，CUDA 可用时使用第一张可见 GPU，否则使用 CPU；不自动分片，不依赖 `accelerate`。可用 `CUDA_VISIBLE_DEVICES` 选择 GPU。

## 学习范围

[模型仓库](https://huggingface.co/inference-optimization/GLM-5.3-0.6B-A0.4B)的架构为 `GlmMoeDsaForCausalLM`，共 621,752,064 个参数：6 层，包含 MLA、DSA indexer、dense / MoE MLP、路由专家和共享专家。当前环境的 Transformers 5.20.2 已验证可加载。

它不是完整 GLM-5.3 的等价缩小版：模型卡说明权重随机初始化后仅在玩具文本数据上训练，且没有 MTP 层，权重使用 BF16 而非原模型的 FP8。适合观察结构与执行路径，不适合评估对话能力；重复输出不等于下载或加载失败。

## 不下载权重，查看官方 GLM-5.3 参数

`print_model_parameters.py` 直接读取官方 `zai-org/GLM-5.3` 的 checkpoint 张量元数据，不构建模型、不下载权重数据、不需要 GPU。当前环境的 huggingface-hub 2.3.0 已验证可用。

```bash
# 默认输出全部张量，可重定向保存
HF_ENDPOINT=https://hf-mirror.com python print_model_parameters.py > glm53-parameters.txt

# 只看第 3 层的第 0 个路由专家
HF_ENDPOINT=https://hf-mirror.com python print_model_parameters.py --prefix model.layers.3.mlp.experts.0.

# 只看 MTP 层
HF_ENDPOINT=https://hf-mirror.com python print_model_parameters.py --prefix model.layers.78.
```

逐张量输出 checkpoint 原始名称、shape、存储 dtype、元素数量、数据字节数和分片名，包含 FP8 权重及其 FP32 `weight_scale_inv`。`--prefix` 仅筛选逐张量输出，全局统计仍覆盖全部 checkpoint；打印的字节数是权重的存储大小，不是本次网络下载量。

脚本每次联网调用 Hub 的 safetensors 元数据接口，读取 `main` 当前版本；不读取权重数值，也不执行推理。元数据包含量化 scale 和 buffer，因此全部张量元素数不能直接等同于模型的可训练参数量。完整输出已验证包含 118,629 个张量，其中 `model.layers.78.*` 的 MTP 张量为 1,569 个。

## 不加载权重，打印 Transformers 模型结构

```bash
HF_ENDPOINT=https://hf-mirror.com python glm_moe_dsa_class.py
```

只下载官方 `zai-org/GLM-5.3` 的配置，以 `meta` 张量构建 `GlmMoeDsaForCausalLM`，打印类名、源码文件位置和原始 `print(model)` 输出。不下载权重、不需要 GPU、不分配参数的实际存储；Python、PyTorch 和模块对象仍占用内存。

**首次必须先执行上面的联网命令，成功缓存配置后才能离线运行。** 缓存位置由当前终端的 `HF_HOME` / `HF_HUB_CACHE` 决定；未设置时默认使用 `~/.cache/huggingface/hub`。激活同一 Python 虚拟环境不会同步这些变量，更换终端后需保持相同缓存设置，或重新执行联网命令下载配置。`HF_HUB_OFFLINE=1` 不会下载缺失的配置：

```bash
HF_HUB_OFFLINE=1 python glm_moe_dsa_class.py
```

当前 Transformers 5.20.2 实跑构建 78 个 decoder 层、743,377,000,704 个参数元素，参数和 buffer 全部在 `meta` 上。模型没有真实权重数值，不能直接用于推理。

这展示的是 Transformers 类的结构，不是原始 checkpoint 布局：当前实现不构建 MTP 层，专家投影合并为三维参数，按配置构建的参数为 BF16，而非 checkpoint 的 FP8 权重及其量化 scale。查看后者请使用 `print_model_parameters.py`。
