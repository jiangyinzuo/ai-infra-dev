"""双卡 SM120：GLM-5.3 tiny random P/D + HiSparse，仅供执行路径学习。"""

import json
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve_config import ServeConfig

_PD = runpy.run_path(str(Path(__file__).with_name("deepseek-v3.2-hisparse-pd.py")))

# 保留官方 MLA 维度、RoPE 和前三层 indexer + 后三层共享模式；只缩小计算规模。
_HF_OVERRIDES = {
    "num_hidden_layers": 6,
    "hidden_size": 256,
    "intermediate_size": 512,
    "moe_intermediate_size": 512,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "q_lora_rank": 256,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "indexer_types": ["full", "full", "full", "shared", "shared", "shared"],
    "mlp_layer_types": ["dense", "dense", "dense", "sparse", "sparse", "sparse"],
    "quantization_config": None,
}

config = ServeConfig(
    model="zai-org/GLM-5.3",
    device_count=2,
    serve_command=[sys.executable, str(Path(__file__).resolve()), "{model}"],
)


def service_commands(model):
    # 复用 P/D 学习配置，两个角色都保留 --disable-flashinfer-autotune。
    commands = _PD["service_commands"](model)
    for role in ("prefill", "decode"):
        command = commands[role]
        command[command.index("--hf-overrides") + 1] = json.dumps(_HF_OVERRIDES)
        command[command.index("--served-model-name") + 1] = "tiny-glm-5.3"
    return commands


if __name__ == "__main__":
    sys.exit(_PD["main"](service_commands))
