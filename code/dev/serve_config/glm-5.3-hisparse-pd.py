"""GLM-5.3 tiny random P/D + HiSparse，默认 NIXL pull。"""

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


def service_commands(model, *, transfer_mode="pull", hisparse=True):
    # 复用 P/D 学习配置，两个角色都保留 --disable-flashinfer-autotune。
    commands = _PD["service_commands"](model)
    for role in ("prefill", "decode"):
        command = commands[role]
        command[command.index("--hf-overrides") + 1] = json.dumps(_HF_OVERRIDES)
        command[command.index("--served-model-name") + 1] = "tiny-glm-5.3"
        index = command.index("--kv-transfer-config") + 1
        transfer = json.loads(command[index])
        if role == "prefill":
            nixl = transfer
        else:
            nixl = transfer["kv_connector_extra_config"]["connectors"][0]
        transfer["engine_id"] = f"glm-5.3-{role}"
        if hisparse:
            nixl["kv_connector_extra_config"] = {"transfer_mode": transfer_mode}
        else:
            transfer.update(
                kv_connector="NixlConnector",
                kv_role="kv_producer" if role == "prefill" else "kv_consumer",
                kv_load_failure_policy="fail",
                kv_connector_extra_config={"transfer_mode": transfer_mode},
            )
            if role == "decode":
                attention_index = command.index("--attention-config") + 1
                attention = json.loads(command[attention_index])
                attention.pop("hisparse_config", None)
                command[attention_index] = json.dumps(attention)
        command[index] = json.dumps(transfer)
    if transfer_mode == "push":
        proxy = (
            Path(__file__).resolve().parents[2]
            / "vllm-workspace/vllm/examples/disaggregated/disaggregated_serving"
            / "disagg_proxy_pushconnector_demo.py"
        )
        commands["proxy"] = [
            sys.executable, str(proxy),
            "--model", "tiny-glm-5.3",
            "--port", "8192",
            "--prefill", "127.0.0.1:8100",
            "--decode", "127.0.0.1:8200",
            "--prefill-engine-id", "glm-5.3-prefill",
            "--prefill-kv-host", "127.0.0.1",
            "--prefill-side-channel-port", "5600",
            "--prefill-tp-size", "1",
            "--prefill-pp-size", "1",
        ]
    return commands


if __name__ == "__main__":
    sys.exit(_PD["main"](service_commands))
