## 双卡 DeepSeek-V3.2 P/D + HiSparse

`dev/serve_config/deepseek-v3.2-hisparse-pd.py` 复用单卡配置的缩小版 DSA
架构与 dummy 随机权重，不下载完整权重，不用于质量或性能评估。
需要两张 SM120 GPU，以及 DeepSeek-V3.2 的配置和 tokenizer。

先将 `P_GPU_UUID`、`D_GPU_UUID` 设置为两张卡的完整 UUID；
在本目录安装 NIXL 后启动（设备顺序为 P、D）：

```bash
uv pip install --python vllm-workspace/vllm/.venv/bin/python 'nixl==1.5.2' 'nixl-cu13==1.5.2'
vllm-workspace/vllm/.venv/bin/python dev vllm serve \
  --script deepseek-v3.2-hisparse-pd.py \
  --devices "${P_GPU_UUID:?请设置P卡完整UUID},${D_GPU_UUID:?请设置D卡完整UUID}"
```
省略 `--devices` 时进入交互选择，提示会显示配置要求的设备数量。
fzf 中用 ↑/↓ 移动，Tab 选中/取消，Enter 提交，Esc 取消；双卡配置需选中两项。


P、D 的 HTTP 端口分别为 8100、8200，NIXL 控制端口为 5600、5601；
均绑定本机。通过代理 `http://127.0.0.1:8192/v1/completions` 发请求，
模型名为 `tiny-dsa`。只有 D 开启 HiSparse。启动提示不代表 API 就绪，
可分别请求 P、D 的 `/health` 检查。

tiny 模型的历史 KV 可能全部驻留 GPU；长输入不保证触发 Host 换入。
检查 D 侧 `/metrics` 中 `vllm:hisparse_host_to_device_bytes_total` 是否增长，
才能确认实际发生了 HiSparse Host→GPU 换入。
