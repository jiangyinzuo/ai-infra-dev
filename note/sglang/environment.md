```bash
# https://hf-mirror.com/
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
export HF_ENDPOINT=https://hf-mirror.com
apt install aria2

./hfd.sh Qwen/Qwen2.5-7B-Instruct
```


```bash
apt install numactl
pip install -e "python"
```

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NCCL_DEBUG=INFO \
PYTHONUNBUFFERED=1 \
python3 -m sglang.launch_server \
  --model-path /root/autodl-tmp/sglang/Qwen2.5-7B-Instruct \
  --tp-size 1 \
  --pp-size 2 \
  --context-length 1024 \
  --max-running-requests 1 \
  --mem-fraction-static 0.7 \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --pp-async-batch-depth 0 \
  --trust-remote-code
```
