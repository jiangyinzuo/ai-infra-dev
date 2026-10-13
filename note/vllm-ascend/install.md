## 初始化bash环境

.bashrc中应该会有
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

- libascendcl.so：昇腾计算语言（AscendCL）库
- libatb.so：昇腾加速库（Ascend Transformer Boost，ATB）库：矩阵乘、自注意力、Norm、激活函数等常用算子

验证torch_npu
```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 - <<'PY'
import torch
import torch_npu

x = torch.tensor([1.0, 2.0], device="npu:0")
print((x + 1).cpu())
PY
```

## （编译与）安装

COMPILE_CUSTOM_KERNELS: 不编译kernel。默认为1
```bash

VLLM_TARGET_DEVICE=empty COMPILE_CUSTOM_KERNELS=0 pip3 install -v -e . --no-deps --no-build-isolation
```

查看安装结果
```bash
pip show vllm
pip show vllm_ascend
python -c "import vllm; print(vllm.__version__)"
python -c "import vllm_ascend; print(vllm_ascend.__version__)"
```

## 运行

启动: https://docs.vllm.ai/projects/ascend/en/latest/getting_started/quick_start.html

