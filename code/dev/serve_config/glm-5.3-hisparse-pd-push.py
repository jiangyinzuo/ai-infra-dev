"""GLM-5.3 HiSparse + NIXL push；当前会报 mixed-memory KV caches 不支持。"""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve_config import ServeConfig

_PULL = runpy.run_path(str(Path(__file__).with_name("glm-5.3-hisparse-pd.py")))

config = ServeConfig(
    model=_PULL["config"].model,
    device_count=_PULL["config"].device_count,
    serve_command=[sys.executable, str(Path(__file__).resolve()), "{model}"],
)


def service_commands(model):
    return _PULL["service_commands"](model, transfer_mode="push")


if __name__ == "__main__":
    sys.exit(_PULL["_PD"]["main"](service_commands))
