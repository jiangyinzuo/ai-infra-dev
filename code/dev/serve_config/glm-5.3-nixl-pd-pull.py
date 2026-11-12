"""GLM-5.3 tiny random P/D，纯 NIXL pull，无 HiSparse。"""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve_config import ServeConfig

_GLM = runpy.run_path(str(Path(__file__).with_name("glm-5.3-hisparse-pd.py")))

config = ServeConfig(
    model=_GLM["config"].model,
    device_count=_GLM["config"].device_count,
    serve_command=[sys.executable, str(Path(__file__).resolve()), "{model}"],
)


def service_commands(model):
    return _GLM["service_commands"](model, transfer_mode="pull", hisparse=False)


if __name__ == "__main__":
    sys.exit(_GLM["_PD"]["main"](service_commands))
