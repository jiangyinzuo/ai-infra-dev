"""GLM-5.3：P 侧 Mooncake 共享池，NIXL pull，D 侧 HiSparse。"""

import json
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve_config import ServeConfig

_GLM = runpy.run_path(str(Path(__file__).with_name("glm-5.3-hisparse-pd.py")))

config = ServeConfig(
    model=_GLM["config"].model,
    device_count=2,
    serve_command=[sys.executable, str(Path(__file__).resolve()), "{model}"],
)


def service_commands(model):
    commands = _GLM["service_commands"](model)
    prefill = commands["prefill"]
    index = prefill.index("--kv-transfer-config") + 1
    nixl = json.loads(prefill[index])
    engine_id = nixl.pop("engine_id")
    prefill[index] = json.dumps({
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        "engine_id": engine_id,
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {
            "connectors": [
                nixl,
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "cache_prefix": "tiny-glm-5.3-pd",
                    },
                },
            ],
        },
    })
    return commands


if __name__ == "__main__":
    sys.exit(_GLM["_PD"]["main"](service_commands))
