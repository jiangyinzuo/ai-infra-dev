"""Repository code and resources; DEVTOOLS_DATA_DIR overrides this host only."""

import os
from pathlib import Path

DEVTOOLS = Path(__file__).resolve().parents[1]
# Both local and remote resources follow the checkout, never the shell cwd.
override = os.environ.get("DEVTOOLS_DATA_DIR")
if override:
    DATA_ROOT = Path(override).expanduser()
    if not DATA_ROOT.is_absolute():
        raise ValueError("DEVTOOLS_DATA_DIR 必须是绝对路径")
else:
    DATA_ROOT = DEVTOOLS
PLATFORM_DIR = DATA_ROOT / "openeuler24.03-lts-aarch64"
