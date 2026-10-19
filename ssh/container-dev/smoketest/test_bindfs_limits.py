from pathlib import Path
import subprocess
import sys


def test_raise_limit_preserves_higher_limits_and_is_idempotent():
    core = Path(__file__).resolve().parents[1] / "devtools/containers/core"
    code = """import resource,sys
sys.path.insert(0,sys.argv[1])
import bindfs_limits
bindfs_limits.NOFILE=2048
soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE,(1024,hard))
bindfs_limits.raise_limit(0)
assert resource.getrlimit(resource.RLIMIT_NOFILE)==(2048,hard)
bindfs_limits.raise_limit(0)
assert resource.getrlimit(resource.RLIMIT_NOFILE)==(2048,hard)
resource.setrlimit(resource.RLIMIT_NOFILE,(4096,hard))
bindfs_limits.raise_limit(0)
assert resource.getrlimit(resource.RLIMIT_NOFILE)==(4096,hard)
print("limits raised and preserved")
"""
    subprocess.run([sys.executable, "-c", code, str(core)], check=True)
