"""WSL 与服务器共用的配置；不存储密钥值。"""

from dataclasses import dataclass

HOST_GROUPS = {"suzhou": "windows", "xiaoqiaoling": "wsl", "lianqiuhu": "windows"}
SSH_ROUTES = {
    "suzhou": ("/mnt/c/Windows/System32/OpenSSH/ssh.exe", "USG0_IP"),
    "lianqiuhu": ("/mnt/c/Windows/System32/OpenSSH/ssh.exe", "USG1_IP"),
    "xiaoqiaoling": ("/usr/bin/ssh", None),
}


@dataclass(frozen=True)
class SSHSource:
    config: str
    ssh_config: str
    public_key: str
    password_method: str


SSH_SOURCES = {
    "windows": SSHSource(
        "${WINDOWS_SSH_CONFIG}",
        "${WINDOWS_SSH_CONFIG_NATIVE}",
        "${WINDOWS_SSH_HOME}/id_ed25519.pub",
        "windows-askpass",
    ),
    "wsl": SSHSource(
        "~/.ssh/config", "~/.ssh/config", "${WSL_SSH_HOME}/id_ed25519.pub", "sshpass"
    ),
}

REMOTE_USER = "jiangyinzuo"
REMOTE_DIR = "~/devtools"
PROXY_ENV_KEYS = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)
HOST_ENV_KEYS = ("USG_IP", *PROXY_ENV_KEYS, "SUDOCODE_API_KEY", "OPENAI_API_KEY")
# SoC 由服务器宿主机决定，不能从 WSL 经 SSH 覆盖。
CONTAINER_ENV_KEYS = (*HOST_ENV_KEYS, "SOC_VERSION")
TOOL_PLATFORMS = {
    ("openeuler", "aarch64"): "openeuler24.03-lts-aarch64",
    ("openeuler", "arm64"): "openeuler24.03-lts-aarch64",
}
NPU_PLATFORMS = {"a3": "800I-A3-py311-openEuler24.03-lts-aarch64"}
