"""SSH 和 Docker 共用的环境透传名单；不存储变量值。"""

PROXY_ENV_KEYS = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)
ENV_KEYS = ("USG_IP", *PROXY_ENV_KEYS, "SUDOCODE_API_KEY", "OPENAI_API_KEY")
