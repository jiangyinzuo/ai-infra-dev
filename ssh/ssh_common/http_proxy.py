"""Build ncat HTTP CONNECT arguments without logging proxy credentials."""

import os
from urllib.parse import urlsplit, unquote


def proxy_arguments(value):
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError
        host = parsed.hostname
        address = (
            ("[" + host + "]" if ":" in host else host) + ":" + str(parsed.port or 80)
        )
        args = ["--proxy", address, "--proxy-type", "http"]
        if parsed.username is not None:
            username = unquote(parsed.username)
            password = unquote(parsed.password or "")
            if ":" in username or any(c in username + password for c in "\r\n\0"):
                raise ValueError
            args += ["--proxy-auth", username + ":" + password]
        return args
    except ValueError:
        raise ValueError(
            "DEVTOOLS_HTTP_PROXY 必须是有效的 HTTP CONNECT 代理 URL"
        ) from None


if __name__ == "__main__":
    import sys

    try:
        args = proxy_arguments(os.environ.get("DEVTOOLS_HTTP_PROXY", ""))
        os.execvp("ncat", ["ncat", *args, *sys.argv[1:]])
    except (OSError, ValueError):
        print("error: GitHub HTTP CONNECT 代理配置无效或 ncat 不可用", file=sys.stderr)
        sys.exit(1)
