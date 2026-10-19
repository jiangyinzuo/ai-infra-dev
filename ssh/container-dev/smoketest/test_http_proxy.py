import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ssh_common.http_proxy import proxy_arguments


def test_encoded_credentials_reach_ncat_without_shell_expansion(tmp_path):
    ncat = tmp_path / "ncat"
    ncat.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    ncat.chmod(0o755)
    helper = Path(__file__).resolve().parents[2] / "ssh_common/http_proxy.py"
    environment = dict(
        os.environ,
        PATH=str(tmp_path) + ":" + os.environ["PATH"],
        DEVTOOLS_HTTP_PROXY="http://test-user:p%2B%25%24%28test%29@proxy.example:8080",
    )
    result = subprocess.run(
        [sys.executable, str(helper), "ssh.github.com", "443"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [
        "--proxy",
        "proxy.example:8080",
        "--proxy-type",
        "http",
        "--proxy-auth",
        "test-user:p+%$(test)",
        "ssh.github.com",
        "443",
    ]


@pytest.mark.parametrize(
    "value", ["https://user:secret@proxy:8080", "http://proxy:bad", "http://proxy/path"]
)
def test_invalid_proxy_errors_do_not_disclose_credentials(value):
    with pytest.raises(ValueError) as error:
        proxy_arguments(value)
    assert value not in str(error.value)
    assert "secret" not in str(error.value)
