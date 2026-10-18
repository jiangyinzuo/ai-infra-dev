import io
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
from workflow import nvim_plugins
from cli import build_parser
from workflow import binaries


def archive_bytes(payload):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        entry = tarfile.TarInfo("sample-commit/plugin.lua")
        entry.size = len(payload)
        archive.addfile(entry, io.BytesIO(payload))
    return buffer.getvalue()


def test_plugin_download_reuses_matching_url_and_hash(tmp_path, monkeypatch):
    destination = tmp_path / "sample.tar.gz"
    destination.write_bytes(archive_bytes(b"return true"))
    url = "https://example.invalid/sample/archive/commit.tar.gz"
    cached = {"url": url, "sha256": nvim_plugins.sha256(destination)}
    original = destination.read_bytes()
    initial_mtime = destination.stat().st_mtime_ns

    def unexpected_request(*args, **kwargs):
        pytest.fail("cached plugin must not make a network request")

    monkeypatch.setattr(nvim_plugins.urllib.request, "urlopen", unexpected_request)
    assert nvim_plugins.download(url, destination, cached) == cached
    assert destination.read_bytes() == original
    assert destination.stat().st_mtime_ns == initial_mtime


@pytest.mark.parametrize("change", ["corrupt", "url", "missing"])
def test_plugin_download_refreshes_invalid_cache(tmp_path, monkeypatch, change):
    destination = tmp_path / "sample.tar.gz"
    destination.write_bytes(archive_bytes(b"return true"))
    url = "https://example.invalid/sample/archive/commit.tar.gz"
    cached = {"url": url, "sha256": nvim_plugins.sha256(destination)}
    if change == "corrupt":
        destination.write_bytes(b"corrupt")
    elif change == "url":
        url = "https://example.invalid/sample/archive/new-commit.tar.gz"
    else:
        destination.unlink()
    replacement = archive_bytes(b"return false")
    requests = []

    def urlopen(request, timeout):
        requests.append((request.full_url, timeout))
        return io.BytesIO(replacement)

    monkeypatch.setattr(nvim_plugins.urllib.request, "urlopen", urlopen)
    result = nvim_plugins.download(url, destination, cached)
    assert requests == [(url, 120)]
    assert destination.read_bytes() == replacement
    assert result == {"url": url, "sha256": nvim_plugins.sha256(destination)}
    assert list(tmp_path.glob("sample.tar.gz.tmp.*")) == []


@pytest.mark.parametrize("action", ["download", "extract"])
def test_merged_binaries_remote_forwards_plugin_options(monkeypatch, action):
    calls = []
    monkeypatch.setattr(
        binaries,
        "run_remote_cli",
        lambda host, args: calls.append((host, args)) or 7,
    )
    options = (
        ["--lock", "lock file.json", "--repos", "repo file.json"]
        if action == "download"
        else []
    )
    args = build_parser().parse_args(
        [
            "binaries",
            action,
            "--host",
            "suzhou-one",
            "--jobs",
            "2",
            *options,
            "--output",
            "plugin archives",
        ]
    )
    assert args.handler(args) == 7
    assert calls == [
        (
            "suzhou-one",
            [
                "binaries",
                action,
                "--jobs",
                "2",
                *options,
                "--output",
                "plugin archives",
            ],
        )
    ]


def test_removed_nvim_plugins_command_is_rejected():
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["nvim-plugins", "download"])
    assert error.value.code == 2


def test_binaries_download_runs_tools_and_plugins(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        binaries.binaries_download,
        "download",
        lambda output, jobs: calls.append(("tools", output, jobs)) or 0,
    )
    monkeypatch.setattr(
        binaries.nvim_plugins,
        "download_plugins",
        lambda args: calls.append(("plugins", args.output, args.jobs)),
    )
    output = tmp_path / "plugin archives"
    args = build_parser().parse_args(
        ["binaries", "download", "--jobs", "2", "--output", str(output)]
    )
    assert args.handler(args) == 0
    assert calls == [
        ("tools", binaries.PLATFORM_DIR, 2),
        ("plugins", output, 2),
    ]


@pytest.mark.parametrize("option", ["--lock", "--repos"])
def test_binaries_extract_rejects_download_only_options(option):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["binaries", "extract", option, "unused.json"])
    assert error.value.code == 2


def test_binaries_download_cancellation_does_not_start_plugins(monkeypatch):
    monkeypatch.setattr(binaries.binaries_download, "download", lambda *args: 130)

    def unexpected_plugins(args):
        pytest.fail("cancellation must not start plugin downloads")

    monkeypatch.setattr(binaries.nvim_plugins, "download_plugins", unexpected_plugins)
    args = build_parser().parse_args(["binaries", "download"])
    assert args.handler(args) == 130
