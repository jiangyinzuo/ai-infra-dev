import os
from pathlib import Path
import shlex
import subprocess

import pytest


SSH = Path(__file__).resolve().parents[2]
COMPLETION = SSH / "completion.bash"


def complete(tmp_path, words, *, setup="", cwd=None):
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True, exist_ok=True)
    (home / ".ssh/config").write_text("Host server-one server-two\nHost wildcard-*\n")
    script = f"""
{setup}
source {shlex.quote(str(COMPLETION))}
_ssh_usg_config=/nonexistent
COMP_WORDS=({" ".join(shlex.quote(word) for word in words)})
COMP_CWORD=$((${{#COMP_WORDS[@]}} - 1))
COMP_LINE="${{COMP_WORDS[*]}}"
COMP_POINT=${{#COMP_LINE}}
if [[ ${{COMP_WORDS[0]##*/}} == python3 ]]; then
    _cli_python_complete python3 "${{COMP_WORDS[COMP_CWORD]}}" "${{COMP_WORDS[COMP_CWORD-1]}}"
else
    _cli_complete
fi
printf '%s\\n' "${{COMPREPLY[@]}}"
"""
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", script],
        cwd=cwd or tmp_path,
        env={**os.environ, "HOME": str(home)},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


@pytest.mark.parametrize(
    "prefix",
    [["devtools/cli.py"], ["python3", "/some directory/devtools/cli.py"]],
)
@pytest.mark.parametrize(
    "args,expected",
    [
        ([""], {"nvim-plugins", "git-repo", "binaries", "containers"}),
        (
            ["--no-hints", ""],
            {"nvim-plugins", "git-repo", "binaries", "containers"},
        ),
        (["nvim-plugins", ""], {"download", "extract"}),
        (["binaries", ""], {"download", "extract", "sync"}),
        (["containers", ""], {"create", "delete", "download", "ls"}),
        (
            ["containers", "create", "--"],
            {"--host", "--image", "--name", "--share", "--test-home", "--no-bindfs"},
        ),
        (
            ["containers", "delete", "--"],
            {
                "--host",
                "--name",
                "--keep-bindfs",
                "--remove-bindfs",
                "--cleanup-bindfs",
                "--yes",
            },
        ),
        (
            ["containers", "download", "--"],
            {"--host", "--date", "--version", "--npu-platform"},
        ),
        (["containers", "ls", "--"], {"--host", "--json"}),
        (["git-repo", ""], {"update"}),
        (["git-repo", "update", "--"], {"--host", "--debug"}),
    ],
)
def test_commands_and_options(tmp_path, prefix, args, expected):
    replies = set(complete(tmp_path, prefix + args))
    assert expected <= replies
    if args in ([""], ["--no-hints", ""]):
        assert "sync" not in replies


@pytest.mark.parametrize(
    "args",
    [
        ["nvim-plugins", "download"],
        ["nvim-plugins", "extract"],
        ["binaries", "download"],
        ["binaries", "extract"],
        ["binaries", "sync", "--host", "server-one"],
        ["git-repo", "update"],
        ["containers", "create"],
        ["containers", "delete"],
        ["containers", "download"],
        ["containers", "ls"],
    ],
)
def test_hosts_and_repeatable_sync_host(tmp_path, args):
    assert complete(tmp_path, ["devtools/cli.py", *args, "--host", "server-"]) == [
        "server-one",
        "server-two",
    ]
    assert "--host" in complete(tmp_path, ["devtools/cli.py", *args, "--"])


def test_platform_and_localhost(tmp_path):
    assert complete(
        tmp_path, ["devtools/cli.py", "containers", "download", "--npu-platform", ""]
    ) == ["a3"]
    assert complete(
        tmp_path, ["devtools/cli.py", "containers", "download", "--host", "local"]
    ) == ["localhost"]


@pytest.mark.parametrize("python", [False, True])
@pytest.mark.parametrize(
    "directory,option", [("createuser", "--workers"), ("npu-top", "--inventory")]
)
def test_profile_completions_preserved(tmp_path, python, directory, option):
    words = [f"{directory}/cli.py", "--"]
    if python:
        words.insert(0, "python3")
    assert option in complete(tmp_path, words)


def test_bare_cli_in_devtools(tmp_path):
    directory = tmp_path / "devtools"
    directory.mkdir()
    assert "binaries" in complete(tmp_path, ["cli.py", ""], cwd=directory)
    assert "binaries" in complete(tmp_path, ["python3", "./cli.py", ""], cwd=directory)


def test_other_python_completion_preserved(tmp_path):
    setup = "_original_python() { COMPREPLY=(original-result); }; complete -o nospace -F _original_python python3"
    assert complete(tmp_path, ["python3", "other.py", ""], setup=setup) == [
        "original-result"
    ]
    assert "binaries" in complete(
        tmp_path, ["python3", "devtools/cli.py", ""], setup=setup
    )


def test_python_word_completion_preserved(tmp_path):
    assert complete(
        tmp_path,
        ["python3", "other.py", "old"],
        setup="complete -W 'old-value other-value' python3",
    ) == ["old-value"]


def test_python_file_completion_preserved(tmp_path):
    (tmp_path / "other script.py").touch()
    assert complete(tmp_path, ["python3", "other"]) == ["other script.py"]


def test_path_values_preserve_spaces(tmp_path):
    (tmp_path / "plugin lock.json").touch()
    assert complete(
        tmp_path, ["devtools/cli.py", "nvim-plugins", "download", "--lock", "plugin"]
    ) == ["plugin lock.json"]


def test_source_twice_preserves_python_completion(tmp_path):
    setup = f"_original_python() {{ COMPREPLY=(original-result); }}; complete -F _original_python python3; source {shlex.quote(str(COMPLETION))}"
    assert complete(tmp_path, ["python3", "other.py", ""], setup=setup) == [
        "original-result"
    ]
