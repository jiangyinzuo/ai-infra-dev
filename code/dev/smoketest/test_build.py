"""无网络、Docker 或实际 pip 安装依赖的构建入口测试。"""

import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def builder(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "__main__.py"
    spec = importlib.util.spec_from_file_location("workspace_build_smoke", path)
    module = importlib.util.module_from_spec(spec)
    # 入口为共享模块添加的导入路径不应泄漏给其他测试。
    with monkeypatch.context() as context:
        context.setattr(sys, "path", sys.path.copy())
        spec.loader.exec_module(module)
    (tmp_path / "dev").mkdir()
    monkeypatch.setattr(module, "__file__", str(tmp_path / "dev/__main__.py"))
    monkeypatch.setattr(module, "is_wsl", lambda: False)
    monkeypatch.setattr(module, "is_container", lambda: True)
    monkeypatch.setattr(module.sys, "prefix", sys.base_prefix)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    return module


@pytest.mark.parametrize("compile_ops", [False, True])
def test_editable_install_runs_in_project_and_preserves_failure(
    builder, monkeypatch, tmp_path, compile_ops
):
    project = tmp_path / "vllm-ascend-workspace/vllm-ascend"
    project.mkdir(parents=True)
    (project / "pyproject.toml").touch()
    # This executable accepts only the promised installation contract. A wrong
    # directory, option or compiler environment produces a different exit code.
    pip = tmp_path / "pip"
    pip.write_text(
        "#!/bin/sh\n"
        'test "$*" = "install -v -e . --no-deps --no-build-isolation" || exit 81\n'
        "test -f pyproject.toml || exit 82\n"
        'test "$VLLM_TARGET_DEVICE" = empty || exit 83\n'
        f'test "$COMPILE_CUSTOM_KERNELS" = {int(compile_ops)} || exit 84\n'
        "exit 17\n"
    )
    pip.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("COMPILE_CUSTOM_KERNELS", "opposite")
    monkeypatch.chdir(tmp_path.parent)
    args = ["vllm", "install"]
    if compile_ops:
        args.append("--compile-ops")
    assert builder.main(args) == 17


def test_devtools_install_does_not_require_source_checkout(
    builder, monkeypatch, tmp_path
):
    requirements = tmp_path / "dev/requirements-dev.txt"
    requirements.write_text("pyrefly>=1.3.0\npylint\n")
    pip = tmp_path / "pip"
    pip.write_text(
        "#!/bin/sh\n"
        'test "$#" = 3 && test "$1" = install && test "$2" = -r || exit 81\n'
        'test -f "$3" || exit 82\n'
        "exit 19\n"
    )
    pip.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.chdir(tmp_path.parent)
    assert builder.main(["devtools", "install"]) == 19


@pytest.mark.parametrize("resource", ["vllm", "devtools"])
def test_host_forbidden_before_install(builder, monkeypatch, resource):
    monkeypatch.setattr(builder, "is_container", lambda: False)
    with pytest.raises(ValueError):
        builder.main([resource, "install"])


@pytest.mark.parametrize("variable", ["VIRTUAL_ENV", "CONDA_PREFIX"])
def test_virtual_environment_forbidden(builder, monkeypatch, variable):
    monkeypatch.setenv(variable, "/some/environment")
    with pytest.raises(ValueError):
        builder.main(["devtools", "install"])


def test_invalid_existing_project_not_overwritten(builder, tmp_path):
    (tmp_path / "vllm-ascend-workspace/vllm-ascend").mkdir(parents=True)
    with pytest.raises(ValueError):
        builder.main(["vllm", "install"])


def test_missing_checkout_clones_selected_branch_before_install(
    builder, monkeypatch, tmp_path
):
    import subprocess
    import os

    origin = tmp_path / "origin"
    subprocess.run(
        ["git", "init", "-q", "--initial-branch=main", str(origin)], check=True
    )
    (origin / "setup.py").write_text("# fixture")
    subprocess.run(["git", "-C", str(origin), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(origin),
            "-c",
            "user.name=Smoke",
            "-c",
            "user.email=smoke@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    answers = iter([origin.as_uri(), ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    pip = tmp_path / "pip"
    pip.write_text("#!/bin/sh\ntest -f setup.py || exit 82\nexit 17\n")
    pip.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + ":" + os.environ["PATH"])
    assert builder.main(["vllm", "install"]) == 17
    checkout = tmp_path / "vllm-ascend-workspace/vllm-ascend"
    assert (
        subprocess.check_output(
            ["git", "-C", str(checkout), "remote", "get-url", "origin"], text=True
        ).strip()
        == origin.as_uri()
    )
    assert (
        subprocess.check_output(
            ["git", "-C", str(checkout), "branch", "--show-current"], text=True
        ).strip()
        == "main"
    )


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--host", "suzhou-one"],
        ["--container", "dev1"],
        ["--host", "suzhou-one", "--container", "-invalid"],
    ],
)
def test_wsl_requires_valid_remote_target(builder, monkeypatch, options):
    monkeypatch.setattr(builder, "is_wsl", lambda: True)
    with pytest.raises(SystemExit) as error:
        builder.main(["devtools", "install", *options])
    assert error.value.code == 2


def test_unsupported_python_rejected(builder, monkeypatch):
    monkeypatch.setattr(builder.sys, "version_info", (3, 10, 0))
    with pytest.raises(ValueError):
        builder.main(["devtools", "install"])


@pytest.mark.parametrize("operation", [["install"], ["serve", "stop"]])
def test_interactive_remote_uses_managed_user(
    builder, monkeypatch, tmp_path, operation
):
    from ssh_common import remote

    ssh = tmp_path / "ssh2"
    ssh.write_text(
        '#!/bin/sh\n[ "$1" = -t ] || exit 81\n[ "$2" = jiangyinzuo@suzhou-one ] || exit 82\ncase "$3" in *"docker exec --user root -it dev1"*) exit 0;; *) exit 83;; esac\n'
    )
    ssh.chmod(0o755)
    monkeypatch.setattr(remote, "SSH_DIR", tmp_path)
    monkeypatch.setattr(builder, "is_wsl", lambda: True)
    monkeypatch.setattr(builder, "checked_command", lambda host, command: command)
    assert (
        builder.main(
            ["vllm", *operation, "--host", "suzhou-one", "--container", "dev1"]
        )
        == 0
    )


@pytest.mark.parametrize("remove", [False, True])
def test_csrc_build_cleanup_is_opt_in_and_precedes_pip(
    builder, monkeypatch, tmp_path, remove
):
    import os

    project = tmp_path / "vllm-ascend-workspace/vllm-ascend"
    build = project / "csrc/build"
    build.mkdir(parents=True)
    (project / "setup.py").touch()
    (build / "CMakeCache.txt").write_text("/usr/local/python3.12.13/bin/ninja")
    cache = project / "csrc/build_cache"
    cache.mkdir()
    (cache / "keep").write_text("valid cache")
    pip = tmp_path / "pip"
    pip.write_text(
        "#!/bin/sh\n"
        + ("test ! -e csrc/build" if remove else "test -f csrc/build/CMakeCache.txt")
        + " || exit 81\nexit 17\n"
    )
    pip.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + ":" + os.environ["PATH"])
    arguments = ["vllm", "install"] + (["--remove-csrc-build"] if remove else [])
    assert builder.main(arguments) == 17
    assert build.exists() == (not remove)
    assert (cache / "keep").read_text() == "valid cache"


def test_remote_cleanup_flag_reaches_container(builder, monkeypatch):
    import shlex

    monkeypatch.setattr(builder, "is_wsl", lambda: True)
    monkeypatch.setattr(builder, "checked_command", lambda host, command: command)

    def ssh(host, command, **kwargs):
        container = shlex.split(command)
        assert "--remove-csrc-build" in shlex.split(container[-1])
        return 17

    monkeypatch.setattr(builder, "ssh_command", ssh)
    assert (
        builder.main(
            [
                "vllm",
                "install",
                "--remove-csrc-build",
                "--host",
                "suzhou-one",
                "--container",
                "dev1",
            ]
        )
        == 17
    )


def test_host_venv_can_stop_managed_service(builder, monkeypatch, tmp_path):
    import serve

    monkeypatch.setattr(builder, "is_container", lambda: False)
    monkeypatch.setenv("VIRTUAL_ENV", "/active-venv")
    monkeypatch.setattr(serve, "state_directory", lambda: tmp_path)
    assert builder.main(["vllm", "serve", "stop", "--no-hints"]) == 0
