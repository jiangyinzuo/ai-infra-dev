import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


MAPPING = """NPU ID  Chip ID  Chip Logic ID  Chip Phy-ID  Chip Name
8 0 2 10 Ascend
8 1 3 11 Ascend
8 2 - - Mcu
9 0 6 14 Ascend
9 1 7 15 Ascend
"""


def metrics(busy=False, processes=True):
    text = "| NPU Name | Health | Power |\n| Chip | Bus-Id | AICore(%) Memory-Usage(MB) HBM-Usage(MB) |\n"
    for card in (8, 9):
        text += f"| {card} Ascend | OK | 100 |\n"
        for chip in (0, 1):
            usage = 25 if busy and (card, chip) == (8, 1) else 0
            text += f"| {chip} | 0000:01:00.0 | {usage} 0 / 100 0 / 100 |\n"
    if processes:
        text += "| NPU Chip | Process id | Process name |\n"
    return text


@pytest.fixture
def resources(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "serve_resources.py"
    spec = importlib.util.spec_from_file_location("serve_resources_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        module, "_run_smi", lambda *args: MAPPING if args else metrics()
    )
    return module


def test_model_requires_existing_local_directory(resources, tmp_path, monkeypatch):
    monkeypatch.setattr(
        resources, "_discover_models", lambda: pytest.fail("explicit path scanned")
    )
    assert resources.choose_model(str(tmp_path)) == str(tmp_path.resolve())
    with pytest.raises(ValueError, match="local directory"):
        resources.choose_model("org/nonexistent-model")
    file = tmp_path / "file"
    file.touch()
    with pytest.raises(ValueError):
        resources.choose_model(str(file))


def test_discovery_uses_existing_one_level_scanner(resources, tmp_path, monkeypatch):
    root = tmp_path / "models"
    direct = root / "direct"
    nested = root / "other" / "nested"
    for path in (direct, nested):
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}")
        (path / "model.safetensors").touch()
    # Load the real discovery function, replacing only its default search roots.
    for path in (resources.ROOT / "ssh", resources.ROOT / "ssh/container-dev/devtools"):
        monkeypatch.syspath_prepend(str(path))
    import models

    monkeypatch.setattr(models, "DEFAULT_ROOTS", (str(root),))
    monkeypatch.setattr(models, "is_wsl", lambda: False)
    assert resources._discover_models() == [str(direct)]
    monkeypatch.setattr(resources.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr("builtins.input", lambda prompt: "1")
    assert resources.choose_model(None) == str(direct)


def test_a3_mapping_never_equates_card_and_runtime_ids(resources, capsys):
    assert resources.choose_devices("3,6") == "3,6"
    assert "Runtime=3 NPU=8 Chip=1" in capsys.readouterr().err
    with pytest.raises(ValueError):
        resources.choose_devices("8")


def test_busy_sibling_blocks_whole_card(resources, monkeypatch):
    monkeypatch.setattr(
        resources, "_run_smi", lambda *args: MAPPING if args else metrics(busy=True)
    )
    with pytest.raises(ValueError, match="busy"):
        resources.choose_devices("2")
    assert resources.choose_devices("6,7") == "6,7"


def test_visibility_mask_preserves_global_ids(resources, monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "7,3")
    assert resources.choose_devices("3,7") == "3,7"
    with pytest.raises(ValueError):
        resources.choose_devices("0")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "99")
    with pytest.raises(RuntimeError, match="cannot be mapped"):
        resources.choose_devices("99")


@pytest.mark.parametrize("value", ["", "-1", "1,", "2,2", "0-3", "x"])
def test_invalid_device_and_visibility_lists(resources, monkeypatch, value):
    with pytest.raises(ValueError):
        resources.choose_devices(value)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", value)
    with pytest.raises(ValueError):
        resources.choose_devices("2")


def test_missing_process_visibility_is_not_idle(resources, monkeypatch):
    monkeypatch.setattr(
        resources,
        "_run_smi",
        lambda *args: MAPPING if args else metrics(processes=False),
    )
    with pytest.raises(ValueError, match="busy"):
        resources.choose_devices("2")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "NPU ID  Chip ID\n8 0",
        MAPPING + "9 2 3 20 Ascend\n",
        MAPPING + "9 1 8 20 Ascend\n",
    ],
)
def test_missing_or_ambiguous_mapping_rejected(resources, text):
    with pytest.raises(RuntimeError):
        resources._parse_mapping(text)


def test_missing_chip_metrics_rejected(resources, monkeypatch):
    monkeypatch.setattr(
        resources,
        "_run_smi",
        lambda *args: (
            MAPPING
            if args
            else metrics().replace("| 1 | 0000:01:00.0 | 0 0 / 100 0 / 100 |\n", "")
        ),
    )
    with pytest.raises(RuntimeError, match="every mapped"):
        resources.choose_devices("2")


def test_smi_failure_does_not_use_sudo_or_fallback(resources, monkeypatch):
    commands = []

    def fail(command, **kwargs):
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(resources.subprocess, "run", fail)
    # Restore the real runner replaced by the fixture.
    path = Path(resources.__file__)
    spec = importlib.util.spec_from_file_location("serve_resources_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(RuntimeError, match="Cannot obtain"):
        module.choose_devices("2")
    assert commands == [["npu-smi", "info", "-m"]]


def test_numbered_selection_and_cancellation(resources, monkeypatch):
    monkeypatch.setattr(resources.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    answers = iter(["bad", "0", "2,4"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert resources.choose_devices(None) == "3,7"
    monkeypatch.setattr("builtins.input", lambda prompt: "q")
    with pytest.raises(EOFError):
        resources.choose_devices(None)


def test_fzf_uses_hardcoded_home_path(resources, monkeypatch, tmp_path):
    fzf = tmp_path / "plugged/fzf/bin/fzf"
    fzf.parent.mkdir(parents=True)
    fzf.touch(mode=0o755)
    monkeypatch.setattr(resources.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(resources.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(resources.sys, "stdout", SimpleNamespace(isatty=lambda: True))
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["input"] == "one\ntwo\n"
        return SimpleNamespace(returncode=0, stdout="two\n")

    monkeypatch.setattr(resources.subprocess, "run", run)
    assert resources._select(["one", "two"], "Model") == ["two"]
    assert commands[0][0] == str(fzf)
    monkeypatch.setattr(
        resources.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=130, stdout=""),
    )
    with pytest.raises(EOFError):
        resources._select(["one"], "Model")
