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
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    monkeypatch.setattr(module.shutil, "which", lambda name: "/bin/npu-smi" if name == "npu-smi" else None)
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


def test_a3_mapping_never_equates_card_and_runtime_ids(resources):
    assert resources.choose_devices("3,6") == "3,6"
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
    assert all(command[0] == "npu-smi" for command in commands)
    assert len(commands) == 1


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


GPU_A = "GPU-aaaaaaaa-1111-2222-3333-444444444444"
GPU_B = "GPU-bbbbbbbb-1111-2222-3333-444444444444"


def gpu_metrics(utilization=0):
    # Index and PCI ordering deliberately disagree: CUDA PCI ordinal 0 is index 7.
    return (
        f"2, {GPU_A}, RTX PRO 2000, 16384, 512, {utilization}, [N/A], 00000000:04:00.0\n"
        f"7, {GPU_B}, RTX PRO 2000, 16384, 128, 0, [N/A], 00000000:01:00.0\n"
    )


@pytest.fixture
def gpu_resources(resources, monkeypatch):
    monkeypatch.setattr(resources.shutil, "which", lambda name: "/bin/nvidia-smi" if name == "nvidia-smi" else None)

    def run(command, **kwargs):
        assert command[0] == "nvidia-smi", "GPU selection must not invoke npu-smi"
        query = command[1]
        if query.startswith("--query-gpu="):
            return SimpleNamespace(stdout=gpu_metrics())
        assert query == "--query-compute-apps=gpu_uuid"
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(resources.subprocess, "run", run)
    return resources


def test_gpu_selects_full_uuids_with_residual_memory(gpu_resources):
    assert gpu_resources.device_visibility_variable() == "CUDA_VISIBLE_DEVICES"
    assert gpu_resources.choose_devices("7,2") == f"{GPU_B},{GPU_A}"


def test_gpu_compute_process_blocks_selection(gpu_resources, monkeypatch):
    monkeypatch.setattr(
        gpu_resources, "_run_nvidia_smi",
        lambda query: gpu_metrics() if query.startswith("--query-gpu=") else GPU_A + "\n",
    )
    with pytest.raises(ValueError):
        gpu_resources.choose_devices("2")
    assert gpu_resources.choose_devices("7") == GPU_B


def test_gpu_utilization_blocks_without_processes(gpu_resources, monkeypatch):
    monkeypatch.setattr(
        gpu_resources, "_run_nvidia_smi",
        lambda query: gpu_metrics(1) if query.startswith("--query-gpu=") else "",
    )
    with pytest.raises(ValueError):
        gpu_resources.choose_devices("2")
    assert gpu_resources.choose_devices("7") == GPU_B


def test_gpu_uuid_visibility_and_selection_order(gpu_resources, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{GPU_B},GPU-aaaaaaaa")
    monkeypatch.setattr(gpu_resources.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr("builtins.input", lambda prompt: "1,2")
    assert gpu_resources.choose_devices(None) == f"{GPU_B},{GPU_A}"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_B)
    with pytest.raises(ValueError):
        gpu_resources.choose_devices("2")


def test_gpu_numeric_mask_uses_pci_ordinals_not_smi_indices(gpu_resources, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    with pytest.raises(ValueError):
        gpu_resources.choose_devices("7")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setattr(gpu_resources.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr("builtins.input", lambda prompt: "1,2")
    assert gpu_resources.choose_devices(None) == f"{GPU_A},{GPU_B}"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert gpu_resources.choose_devices("7") == GPU_B
    with pytest.raises(ValueError):
        gpu_resources.choose_devices("2")


@pytest.mark.parametrize("mask", ["", "-1"])
def test_gpu_disabled_visibility(gpu_resources, monkeypatch, mask):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    with pytest.raises(ValueError):
        gpu_resources.choose_devices("2")
    with pytest.raises(RuntimeError):
        gpu_resources.choose_devices(None)


@pytest.mark.parametrize("value", ["99", "GPU-unknown", "GPU-", "2,2", "2,GPU-aaaaaaaa", "MIG-123", "2,", "-1"])
def test_gpu_invalid_requests(gpu_resources, value):
    with pytest.raises(ValueError):
        gpu_resources.choose_devices(value)


def test_gpu_ambiguous_uuid_prefix_rejected(gpu_resources, monkeypatch):
    other = GPU_A.replace("444444444444", "555555555555")
    monkeypatch.setattr(
        gpu_resources, "_run_nvidia_smi",
        lambda query: gpu_metrics().replace(GPU_B, other) if query.startswith("--query-gpu=") else "",
    )
    with pytest.raises(ValueError):
        gpu_resources.choose_devices("GPU-aaaaaaaa")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaaaaaaa")
    with pytest.raises(ValueError):
        gpu_resources.choose_devices("2")


@pytest.mark.parametrize(
    "telemetry",
    [
        "", "2, GPU-unknown, RTX\n",
        gpu_metrics().replace("16384, 512", "N/A, 512"),
        gpu_metrics().replace("16384, 512", "16384, 17000"),
        gpu_metrics().replace("512, 0", "512, 101"),
        gpu_metrics().replace("[N/A]", "Enabled"),
        gpu_metrics().replace("[N/A]", "Unknown"),
        gpu_metrics().replace("00000000:04:00.0", "unknown"),
        gpu_metrics().replace(GPU_B, GPU_A),
        gpu_metrics().replace("7, GPU-", "2, GPU-"),
    ],
)
def test_gpu_bad_telemetry_fails_closed(gpu_resources, monkeypatch, telemetry):
    monkeypatch.setattr(
        gpu_resources, "_run_nvidia_smi",
        lambda query: telemetry if query.startswith("--query-gpu=") else "",
    )
    with pytest.raises(RuntimeError):
        gpu_resources.choose_devices("2")


@pytest.mark.parametrize("processes", ["N/A\n", "GPU-unknown\n", "MIG-123\n", "\n", f"{GPU_A}, unexpected\n"])
def test_gpu_bad_process_telemetry_fails_closed(gpu_resources, monkeypatch, processes):
    monkeypatch.setattr(
        gpu_resources, "_run_nvidia_smi",
        lambda query: gpu_metrics() if query.startswith("--query-gpu=") else processes,
    )
    with pytest.raises(RuntimeError):
        gpu_resources.choose_devices("2")


@pytest.mark.parametrize("fail_process_query", [False, True])
def test_nvidia_failure_never_falls_back(gpu_resources, monkeypatch, fail_process_query):
    commands = []

    def fail(command, **kwargs):
        commands.append(command)
        if fail_process_query and command[1].startswith("--query-gpu="):
            return SimpleNamespace(stdout=gpu_metrics())
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(gpu_resources.subprocess, "run", fail)
    with pytest.raises(RuntimeError):
        gpu_resources.choose_devices("2")
    assert len(commands) == (2 if fail_process_query else 1)
    assert all(command[0] == "nvidia-smi" for command in commands)


@pytest.mark.parametrize("tools", [set(), {"nvidia-smi", "npu-smi"}])
def test_missing_or_ambiguous_platform_rejected(resources, monkeypatch, tools):
    monkeypatch.setattr(resources.shutil, "which", lambda name: "/bin/" + name if name in tools else None)
    monkeypatch.setattr(resources.subprocess, "run", lambda *a, **kw: pytest.fail("ambiguous platform queried"))
    with pytest.raises(RuntimeError):
        resources.choose_devices("2")
