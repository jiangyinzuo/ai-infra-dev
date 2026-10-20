"""Local model selection and fail-closed accelerator occupancy checks.

NPU IDs remain global ASCEND runtime IDs from ``npu-smi info -m``; an idle
card requires all sibling chips to be idle. NVIDIA indices are nvidia-smi
indices, never CUDA ordinals; selected GPUs are returned as full UUIDs.
Visibility masks restrict selection without renumbering it. Numeric CUDA masks
require an existing CUDA_DEVICE_ORDER=PCI_BUS_ID; otherwise use UUID masks.
MIG is unsupported.
Occupancy is a snapshot, not a reservation; process visibility can be container-limited.
GPU idle means no compute processes and zero utilization, not zero used memory:
display/driver allocations can remain and reduce the memory available to vLLM.
Unknown devices, ambiguous platforms and incomplete telemetry are errors.
"""

import csv
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]


def _load_summary():
    path = ROOT / "ssh/npu-top/npu_smi_summary.py"
    spec = importlib.util.spec_from_file_location("serve_npu_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _discover_models() -> list[str]:
    # models imports ssh_common; use the same repository paths as the SSH CLI.
    for path in (ROOT / "ssh", ROOT / "ssh/container-dev/devtools"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    from models import discover_models

    return discover_models()


def _select(options: list[str], prompt: str, *, multiple: bool = False) -> list[str]:
    if not options:
        raise RuntimeError(f"No candidates available: {prompt}")
    fzf = Path.home() / "plugged/fzf/bin/fzf"
    if (
        sys.stdin.isatty()
        and sys.stdout.isatty()
        and fzf.is_file()
        and os.access(fzf, os.X_OK)
    ):
        command = [str(fzf), "--prompt", prompt + ": "]
        if multiple:
            command.append("--multi")
        try:
            result = subprocess.run(
                command,
                input="\n".join(options) + "\n",
                text=True,
                stdout=subprocess.PIPE,
                check=False,
            )
        except OSError as exc:
            raise RuntimeError(f"Cannot execute fzf: {exc}") from exc
        if result.returncode in (1, 130):
            raise EOFError("Selection cancelled")
        if result.returncode:
            raise RuntimeError(f"fzf failed with exit code {result.returncode}")
        selected = result.stdout.splitlines()
        if not selected:
            raise EOFError("Selection cancelled")
        if any(value not in options for value in selected) or (
            not multiple and len(selected) != 1
        ):
            raise RuntimeError("fzf returned an invalid selection")
        return list(dict.fromkeys(selected))
    print(
        "Using numbered selection (fzf missing/not executable or terminal is not a TTY).",
        file=sys.stderr,
    )
    for index, value in enumerate(options, 1):
        print(f"{index}. {value}", file=sys.stderr)
    while True:
        answer = input(
            prompt
            + (
                " [comma-separated numbers, q cancels]: "
                if multiple
                else " [number, q cancels]: "
            )
        ).strip()
        if not answer or answer.lower() in {"q", "quit"}:
            raise EOFError("Selection cancelled")
        parts = answer.split(",")
        if (multiple or len(parts) == 1) and all(
            part.strip().isdigit() for part in parts
        ):
            indices = [int(part.strip()) - 1 for part in parts]
            if all(0 <= index < len(options) for index in indices):
                return list(dict.fromkeys(options[index] for index in indices))
        print("Invalid selection; choose listed numbers only.", file=sys.stderr)


def choose_model(model: str | None) -> str:
    """Return an absolute local directory; remote HF IDs are intentionally rejected."""
    if model is None:
        model = _select(_discover_models(), "Model")[0]
    path = Path(model).expanduser()
    if not path.is_dir():
        raise ValueError(
            f"Model must be an existing local directory (HF IDs are unsupported): {model}"
        )
    return str(path.resolve())


def _run_smi(*arguments: str) -> str:
    try:
        return subprocess.run(
            ["npu-smi", "info", *arguments],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Cannot obtain npu-smi metrics/mapping: {exc}") from exc


def _parse_mapping(text: str) -> dict[int, tuple[int, int]]:
    """Parse documented -m columns; skip non-compute chips marked '-'."""
    lines = [
        line.strip().strip("|").strip() for line in text.splitlines() if line.strip()
    ]
    header_index = next(
        (
            i
            for i, line in enumerate(lines)
            if "Chip Logic ID" in line and "NPU ID" in line
        ),
        None,
    )
    if header_index is None:
        raise RuntimeError("npu-smi -m must report NPU ID, Chip ID and Chip Logic ID")
    header = [
        part.strip() for part in re.split(r"\s{2,}|\s*\|\s*", lines[header_index])
    ]
    try:
        card_index, chip_index, logic_index = [
            header.index(name) for name in ("NPU ID", "Chip ID", "Chip Logic ID")
        ]
    except ValueError as exc:
        raise RuntimeError("Unrecognized npu-smi -m header") from exc
    mapping: dict[int, tuple[int, int]] = {}
    keys: set[tuple[int, int]] = set()
    for line in lines[header_index + 1 :]:
        parts = line.replace("|", " ").split()
        if not parts or not parts[0].isdigit():
            continue
        if len(parts) <= max(card_index, chip_index, logic_index):
            raise RuntimeError("Incomplete npu-smi mapping row")
        logic = parts[logic_index]
        if logic == "-":
            continue
        if not all(
            parts[index].isdigit() for index in (card_index, chip_index, logic_index)
        ):
            raise RuntimeError("Invalid npu-smi mapping row")
        logical = int(logic)
        key = (int(parts[card_index]), int(parts[chip_index]))
        if logical in mapping or key in keys:
            raise RuntimeError("Ambiguous npu-smi mapping")
        mapping[logical] = key
        keys.add(key)
    if not mapping:
        raise RuntimeError("No runtime devices in npu-smi mapping")
    return mapping


def _ids(value: str, label: str) -> list[int]:
    parts = value.split(",")
    if not parts or any(not re.fullmatch(r"[0-9]+", part.strip()) for part in parts):
        raise ValueError(
            f"{label} must be comma-separated nonnegative global runtime IDs"
        )
    ids = [int(part.strip()) for part in parts]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} contains duplicate IDs")
    return ids


def _choose_npu_devices(devices: str | None) -> str:
    """Select only fully reported idle cards, returning global runtime device IDs.

    A busy or unreported sibling chip makes the entire card ineligible. No sudo,
    guessed card-to-device arithmetic, or fallback on missing metrics is used.
    """
    mapping = _parse_mapping(_run_smi("-m"))
    summary = _load_summary()
    try:
        rows = summary.parse_summary(_run_smi())
    except ValueError as exc:
        raise RuntimeError(f"Cannot parse NPU occupancy: {exc}") from exc
    reported = {(row["npu"], row["chip"]): row for row in rows}
    if len(reported) != len(rows):
        raise RuntimeError("Duplicate NPU occupancy rows")
    # Without chip-level metrics there is no evidence for mapping individual dies.
    if any(key not in reported for key in mapping.values()):
        raise RuntimeError("NPU metrics do not cover every mapped card/chip")
    idle = set(summary.idle_cards(rows))
    visible = list(mapping)
    mask = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if mask is not None:
        visible = _ids(mask, "ASCEND_RT_VISIBLE_DEVICES")
        if any(device not in mapping for device in visible):
            raise RuntimeError(
                "ASCEND_RT_VISIBLE_DEVICES cannot be mapped to npu-smi global IDs"
            )
    available = {device for device in visible if mapping[device][0] in idle}
    for device in visible:
        card, chip = mapping[device]
        row = reported[(card, chip)]
        state = "idle/selectable" if device in available else "busy/unavailable"
        print(f"Runtime={device} {summary.format_row(row)} [{state}]", file=sys.stderr)
    if devices is None:
        choices = [
            f"{device}: card={mapping[device][0]} chip={mapping[device][1]} idle"
            for device in visible
            if device in available
        ]
        selected = _select(choices, "Idle devices", multiple=True)
        chosen = [int(value.split(":", 1)[0]) for value in selected]
    else:
        chosen = _ids(devices, "--devices")
    if any(device not in available for device in chosen):
        raise ValueError(
            "Selected devices are busy, unavailable, unknown, or outside ASCEND_RT_VISIBLE_DEVICES"
        )
    print("已选择 NPU devices: " + ",".join(map(str, chosen)), file=sys.stderr)
    return ",".join(map(str, chosen))


def device_visibility_variable() -> str:
    """Detect one installed accelerator CLI without importing torch or probing peers.

    Having both CLIs is ambiguous; failure of the selected CLI is never grounds
    for trying the other platform, since that could launch on unintended devices.
    """
    nvidia = shutil.which("nvidia-smi") is not None
    ascend = shutil.which("npu-smi") is not None
    if nvidia and ascend:
        raise RuntimeError(
            "Both nvidia-smi and npu-smi are available; accelerator platform is ambiguous"
        )
    if nvidia:
        return "CUDA_VISIBLE_DEVICES"
    if ascend:
        return "ASCEND_RT_VISIBLE_DEVICES"
    raise RuntimeError("No supported accelerator CLI found (nvidia-smi or npu-smi)")


def _run_nvidia_smi(query: str) -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Cannot obtain nvidia-smi telemetry: {exc}") from exc


class _Gpu(NamedTuple):
    index: str
    uuid: str
    name: str
    total: int
    used: int
    utilization: int
    bus: tuple[int, int, int, int]


def _gpu_metrics() -> tuple[list[_Gpu], set[str]]:
    # MIG 模式也必须查询；仅看父 GPU 的利用率无法证明 MIG 实例空闲。
    text = _run_nvidia_smi(
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,mig.mode.current,pci.bus_id"
    )
    gpus: list[_Gpu] = []
    indices: set[str] = set()
    uuids: set[str] = set()
    for row in csv.reader(text.splitlines()):
        if len(row) != 8:
            raise RuntimeError("Incomplete nvidia-smi GPU telemetry")
        index, uuid, name, total, used, utilization, mig, bus = [
            part.strip() for part in row
        ]
        if mig == "Enabled" or uuid.startswith("MIG-"):
            raise RuntimeError("MIG devices are unsupported by idle GPU selection")
        if mig not in {"Disabled", "N/A", "[N/A]"}:
            raise RuntimeError("Unknown nvidia-smi MIG telemetry")
        bus_match = re.fullmatch(
            r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", bus
        )
        if (
            not index.isascii()
            or not index.isdecimal()
            or not re.fullmatch(
                r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid
            )
            or not name
            or name in {"N/A", "[N/A]"}
            or any(
                not re.fullmatch(r"[0-9]+", value)
                for value in (total, used, utilization)
            )
            or bus_match is None
        ):
            raise RuntimeError("Invalid or unavailable nvidia-smi GPU telemetry")
        gpu = _Gpu(
            str(int(index)), uuid, name, int(total), int(used), int(utilization),
            tuple(int(part, 16) for part in bus_match.groups()),
        )
        if gpu.total <= 0 or gpu.used > gpu.total or gpu.utilization > 100:
            raise RuntimeError("Out-of-range nvidia-smi GPU telemetry")
        if gpu.index in indices or gpu.uuid in uuids:
            raise RuntimeError("Duplicate nvidia-smi GPU index or UUID")
        indices.add(gpu.index)
        uuids.add(gpu.uuid)
        gpus.append(gpu)
    if not gpus:
        raise RuntimeError("No GPUs reported by nvidia-smi")
    processes = _run_nvidia_smi("--query-compute-apps=gpu_uuid")
    busy: set[str] = set()
    for row in csv.reader(processes.splitlines()):
        if len(row) != 1 or row[0].strip() not in uuids:
            raise RuntimeError(
                "Unknown GPU or unavailable nvidia-smi compute-process telemetry "
                "(MIG unsupported)"
            )
        busy.add(row[0].strip())
    return gpus, busy


def _gpu_ids(value: str, gpus: list[_Gpu], label: str) -> list[str]:
    """Resolve global indices or unambiguous UUID prefixes, preserving order."""
    result: list[str] = []
    for part in value.split(","):
        token = part.strip()
        if token.startswith("MIG-"):
            raise ValueError(f"{label}: MIG devices are unsupported")
        if re.fullmatch(r"[0-9]+", token):
            matches = [gpu.uuid for gpu in gpus if gpu.index == str(int(token))]
        elif token.startswith("GPU-") and len(token) > 4:
            matches = [gpu.uuid for gpu in gpus if gpu.uuid.startswith(token)]
        else:
            matches = []
        if len(matches) != 1:
            raise ValueError(f"{label}: unknown or ambiguous GPU index/UUID: {token!r}")
        if matches[0] in result:
            raise ValueError(f"{label}: duplicate GPU: {token}")
        result.append(matches[0])
    return result


def _gpu_visibility(mask: str | None, gpus: list[_Gpu]) -> list[str]:
    if mask is None:
        return [gpu.uuid for gpu in gpus]
    if mask.strip() in {"", "-1"}:
        return []
    tokens = [part.strip() for part in mask.split(",")]
    if any(re.fullmatch(r"[0-9]+", token) for token in tokens):
        # CUDA 默认 FASTEST_FIRST 顺序并不等于 nvidia-smi index；不能猜测映射。
        if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
            raise ValueError(
                "Numeric CUDA_VISIBLE_DEVICES requires CUDA_DEVICE_ORDER=PCI_BUS_ID; "
                "use GPU UUIDs to avoid ambiguous CUDA ordinals"
            )
        ordered = sorted(gpus, key=lambda gpu: gpu.bus)
        if len({gpu.bus for gpu in gpus}) != len(gpus):
            raise RuntimeError("Ambiguous GPU PCI bus IDs")
        for index, token in enumerate(tokens):
            if re.fullmatch(r"[0-9]+", token):
                ordinal = int(token)
                if ordinal >= len(ordered):
                    raise ValueError(
                        "CUDA_VISIBLE_DEVICES contains an unknown CUDA ordinal"
                    )
                tokens[index] = ordered[ordinal].uuid
    return _gpu_ids(",".join(tokens), gpus, "CUDA_VISIBLE_DEVICES")


def _choose_gpu_devices(devices: str | None) -> str:
    gpus, busy = _gpu_metrics()
    visible = _gpu_visibility(os.environ.get("CUDA_VISIBLE_DEVICES"), gpus)
    by_uuid = {gpu.uuid: gpu for gpu in gpus}
    available = {
        uuid for uuid in visible
        if uuid not in busy and by_uuid[uuid].utilization == 0
    }
    print(
        "GPU idle = no compute processes and 0% utilization; residual display/driver "
        "memory remains occupied. This snapshot does not reserve devices.",
        file=sys.stderr,
    )
    choices: list[str] = []
    for uuid in visible:
        gpu = by_uuid[uuid]
        state = "idle/selectable" if uuid in available else "busy/unavailable"
        description = (
            f"{uuid}: index={gpu.index} {gpu.name} "
            f"memory={gpu.used}/{gpu.total} MiB "
            f"utilization={gpu.utilization}% [{state}]"
        )
        print(description, file=sys.stderr)
        if uuid in available:
            choices.append(description)
    chosen = (
        [
            value.split(":", 1)[0]
            for value in _select(choices, "Idle GPUs", multiple=True)
        ]
        if devices is None else _gpu_ids(devices, gpus, "--devices")
    )
    if any(uuid not in available for uuid in chosen):
        raise ValueError("Selected GPUs are busy or outside CUDA_VISIBLE_DEVICES")
    print("已选择 GPU devices: " + ",".join(chosen), file=sys.stderr)
    return ",".join(chosen)


def choose_devices(devices: str | None) -> str:
    """Select idle visible devices; return full GPU UUIDs or global Ascend IDs."""
    if device_visibility_variable() == "CUDA_VISIBLE_DEVICES":
        return _choose_gpu_devices(devices)
    return _choose_npu_devices(devices)
