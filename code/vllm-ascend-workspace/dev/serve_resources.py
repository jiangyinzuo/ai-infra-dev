"""Local serve resources, with fail-closed NPU occupancy checks.

Explicit models must be existing local directories, not Hugging Face repository IDs.
Device arguments and return values are global runtime IDs for
ASCEND_RT_VISIBLE_DEVICES, never card IDs or indices within an existing visibility
mask. ``npu-smi info -m`` supplies the card/chip/global logical ID mapping (including
both A3 dies); device properties do not expose that mapping. Existing visibility
masks restrict selection without renumbering it. Unknown mapping formats are errors.
Occupancy is a snapshot, not a reservation; process visibility can be container-limited.
"""

import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


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


def choose_devices(devices: str | None) -> str:
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
