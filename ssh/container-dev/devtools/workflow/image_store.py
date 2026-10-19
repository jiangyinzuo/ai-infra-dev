"""Shared server-side image presence, archive reuse and Docker import."""

import json
from pathlib import Path
import re
import shlex
import subprocess
import sys


def ensure_image(directory, image, emit=None):
    directory = Path(directory).expanduser()
    if emit is None:
        emit = lambda command: print(
            json.dumps({"command": shlex.join(command)}), flush=True
        )

    def present(reference):
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return reference in result.stdout.splitlines()

    if present(image):
        return "image"
    repository, tag = image.rsplit(":", 1)
    if repository == "quay.io/ascend/vllm-ascend":
        candidates = [directory / (tag + ".tar.gz")]
        dated = re.fullmatch(r"(nightly-.+)-(\d{8})", tag)
        original = repository + ":" + dated[1] if dated else image
    else:
        candidates = [
            directory / ("vllm-ascend" + separator + tag + ".tar.gz")
            for separator in (":", "_")
        ]
        original = image
    archive = next((path for path in candidates if path.is_file()), None)
    if archive is None:
        return "missing"
    command = ["docker", "load", "--input", str(archive)]
    emit(command)
    subprocess.run(command, stdout=sys.stderr, check=True)
    if not present(original):
        raise RuntimeError("归档导入后未找到预期镜像：" + original)
    if original != image:
        command = ["docker", "tag", original, image]
        emit(command)
        subprocess.run(command, stdout=sys.stderr, check=True)
    if not present(image):
        raise RuntimeError("未找到预期镜像：" + image)
    return "loaded"


if __name__ == "__main__":
    print(json.dumps({"status": ensure_image(sys.argv[1], sys.argv[2])}), flush=True)
