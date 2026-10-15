import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
import npu_devices


def test_discovers_only_supported_character_devices(tmp_path):
    # 使用真实字符设备的符号链接，避免创建设备节点或需要 root。
    for name in (
        "davinci0",
        "davinci17",
        "davinci_manager",
        "devmm_svm",
        "hisi_hdc",
        "davinci_bad",
        "unrelated",
    ):
        (tmp_path / name).symlink_to("/dev/null")
    (tmp_path / "davinci99").write_text("not a character device")
    devices = npu_devices.discover_devices(tmp_path)
    assert {Path(device["path"]).name for device in devices} == {
        "davinci0",
        "davinci17",
        "davinci_manager",
        "devmm_svm",
        "hisi_hdc",
    }
    assert {device["gid"] for device in devices} == {os.stat("/dev/null").st_gid}


def test_group_ids_deduplicate_and_reuse_existing_names(monkeypatch):
    def lookup(gid):
        if gid == 4321:
            return SimpleNamespace(gr_name="driver-group")
        raise KeyError(gid)

    monkeypatch.setattr(npu_devices.grp, "getgrgid", lookup)
    assert npu_devices.device_groups([{"gid": 4321}, {"gid": 7654}, {"gid": 4321}]) == [
        {"name": "driver-group", "gid": 4321},
        {"name": "npu-device-7654", "gid": 7654},
    ]
    assert npu_devices.device_groups([]) == []
