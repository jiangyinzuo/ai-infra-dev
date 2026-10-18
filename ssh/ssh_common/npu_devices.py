"""发现昇腾字符设备及其设备组；不修改权限。"""

import grp
import json
from pathlib import Path
import re
import stat


def discover_devices(directory=Path("/dev")):
    devices = []
    for path in sorted(directory.iterdir()):
        if not re.fullmatch(
            r"davinci(?:[0-9]+|_manager)|devmm_svm|hisi_hdc", path.name
        ):
            continue
        info = path.stat()
        if stat.S_ISCHR(info.st_mode):
            devices.append({"path": str(path), "gid": info.st_gid})
    return devices


def device_groups(devices):
    groups = []
    for gid in sorted({device["gid"] for device in devices}):
        try:
            name = grp.getgrgid(gid).gr_name
        except KeyError:
            name = f"npu-device-{gid}"
        groups.append({"name": name, "gid": gid})
    return groups


if __name__ == "__main__":
    print(json.dumps(device_groups(discover_devices())))
