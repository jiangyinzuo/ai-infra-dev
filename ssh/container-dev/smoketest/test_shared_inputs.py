import argparse
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ssh_common.arguments import add_jobs_argument
from ssh_common.inventory import read_machines, machine_record


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_invalid_concurrency_rejected(value):
    parser = argparse.ArgumentParser()
    add_jobs_argument(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--jobs", value])


def test_inventory_required_columns_and_missing_host(tmp_path):
    path = tmp_path / "machines.csv"
    path.write_text("Host,OS,Arch\na,openEuler,arm64\n")
    machines = read_machines(path, ("OS", "Arch"))
    assert machine_record(machines, "a", path)["Arch"] == "arm64"
    with pytest.raises(ValueError, match="missing machine record"):
        machine_record(machines, "absent", path)
    with pytest.raises(ValueError, match="NPUPlatform"):
        read_machines(path, ("NPUPlatform",))
