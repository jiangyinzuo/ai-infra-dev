import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("createuser", Path(__file__).resolve().parents[1] / "createuser.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_group_keys_overrides_and_disabled_keys(tmp_path):
    windows = tmp_path / "windows.pub"
    wsl = tmp_path / "wsl.pub"
    explicit = tmp_path / "explicit.pub"
    for path in (windows, wsl, explicit):
        path.write_text(f"ssh-ed25519 AAAA {path.stem}")
    config = tmp_path / "users.json"
    config.write_text(json.dumps({"default": {}, "explicit": {"public_keys": "explicit.pub"},
                                  "disabled": {"public_keys": []}}))
    targets = [{"host": "one", "group": "windows", "public_key": str(windows)},
               {"host": "two", "group": "wsl", "public_key": str(wsl)}]
    plan = MODULE.build_plan(targets, MODULE.load_users(config), tmp_path)
    for target, default in zip(plan, (windows, wsl)):
        users = {u["name"]: u for u in target["users"]}
        assert users["default"]["authorized_keys"] == default.read_text()
        assert users["explicit"]["authorized_keys"] == explicit.read_text()
        assert users["disabled"]["authorized_keys"] == ""
    summary = MODULE.describe_plan(plan, "plan")
    assert [u["public_key_count"] for u in summary["hosts"][0]["users"]] == [1, 1, 0]
