import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
from workflow import checkhealth


def test_plugin_summary_missing_and_empty_directories(tmp_path):
    lock = tmp_path / ".config/nvim/lazy-lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({"one": {}, "two": {}, "wezterm.nvim": {}}))
    for name in [".local/share/nvim/lazy/lazy.nvim", "plugged/one"]:
        path = tmp_path / name
        path.mkdir(parents=True)
        (path / "entry").touch()
    (tmp_path / "plugged/two").mkdir()
    text, ok = checkhealth.plugins(tmp_path)
    assert not ok and text.startswith("2/3 ")
    (tmp_path / "plugged/two/entry").touch()
    assert checkhealth.plugins(tmp_path)[1]


def test_git_tracked_changes_are_reported(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Smoke",
            "-c",
            "user.email=smoke@example.test",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "initial",
        ],
        check=True,
    )
    text, ok = checkhealth.repository(tmp_path)
    assert ok and "干净" in text
    path = tmp_path / "tracked"
    path.write_text("data")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked"], check=True)
    text, ok = checkhealth.repository(tmp_path)
    assert ok and "有已跟踪改动" in text
