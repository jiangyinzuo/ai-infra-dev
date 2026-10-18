import io
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
from workflow.binaries_download import Downloader, Display


@pytest.mark.parametrize(
    "extra", [None, "bindfs-1.18.4/../escape", "/absolute", "other/file"]
)
def test_bindfs_root_directory_and_unsafe_paths(tmp_path, monkeypatch, extra):
    archive = tmp_path / "bindfs-1.18.4.tar.gz"
    archive.write_bytes(b"previous archive")
    downloader = Downloader(tmp_path, Display(["bindfs"]))
    monkeypatch.setattr(downloader, "metadata", lambda *a, **kw: "bindfs-1.18.4.tar.gz")

    def transfer(name, url, output):
        with tarfile.open(output, "w:gz") as tar:
            root = tarfile.TarInfo("bindfs-1.18.4/")
            root.type = tarfile.DIRTYPE
            tar.addfile(root)
            configure = tarfile.TarInfo("bindfs-1.18.4/configure")
            configure.size = 2
            tar.addfile(configure, io.BytesIO(b"#!"))
            if extra:
                tar.addfile(tarfile.TarInfo(extra))

    monkeypatch.setattr(downloader, "transfer", transfer)
    if extra:
        with pytest.raises(ValueError, match="unexpected archive paths"):
            downloader.bindfs()
        assert archive.read_bytes() == b"previous archive"
    else:
        downloader.bindfs()
        with tarfile.open(archive) as tar:
            assert tar.extractfile("bindfs-1.18.4/configure").read() == b"#!"
        import hashlib

        checksum = archive.with_name(archive.name + ".sha256")
        assert (
            checksum.read_text().strip()
            == hashlib.sha256(archive.read_bytes()).hexdigest()
        )
        monkeypatch.setattr(
            downloader,
            "transfer",
            lambda *a, **kw: pytest.fail("cache downloaded again"),
        )
        downloader.bindfs()
        archive.write_bytes(b"corrupted")
        monkeypatch.setattr(downloader, "transfer", transfer)
        downloader.bindfs()
        with tarfile.open(archive) as tar:
            assert tar.extractfile("bindfs-1.18.4/configure").read() == b"#!"
        assert (
            checksum.read_text().strip()
            == hashlib.sha256(archive.read_bytes()).hexdigest()
        )
    assert not list(tmp_path.glob("*.tmp.*"))
