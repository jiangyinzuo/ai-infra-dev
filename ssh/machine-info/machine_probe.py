"""Read-only Linux probe, sent to the remote Python interpreter over SSH stdin."""
import glob
import json
import os
from pathlib import Path
import shutil
import subprocess
import shlex


PROGRAMS = ("kubelet", "kubectl", "kubeadm", "k3s", "rke2", "microk8s")
SERVICES = ("kubelet", "k3s", "k3s-agent", "rke2-server", "rke2-agent",
            "snap.microk8s.daemon-kubelite")
CONFIG_PATHS = ("/etc/kubernetes", "/var/lib/kubelet/config.yaml",
                "/etc/rancher/k3s", "/etc/rancher/rke2", "/var/snap/microk8s/current")


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=8)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)}


def operating_system(path=Path("/etc/os-release")):
    """Parse PRETTY_NAME as data; never source or execute os-release contents."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "PRETTY_NAME":
                values = shlex.split(value, comments=True)
                if len(values) != 1 or not values[0]:
                    raise ValueError("PRETTY_NAME must contain a nonempty value")
                return {"pretty_name": values[0], "errors": []}
        raise ValueError("PRETTY_NAME is missing")
    except (OSError, UnicodeError, ValueError) as exc:
        return {"pretty_name": None, "errors": [f"{path}: {exc}"]}


def kubernetes():
    programs = {name: shutil.which(name) for name in PROGRAMS}
    # Some binary deployments do not add /opt/kube/bin to noninteractive PATH.
    for name in PROGRAMS:
        candidate = Path("/opt/kube/bin") / name
        if not programs[name] and candidate.is_file() and os.access(candidate, os.X_OK):
            programs[name] = str(candidate)
    services = {}
    errors = []
    for name in SERVICES:
        result = command(["systemctl", "show", name + ".service", "--no-pager",
                          "--property=LoadState,ActiveState,SubState"])
        properties = dict(line.split("=", 1) for line in result["stdout"].splitlines() if "=" in line)
        if result["returncode"] != 0 or not properties:
            errors.append(name + ": " + (result["stderr"] or "service state unavailable"))
        services[name] = properties
    configs = [path for path in CONFIG_PATHS if Path(path).exists()]
    versions = {}
    for name, args in (("kubelet", ["--version"]), ("kubectl", ["version", "--client", "-o", "json"]),
                       ("kubeadm", ["version", "-o", "short"]), ("k3s", ["--version"]),
                       ("rke2", ["--version"]), ("microk8s", ["version"])):
        if not programs[name]:
            continue
        result = command([programs[name], *args])
        if result["returncode"] == 0:
            if name == "kubectl":
                try:
                    versions[name] = json.loads(result["stdout"])["clientVersion"]["gitVersion"]
                except (ValueError, KeyError, TypeError):
                    errors.append("kubectl: unrecognized client version output")
            else:
                versions[name] = result["stdout"]
        else:
            errors.append(name + " version: " + result["stderr"])
    node_programs = ("kubelet", "k3s", "rke2", "microk8s")
    evidence = any(programs[name] for name in node_programs) or any(
        state.get("LoadState") == "loaded" for state in services.values()) or bool(configs)
    status = "installed" if evidence else "unknown" if errors else "client-only" if programs["kubectl"] or programs["kubeadm"] else "not-detected"
    return {"status": status, "programs": programs, "versions": versions,
            "services": services, "config_paths": configs, "errors": errors}


def data_directories(pattern):
    directories = []
    errors = []
    for path in sorted(glob.glob(pattern)):
        if not os.path.isdir(path):
            continue
        entry = {"path": path, "real_path": os.path.realpath(path), "symlink": os.path.islink(path)}
        # GNU df resolves the filesystem for a directory, including bind mounts.
        result = command(["df", "-B1", "--output=source,size,used,avail,pcent,target", "--", path])
        try:
            if result["returncode"] != 0:
                raise ValueError(result["stderr"])
            source, total, used, available, percent, mount = result["stdout"].splitlines()[1].split(None, 5)
            entry.update(filesystem=source, total_bytes=int(total), used_bytes=int(used),
                         available_bytes=int(available), use_percent=int(percent.rstrip("%")), mount=mount)
        except (ValueError, IndexError) as exc:
            entry["error"] = str(exc)
            errors.append(path + ": " + str(exc))
        directories.append(entry)
    return {"directories": directories, "errors": errors}


def main():
    os.environ["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:" + os.environ.get("PATH", "")
    os.environ["LC_ALL"] = "C"
    import sys
    report = {"arch": os.uname().machine, "os": operating_system(), "kubernetes": kubernetes(),
              "storage": data_directories(sys.argv[1])}
    print("__MACHINE_INFO__" + json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
