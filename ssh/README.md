## Install Dependencies

```bash
python3 -m pip install sshconf==0.2.7
uv tool install --force --with ansible ansible-core
```

## Shared connection profiles

`hosts.json` defines group selection, SSH config/client paths and default public
keys. `ssh_hosts.py` loads profiles, discovers Windows SSH paths once per process,
and builds inventories for NPU queries, machine reports and user provisioning.
Each tool adds its own host variables; connection logic stays in the shared module.

Public-key installation uses the same profiles, including `password_method`
(`sshpass` by default, `windows-askpass` for Windows clients):

```bash
python3 ssh/install-ssh-keys.py --group suzhou --dry-run
python3 ssh/install-ssh-keys.py --group xiaoqiaoling --host-glob '*-24' --dry-run
```

Run these examples from the repository root. Installation wrapper scripts keep
their existing group-specific entrypoints. `--host-glob` filters within the selected
profiles; `--public-key` overrides their default keys. The old installer-specific
`--config`, `--ssh-config`, `--ssh` and `--password-method` flags are replaced by
`--profiles` configuration.

User provisioning parses users once, caches key contents for the current run and
uses one resolved plan for preview and application. Default mode is plan only:

```bash
python3 ssh/createuser/createuser.py --group xiaoqiaoling
uv run --with pytest --with sshconf==0.2.7 python -m pytest -q \
  ssh/createuser/smoketest ssh/machine-info/smoketest ssh/npu-top/smoketest
```

USG operations require exported source IPs from `source ssh/ssh-get-usg-ip`.
Refactoring does not change sudo grants, key validation or overlapping-alias policy.
