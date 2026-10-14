# 批量管理服务器用户

`createuser.py` 通过 Ansible 固定管理 `jiangyinzuo` 账号、SSH 密钥、Docker 用户组及 sudo 权限。支持创建新用户和更新已有用户；默认仅输出 plan，不连接远端。

## 使用方式

```bash
# 本地预览
python3 ssh/createuser/createuser.py --group suzhou

# 连接目标检查，不应用修改
python3 ssh/createuser/createuser.py --group suzhou --host-glob '*-130' --check

# 先在单台服务器应用
python3 ssh/createuser/createuser.py --group suzhou --host-glob '*-130' --apply

# 批量应用；需要 sudo 密码时追加 --ask-become-pass
python3 ssh/createuser/createuser.py --group suzhou --group lianqiuhu --apply
```

| 参数 | 含义 |
|---|---|
| `--group` | 必填，可重复；选择连接配置中的目标组 |
| `--host-glob` | 进一步筛选主机，可重复 |
| `--profiles` | 连接配置，默认 `ssh/hosts.json` |
| `--check` | 执行 Ansible check mode，连接远端预览变更，不应用修改，与 `--apply` 互斥 |
| `--apply` | 应用修改 |
| `--ask-become-pass` | 提示输入 sudo 认证密码 |
| `--workers` | Ansible 并发数，默认 5，必须为正整数 |

控制端需具备 Python 依赖、Ansible 及 playbook 使用的 `ansible.posix` collection；远端需满足连接配置中的 Python 环境，并已安装 sudo。`--help` 可查看命令行参数。

## 固定账号与密钥

账号固定为 `jiangyinzuo`，shell 为 `/bin/bash`，不提供用户配置文件。

使用控制端（WSL）的 `~/.ssh/id_ed25519.pub` 追加登录授权，并同步该公钥及 `~/.ssh/id_ed25519` 到远端用户 `.ssh/`。同名文件按内容更新，私钥权限为 `0600`，其他密钥和已有登录授权保留。控制端必须存在这对密钥；不再使用连接组中的默认公钥。

GitHub 公钥登记和主机信任需另行配置。

## 重复执行与权限

支持重复执行：配置未变且远端状态一致时，不重复修改。保留已有附加组和非托管文件，不修改密码、UID 或主用户组。

所有托管用户都会加入 Docker 组并获得免密 sudo；不安装 Docker，组权限在下次登录生效。远端需已安装 sudo，并在 sudoers 中启用 `/etc/sudoers.d`。

**NOTE：**Docker 组和免密 sudo 均赋予主机级权限。共享私钥泄露后需统一撤销、轮换；不得将私钥提交到 Git。

