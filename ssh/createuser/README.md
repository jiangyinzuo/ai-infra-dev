# 批量管理服务器用户

通过 Ansible 管理 `jiangyinzuo` 账号、SSH 密钥、Docker 用户组及 sudo 权限。支持创建新用户和更新已有用户；默认仅输出 plan，不连接远端。

## 使用方式

以下命令在 `ssh/` 目录执行。

```bash
# 本地预览
./createuser/cli.py --group suzhou

# 连接目标检查，不应用修改
./createuser/cli.py --group suzhou --host-glob '*-130' --check

# 先在单台服务器应用
./createuser/cli.py --group suzhou --host-glob '*-130' --apply

# 批量应用
./createuser/cli.py --group suzhou --group lianqiuhu --apply
```

在 `ssh/` 执行 `source ./completion.bash` 后，可用 Tab 补全选项、组名和主机名。

| 参数 | 含义 |
|---|---|
| `--group` | 必填，可重复；选择连接配置中的目标组 |
| `--host-glob` | 进一步筛选主机，可重复 |
| `--check` | 执行 Ansible check mode，连接远端预览变更，不应用修改，与 `--apply` 互斥 |
| `--apply` | 应用修改 |
| `--workers` | Ansible 并发数，默认 5，必须为正整数 |

WSL 需安装 Python 依赖、Ansible 和 `ansible.posix` collection；远端需 Python 3 和 sudo。

`--apply` 先检查 root 能否使用对应客户端的 `id_ed25519` 免密登录，成功即跳过公钥安装；否则使用 SSH config 的 `Password`，未配置时隐藏提示输入 root 密码，通过 ssh-copy-id 追加 `id_ed25519.pub`，验证成功后再管理 jiangyinzuo。

root 使用 Windows/WSL SSH 配置对应的密钥对。需安装 ssh-copy-id；Windows 密码认证使用 askpass，WSL 还需 sshpass。首次安装需服务器允许 root 密码登录，密码不写入 inventory 或命令行。

默认 plan 不连接远端。`--check` 需 root 已能免密登录，不上传公钥；连接后预览用户配置和 home 迁移，不应用修改。Ansible 以 root 执行，无需 sudo 密码。

## 固定账号与密钥

账号固定为 `jiangyinzuo`，shell 为 `/bin/bash`，不提供用户配置文件。

将 WSL 的 `~/.ssh/id_ed25519.pub` 追加到登录授权，并同步该公钥和私钥到用户 `.ssh/`。WSL 必须存在这对密钥；同名文件按内容更新，私钥权限为 `0600`，其他密钥和已有授权保留。

自动维护 GitHub SSH 配置：通过 `$USG_IP:3128` 代理连接 `ssh.github.com:443`，重复执行不重复添加，保留其他 Host。服务器需安装 `nc`，登录时需有 `USG_IP`；GitHub 公钥仍需自行登记。首次自动信任主机密钥，已有密钥变化时拒绝连接。

## Home 目录

- 新用户从已存在的 `/home`、`/data`、`/data1`、`/data2`、`/data3` 中选择可用空间最多的位置，例如 `/data3/jiangyinzuo`。
- 已有 home 所在盘可用空间不足 40 GiB 时自动迁移；空间充足则不移动。
    - 不在旧路径创建符号链接，原绝对路径不再有效。
- 所有候选位置均不足 40 GiB 时报错。迁移还会拒绝以下情况：
    - 用户有运行进程、目标目录已存在，或 home 包含挂载点。
    - 目标无法容纳旧数据并保留 40 GiB 可用空间。
- `--check` 显示目标 home，但不创建或迁移。迁移前请备份重要数据，退出该用户会话并停止其任务。

## 重复执行与权限

支持重复执行：配置未变且远端状态一致时，不重复修改。保留已有附加组和非托管文件，不修改密码、UID 或主用户组。

所有托管用户都会加入 Docker 组并获得免密 sudo；不安装 Docker，组权限在下次登录生效。远端需已安装 sudo，并在 sudoers 中启用 `/etc/sudoers.d`。
自动将用户加入实际 NPU 设备所属组，保留已有组；没有 NPU 时跳过，重新登录生效，不修改设备权限。

**NOTE：**Docker 组和免密 sudo 均赋予主机级权限。共享私钥泄露后需统一撤销、轮换；不得将私钥提交到 Git。

