# Podman 安装提示

- 日期：2026-10-16
- 环境：openEuler 22.03、ARM64、内核 5.10、cgroups v1、昇腾 A3。
- git commit：e4183ad14a256993af6ad37dbdbe3a636e24ce1d

Podman的宿主机用户映射到root用户，并不能解决容器内访问被其它进程占用的/dev/davinci0报错的问题。

## 安装与配置

- 使用第三方 `mgoltzsche/podman-static` **v5.8.8** 因为Podman 6 已移除 cgroups v1 支持。
- `~/.config/containers/containers.conf` 需配置以下**绝对路径**（将 `/工具目录` 替换为实际路径）：

```toml
[engine]
conmon_path = ["/home/jiangyinzuo/devtools/openeuler24.03-lts-aarch64/podman-linux-arm64/usr/local/lib/podman/conmon"]
helper_binaries_dir = ["/home/jiangyinzuo/devtools/openeuler24.03-lts-aarch64/podman-linux-arm64/usr/local/lib/podman"]
runtime = "/home/jiangyinzuo/devtools/openeuler24.03-lts-aarch64/podman-linux-arm64/usr/local/bin/crun"
```

解决以下报错问题：
```
[jiangyinzuo@node-97-31 ~]$  podman info --debug
Error: could not find a working conmon binary (configured options: [/usr/libexec/podman/conmon /usr/local/libexec/podman/conmon /usr/local/lib/podman/conmon /usr/bin/conmon /usr/sbin/conmon /usr/local/bin/conmon /usr/local/sbin/conmon /run/current-system/sw/bin/conmon]: invalid argument)
```

- 缺少 `policy.json` 会让 `podman load` 报“unsupported image formats”。可复制包内 `etc/containers/policy.json` 到用户配置目录；其宽松策略不要求签名，但不关闭 HTTPS 校验。
- rootless 需有效的 subuid/subgid 映射及辅助程序；用 `podman info --debug`、`podman unshare cat /proc/self/uid_map` 检查。

## 存储与运行

- 默认镜像/容器存储：`~/.local/share/containers/storage`
- Docker 与 Podman 不共用镜像存储。
- 自定义路径在用户 `storage.conf` 的 `[storage]` 下设置 `graphroot`。改路径不会迁移旧数据；已有配置/存储不自动切换。

参考：[静态包安装说明](https://github.com/mgoltzsche/podman-static#binary-installation-on-a-host) · [昇腾跨容器占用案例](https://www.hiascend.com/document/caselibrary/detail/topic_0000002238354198)
