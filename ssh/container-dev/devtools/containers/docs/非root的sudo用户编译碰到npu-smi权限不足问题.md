# 非root的sudo用户编译碰到npu-smi权限不足问题

## 1. 问题与结论

日期：2026-10-15。服务器 SSH 别名：`suzhou-a3-130`，主机名：`node-97-31`。硬件 PCI Device ID 为 `0xD803`，项目按 A3 平台识别；驱动工具版本为 26.1.0。开发容器名：`jiangyinzuo-dev1`。

现象：宿主机普通用户可以执行`npu-smi`；容器中相同 UID 的普通用户即使已加入设备组，仍然失败。容器 root 查询成功。

失败原因：**容器普通用户打开 `/dev/davinci0` 时，驱动返回 `EBUSY`，不是设备文件的 `EACCES`。** 驱动 namespace 表显示设备 0、1 已分配给其他容器；设备共享状态为 False。这与昇腾官方说明的跨容器设备占用场景吻合。

另发现一个独立问题：容器内 `sudo npu-smi` 找不到 `libc_sec.so`。显式传入容器的 `LD_LIBRARY_PATH` 后查询成功。该问题与普通用户的 `EBUSY/-8020` 不同。

## 2. 原始命令与关键输出

### 2.1 初始安装失败

在容器的 vllm-ascend 仓库中执行：

```bash
VLLM_TARGET_DEVICE=empty COMPILE_CUSTOM_KERNELS=0 \
  pip3 install -v -e . --no-deps --no-build-isolation
```

元数据生成阶段报错：

```text
subprocess.CalledProcessError: Command '['npu-smi', 'info', '-l']' returned non-zero exit status 187.
RuntimeError: Get chip info failed: Command '['npu-smi', 'info', '-l']' returned non-zero exit status 187.
```

### 2.2 检查容器用户和设备组

宿主机执行：

```console
[jiangyinzuo@node-97-31 ~]$ docker inspect --format 'User={{.Config.User}} GroupAdd={{json .HostConfig.GroupAdd}}' jiangyinzuo-dev1
User=9011:9011 GroupAdd=["1000"]

[jiangyinzuo@node-97-31 ~]$ docker exec jiangyinzuo-dev1 sh -c '
  id
  ls -ln /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
  /usr/local/bin/npu-smi info -t board -i 0
'
uid=9011(jiangyinzuo) gid=9011(jiangyinzuo) groups=9011(jiangyinzuo),1000
crw-rw----. 1 1000 1000 235, 0 Oct 15 21:21 /dev/davinci_manager
crw-rw----. 1 1000 1000 511, 0 Oct 15 21:21 /dev/devmm_svm
crw-rw----. 1 1000 1000 510, 0 Oct 15 21:21 /dev/hisi_hdc
Failed to obtain the console log level. The possible causes are as follows:
1. Different containers share the same device;
2. Failed to open the character device when interacting with the OS kernel;
3. Failed to obtain the number of devices.
dcmi model initialized failed, because the device is used. ret is -8020
```

结论：pip install报错原因在于`npu-smi info`执行失败。

**注意**: `1. Different containers share the same device;
`
### 2.3 root用户执行

```bash
docker exec -u 0 -it jiangyinzuo-dev1 \
  /usr/local/bin/npu-smi info -t board -i 0
```

输出：
```text
        NPU ID                         : 0
        Product Name                   : NA
        Model                          : NA
        Manufacturer                   : Huawei
        Serial Number                  : NA
        Software Version               : 26.1.0
        Firmware Version               : 9.0.0.6.200
        Compatibility                  : OK
        Board ID                       : 0x71
        Main Board ID                  : 0x18
        PCB ID                         : B
        BOM ID                         : 2
        PCIe Bus Info                  : 0000:9D:00.0
        Slot ID                        : 0
        Class ID                       : NA
        PCI Vendor ID                  : 0x19E5
        PCI Device ID                  : 0xD803
        Subsystem Vendor ID            : 0x19E5
        Subsystem Device ID            : 0x3001
        Chip Count                     : 2
```

另外，如果直接用sudo查询，会报错：

```console
[jiangyinzuo@node-97-31 ~]$ docker exec -it jiangyinzuo-dev1 sudo /usr/local/bin/npu-smi info -t board -i 0
/usr/local/bin/npu-smi: error while loading shared libraries: libc_sec.so: cannot open shared object file: No such file or directory
[jiangyinzuo@node-97-31 ~]$
```

但如果传入LD_LIBRARY_PATH：
```bash
docker exec jiangyinzuo-dev1 sh -c \
  'sudo -n env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" /usr/local/bin/npu-smi info -t board -i 0'
```

查询成功，返回上述 board 信息。

## 3. 定位过程

### 3.1 排除遗漏附加组和常规设备权限

目的：检查普通用户是否缺少驱动设备组。[昇腾手动挂载容器指南](https://www.hiascend.com/doc_center/source/zh/canncommercial/800/softwareinst/instg/instg_0059.html)要求非 root 进程加入驱动运行组。

```console
[jiangyinzuo@node-97-31 ~]$ id
uid=9011(jiangyinzuo) gid=9011(jiangyinzuo) groups=9011(jiangyinzuo),982(docker),1000(HwHiAiUser)

[jiangyinzuo@node-97-31 ~]$ docker exec jiangyinzuo-dev1 id
uid=9011(jiangyinzuo) gid=9011(jiangyinzuo) groups=9011(jiangyinzuo),1000
```

**结论：宿主机与容器用户均属于设备 GID 1000；结合前文设备权限 `1000:1000 / 0660`，不是漏加设备组。** 组名是否显示为 `HwHiAiUser` 不影响数字 GID 权限。

目的：检查容器是否启用 user namespace，以及后续关联驱动记录所需的容器 PID。

```console
[jiangyinzuo@node-97-31 ~]$ docker inspect --format 'Pid={{.State.Pid}} Ipc={{.HostConfig.IpcMode}} Userns={{.HostConfig.UsernsMode}} Privileged={{.HostConfig.Privileged}}' jiangyinzuo-dev1
Pid=1685616 Ipc=private Userns= Privileged=true
```

**结论：容器为特权模式，未显式配置 UsernsMode；记录 PID 1685616，供后续匹配驱动 namespace。** 这些配置本身不能证明 NPU 可跨容器共享。

目的：检查普通进程的有效 capabilities 和 seccomp 状态。

```console
[jiangyinzuo@node-97-31 ~]$ docker exec jiangyinzuo-dev1 sh -c 'grep -E "^(Cap|NoNewPrivs|Seccomp)" /proc/self/status'
CapInh: 0000003fffffffff
CapPrm: 0000000000000000
CapEff: 0000000000000000
CapBnd: 0000003fffffffff
CapAmb: 0000000000000000
NoNewPrivs:     0
Seccomp:        0
Seccomp_filters:        0
```

**结论：没有 seccomp 过滤，普通用户的有效 capabilities 为零。** 不能把 `--privileged` 理解为普通用户等同 root；也没有证据表明当前错误由缺少某项 capability 导致，继续用 strace 查实际失败点。

### 3.2 使用 strace 定位实际失败系统调用

宿主机已有 `/usr/bin/strace`，容器没有。先用 `ldd /usr/bin/strace` 检查动态库依赖，再将可在该容器运行的宿主机 tracer 临时复制到共享 `/tmp`；没有安装软件，也没有修改容器镜像。

实际执行的关键命令：

```bash
probe=$(mktemp -d /tmp/omp-npu-trace.XXXXXX)
cp /usr/bin/strace "$probe/strace"
chmod 755 "$probe/strace"
docker exec jiangyinzuo-dev1 "$probe/strace" \
  -f -e trace=openat,access,ioctl,mkdir,connect \
  -o "$probe/trace" \
  /usr/local/bin/npu-smi info -t board -i 0
```

查询本身仍打印如下错误，随后从 trace 中筛选设备与权限相关行。以下是当时筛选结果的完整输出，不是完整 strace 文件：

```text
Failed to obtain the console log level. The possible causes are as follows:
1. Different containers share the same device;
2. Failed to open the character device when interacting with the OS kernel;
3. Failed to obtain the number of devices.
dcmi model initialized failed, because the device is used. ret is -8020
TRACE_DIR=/tmp/omp-npu-trace.3iOfB0
755   openat(AT_FDCWD, "/dev/davinci_manager", O_RDWR|O_CLOEXEC) = 3
755   openat(AT_FDCWD, "/dev/davinci_manager", O_RDWR|O_CLOEXEC) = 4
755   openat(AT_FDCWD, "/dev/davinci0", O_RDONLY) = -1 EBUSY (Device or resource busy)
755   openat(AT_FDCWD, "/dev/davinci_manager", O_RDWR|O_CLOEXEC) = 3
755   openat(AT_FDCWD, "/dev/hisi_hdc", O_RDONLY) = 5
755   openat(AT_FDCWD, "/dev/davinci0", O_RDONLY) = -1 EBUSY (Device or resource busy)
755   openat(AT_FDCWD, "/dev/davinci0", O_RDONLY) = -1 EBUSY (Device or resource busy)
755   openat(AT_FDCWD, "/dev/davinci_manager", O_RDWR|O_CLOEXEC) = 6
755   openat(AT_FDCWD, "/dev/davinci0", O_RDONLY) = -1 EBUSY (Device or resource busy)
763   openat(AT_FDCWD, "/sys/firmware/dmi/entries/0-0/raw", O_RDONLY|O_NOCTTY|O_CLOEXEC) = -1 EACCES (Permission denied)
755   openat(AT_FDCWD, "/dev/davinci0", O_RDONLY) = -1 EBUSY (Device or resource busy)
```

**定位结论：容器普通用户能打开管理设备和 HDC 设备，但打开 `/dev/davinci0` 时被驱动返回 `EBUSY`，问题不在设备节点的组读写权限。** 宿主机成功查询时也出现 DMI 路径的 `EACCES`，该错误不能解释容器查询失败。

### 3.3 查询设备共享模式

在宿主机执行只读查询：

```bash
sudo -n env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  /usr/local/bin/npu-smi info -t device-share -i 0 -c 0
```

输出：
```text
Device-share Status            : False
```

### 3.4 根据官方案例检查驱动 namespace 归属

目的：前面的 strace 已发现 `openat(..., "/dev/davinci0", ...) = -1 EBUSY`，这里查询驱动记录中的 `udevid=0`，再关联到具体容器。

[昇腾官方同错误案例](https://www.hiascend.com/document/caselibrary/detail/topic_0000002238354198)指出，其他容器或不同 mount namespace 的进程占用设备可触发 `-8020`，并给出以下定位方法。[vllm-ascend issue #632](https://github.com/vllm-project/vllm-ascend/issues/632)也报告了同类现象。

**第一步：查设备映射到哪个驱动 namespace，以及对应的进程 PID。** 以下完整保留汇总表对应的四个 namespace 块，省略其他 namespace：

```console
[jiangyinzuo@node-97-31 ~]$ sudo -n cat /proc/uda/namespace_node
ns_id 0 identify e7f538a2b7b2 root_tgid 2735387 dev_num 4 namespace 00000000a41b077e, udev list:
logic_devid udevid
       0        0
       1        1
       2       12
       3       13

ns_id 3 identify a3df7af8b265 root_tgid 2397878 dev_num 2 namespace 000000004e2af953, udev list:
logic_devid udevid
       0        2
       1        3

ns_id 1 identify 66472cf6509d root_tgid 2415717 dev_num 4 namespace 000000000038df75, udev list:
logic_devid udevid
       0        4
       1        5
       2        8
       3        9

ns_id 2 identify c1b0a6781d4c root_tgid 3996866 dev_num 1 namespace 00000000dabe210f, udev list:
logic_devid udevid
       0        9

```

`/dev/davinci0` <-> udevid=0，位于root_tgid=2735387

**第二步：把 PID 2735387 对应到具体容器名称。** `docker ps -q` 列出运行中容器，批量 inspect 输出各容器主进程 PID 和名称。以下保留相关行：

```console
[jiangyinzuo@node-97-31 ~]$ docker ps -q | xargs -r docker inspect --format '{{.State.Pid}} {{.Name}}'
1685616 /jiangyinzuo-dev1
2735387 /ascend-demo-task
2397878 /ascend-pr17489-direct-130
2415717 /w00800576-day1-130
3996866 /h00954440-workspace
```

官方案例关注的 `ns_id < 128` 映射（不表示实时利用率，有记录不等于正在计算，无记录也不能保证设备空闲）

| ns_id | root_tgid | udevid | 对应容器 |
|---|---|---|---|
| 0 | 2735387 | 0、1、12、13 | ascend-demo-task |
| 3 | 2397878 | 2、3 | ascend-pr17489-direct-130 |
| 1 | 2415717 | 4、5、8、9 | w00800576-day1-130 |
| 2 | 3996866 | 9 | h00954440-workspace |


**结论：失败设备 `/dev/davinci0` 已映射到其他容器 `ascend-demo-task`；结合共享模式为 False、open 返回 EBUSY，符合官方描述的跨容器设备占用限制。** 下一步应由管理员协调设备分配，而不是继续增加用户组或修改设备文件权限。

## 4. 使用的工具及作用

| 工具/接口 | 作用 | 注意点 |
|---|---|---|
| docker inspect | 检查默认 UID、附加组、PID、namespace 配置 | 不输出完整 Config.Env，避免泄露 API key |
| id、ls -ln | 检查数字 UID/GID 与设备文件权限 | 名称不同不代表数字组权限不同 |
| strace | 找到 openat 返回 EBUSY 的实际设备路径 | 只跟踪只读 board 查询，避免跟踪业务任务 |
| ldd、LD_LIBRARY_PATH | 分析 tracer 可运行性和 sudo 缺库问题 | 不给未知库路径做全局提权配置 |
| npu-smi info | 查询 board、设备共享模式 | 未执行 set，不更改共享模式 |
| /proc/uda/namespace_node | 查看昇腾驱动的 namespace/设备映射 | 按官方案例解读，不能等同于利用率 |
| docker inspect State.Pid | 将 root_tgid 关联到具体容器 | 进程 PID 会变化，应在同一时间窗口采集 |
| dmesg | 搜索近期驱动隔离相关信息 | 本次未找到直接解释当前失败的新日志 |
| 官方文档与项目 issue | 核对错误码、排查流程与操作风险 | 不同型号/版本的设置支持范围不可直接照搬 |


## 5. 解决方法

容器内可以执行：
```bash
sudo -n env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  /usr/local/bin/npu-smi info -t board -i 0
```

`pip install`时设置实际 `SOC_VERSION` 可跳过`npu-smi info`。`SOC_VERSION`由宿主机传入容器。

长期解决方案：联系管理员
- 独占全部NPU / 避免开发容器映射全部NPU
- 开启device-share

