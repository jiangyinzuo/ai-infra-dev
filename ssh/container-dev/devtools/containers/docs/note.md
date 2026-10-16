# 容器环境使用root，但容器root和宿主机用户拥有相同的文件权限

## 非root用户在没有开启device-share时，无法查询被其它用户占用的device

vLLM Ascend容器执行`VLLM_TARGET_DEVICE=empty COMPILE_CUSTOM_KERNELS=1 pip3 install -v -e . --no-deps --no-build-isolation`时，
需要`openat("/dev/davinci0", O_RDONLY)`设备来查询NPU架构。

然而，容器中直接执行`npu-smi info`可能会产生以下报错：

```
Failed to obtain the console log level. The possible causes are as follows:
1. Different containers share the same device;
2. Failed to open the character device when interacting with the OS kernel;
3. Failed to obtain the number of devices.
dcmi model initialized failed, because the device is used. ret is -8020
```

经过排查，
```bash
docker exec jiangyinzuo-dev1 sh -c \
  'sudo -n env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" /usr/local/bin/npu-smi info -t board -i 0'
```

就不会报错了。查询`device-share`

```bash
sudo -n env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  /usr/local/bin/npu-smi info -t device-share -i 0 -c 0
```

输出`Device-share Status            : False`，说明没有开启device-share的时候，普通用户
无法在device被其它用户占用的时候，执行`npu-smi info`。

## 简单对npu-smi info命令提权，无法解决问题

使用以下提权脚本：

/usr/bin/npu-smi
```bash
#!/bin/sh
exec /usr/bin/sudo -n /usr/local/libexec/devtools-npu-query "$@"
```

```bash
#!/bin/sh
set -eu

case "$*" in
  "info"|"info -l")
    ;;
  *)
    [ "$#" -eq 5 ] || [ "$#" -eq 7 ] || exit 2
    [ "$1" = info ] &&
      [ "$2" = -t ] &&
      [ "$3" = board ] &&
      [ "$4" = -i ] || exit 2

    case "$5" in
      ''|*[!0-9]*) exit 2 ;;
    esac

    if [ "$#" -eq 7 ]; then
      [ "$6" = -c ] || exit 2
      case "$7" in
        ''|*[!0-9]*) exit 2 ;;
      esac
    fi
    ;;
esac
```

虽然有一部分`npu-smi info`执行的目的是设置`SOC_VERSION`环境变量，但是，手动设置`SOC_VERSION`，或者给`npu-smi info`加sudo，依然不能解决问题。原因在于python3 import torch_npu也会读取davinci0设备，但并非直接调用`npu-smi`工具。

## strace python3 -c 'import torch_npu'

```bash
docker exec jiangyinzuo-sudo-build-test \
  env -u TORCH_DEVICE_BACKEND_AUTOLOAD \
  /tmp/torch-npu-raw-ap4iopy8/strace \
  -f -tt -T -s 256 \
  -o /tmp/torch-npu-raw-ap4iopy8/strace.log \
  python3 -c 'import torch_npu'
```
执行`trace_log_analysis.py`得到输出：

```
TRACE_DIR=/tmp/torch-npu-raw-eswp4z2n
EXIT_CODE=0

--- RAW STDOUT ---
Failed to obtain the console log level. The possible causes are as follows:
1. Different containers share the same device;
2. Failed to open the character device when interacting with the OS kernel;
3. Failed to obtain the number of devices.

--- RAW STDERR ---

--- DEVICE OPEN ERRORS (excerpt) ---
17193 20:57:42.554165 openat(AT_FDCWD, "/dev/davinci0", O_RDONLY) = -1 EBUSY (Device or resource busy) <0.000031>
17193 20:57:42.557076 openat(AT_FDCWD, "/dev/davinci0", O_RDONLY) = -1 EBUSY (Device or resource busy) <0.000026>
17193 20:57:42.557386 openat(AT_FDCWD, "/dev/davinci0", O_RDONLY) = -1 EBUSY (Device or resource busy) <0.000017>
17193 20:57:42.557759 openat(AT_FDCWD, "/dev/davinci0", O_RDONLY) = -1 EBUSY (Device or resource busy) <0.000015>
17193 20:57:42.560116 openat(AT_FDCWD, "/dev/davinci0", O_RDONLY) = -1 EBUSY (Device or resource busy) <0.000022>
17193 20:57:42.570512 openat(AT_FDCWD, "/dev/davinci0", O_RDONLY) = -1 EBUSY (Device or resource busy) <0.000018>
17193 20:57:42.578672 openat(AT_FDCWD, "/dev/davinci0", O_RDONLY) = -1 EBUSY (Device or resource busy) <0.000019>
17193 20:57:47.394347 openat(AT_FDCWD, "/dev/davinci0", O_RDONLY) = -1 EBUSY (Device or resource busy) <0.000036>

完整系统调用日志：/tmp/torch-npu-raw-eswp4z2n/strace.log
原始标准输出：    /tmp/torch-npu-raw-eswp4z2n/stdout.log
原始标准错误：    /tmp/torch-npu-raw-eswp4z2n/stderr.log
执行命令与退出码：/tmp/torch-npu-raw-eswp4z2n/result.json
```

> TORCH_DEVICE_BACKEND_AUTOLOAD 控制 执行 import torch 时，是否自动加载第三方设备后端，例如昇腾的 torch_npu。
> TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -c 'import torch' 就不会通过插件机制自动导入torch_npu了。

## 可以成功编译的完整命令

```bash
sudo -n /bin/bash -c '
export PATH=/usr/local/python3.11.10/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

unset SOC_VERSION CMAKE_PREFIX_PATH Torch_DIR TORCH_DEVICE_BACKEND_AUTOLOAD
export VLLM_TARGET_DEVICE=empty COMPILE_CUSTOM_KERNELS=1
export MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8

pip3 install -v -e . --no-deps --no-build-isolation
'
```

| 调用方式 | 结果 |
|---|---|
| `sudo pip3 install ...` | 失败：`sudo: pip3: command not found` |
| `sudo /usr/local/python3.11.10/bin/pip3 install ...` | 失败：`npu-smi` 找不到 `libc_sec.so`（LD_LIBRARY_PATH设置不对） |
| sudo 启动 Bash，再设置 PATH、加载 CANN/ATB 环境 | 完整编译安装成功，`SUDO_BUILD_EXIT=0` |


但是，这样一方面需要包装sudo脚本，可能不够方便，另一方面，从宿主机以 jiangyinzuo 检查：

 ```text
csrc/build：10044 个文件/目录属于 UID 0

libvllm_ascend_kernels.so
  uid=0，host_writable=False

vllm_ascend_C.cpython-311-aarch64-linux-gnu.so
  uid=0，host_writable=False
 ```

发现sudo 编译虽然能解决本次构建中的设备访问问题，但不能满足“宿主机普通用户直接修改/删除容器产生的文件”这一要求。

## 基于podman管理容器的尝试

Podman可以创建rootless容器：将宿主机上的普通用户映射为容器中的root用户，但试验发现，通过uid/gid映射的容器root用户并不能免sudo执行`npu-smi info`：

| 验证项 | 结果 |
|---|---|
| 容器默认用户 | `root`，UID/GID 为 `0:0` |
| UID 映射 | 容器 UID 0 → 宿主机 UID 9011 |
| HOME 文件所有权 | 容器显示 `0:0`，宿主机显示 `9011:9011` |
| 宿主机修改、执行容器新建文件 | 通过，包括权限为 `0700` 的文件 |
| 容器内 `npu-smi info` | **失败，驱动返回 `-8020`** |
| 宿主机普通用户 `npu-smi info` | 成功 |
| 加入 host PID namespace | 未解决 |

所以，**rootless 容器 root 不等于宿主机 sudo/root**。仅做 UID 映射不能保证 NPU 驱动接受容器访问。

宿主机当前用户必须已有设备访问权限。rootless Podman 无法赋予宿主机用户本来没有的权限；`--privileged` 也不能突破这个边界。

## 基于bindfs的容器环境创建

https://bindfs.org/
