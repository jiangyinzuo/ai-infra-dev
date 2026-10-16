import json
import shutil
import subprocess
import tempfile
from pathlib import Path

container = "jiangyinzuo-sudo-build-test"

# 容器通过共享 /tmp 访问 tracer 和日志。
trace_dir = Path(tempfile.mkdtemp(prefix="torch-npu-raw-", dir="/tmp"))
tracer = trace_dir / "strace"
trace_log = trace_dir / "strace.log"
stdout_log = trace_dir / "stdout.log"
stderr_log = trace_dir / "stderr.log"
result_file = trace_dir / "result.json"

shutil.copy2("/usr/bin/strace", tracer)

command = [
    "docker",
    "exec",
    container,
    "env",
    "-u",
    "TORCH_DEVICE_BACKEND_AUTOLOAD",
    str(tracer),
    "-f",  # 跟踪子进程和线程
    "-tt",  # 显示微秒精度时间戳
    "-T",  # 显示系统调用耗时
    "-s",
    "256",  # 字符串最多显示 256 字节
    "-o",
    str(trace_log),
    "python3",
    "-c",
    "import torch_npu",
]

# 分别保存原始 stdout/stderr，不与 strace 日志混合。
result = subprocess.run(command, capture_output=True)

stdout_log.write_bytes(result.stdout)
stderr_log.write_bytes(result.stderr)
result_file.write_text(
    json.dumps(
        {
            "command": command,
            "exit_code": result.returncode,
        },
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

print(f"TRACE_DIR={trace_dir}")
print(f"EXIT_CODE={result.returncode}")

print("\n--- RAW STDOUT ---")
print(result.stdout.decode(errors="replace"), end="")

print("\n--- RAW STDERR ---")
print(result.stderr.decode(errors="replace"), end="")

# 这里只筛选屏幕摘要；strace.log 仍保留完整跟踪。
print("\n--- DEVICE OPEN ERRORS (excerpt) ---")
if trace_log.is_file():
    with trace_log.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            is_davinci = "/dev/davinci" in line
            is_access_error = "EBUSY" in line or "EACCES" in line
            if is_davinci and is_access_error:
                print(line, end="")
else:
    print("未生成 strace.log，请检查上面的 stderr。")

print(f"\n完整系统调用日志：{trace_log}")
print(f"原始标准输出：    {stdout_log}")
print(f"原始标准错误：    {stderr_log}")
print(f"执行命令与退出码：{result_file}")
