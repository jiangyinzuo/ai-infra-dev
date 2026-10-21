#!/usr/bin/env python3
"""Python 3.9+ vLLM API check: health, models, interactive streamed question."""

import argparse
import json
import shlex
import subprocess
import sys
import urllib.request

BASE = "http://127.0.0.1:8000"


def request(path, payload=None, *, stream=False):
    # API access is local to the executing machine, never through HTTP proxies.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    with opener.open(req, timeout=120 if payload is not None else 10) as response:
        if stream:
            for line in response:
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
        else:
            return response.read().decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ssh", help="例如 'ssh2 jiangyinzuo@suzhou-a3-129'；省略时本地执行"
    )
    parser.add_argument("--stream", action="store_true", help="流式输出；默认等待完整响应")
    args = parser.parse_args()
    ssh = shlex.split(args.ssh) if args.ssh is not None else []
    if args.ssh is not None and not ssh:
        parser.error("--ssh 不能为空")

    def call(path, payload=None, *, stream=False):
        if not ssh:
            return request(path, payload, stream=stream)
        # Run this standard-library client remotely without installing repository code.
        import inspect

        source = (
            "import json,sys,urllib.request\nBASE="
            + repr(BASE)
            + "\n"
            + inspect.getsource(request)
        )
        source += "\nbody=json.load(sys.stdin)\nresult=request(body['path'],body['payload'],stream=body['stream'])\nif result is not None: print(result,end='')\n"
        command = shlex.join(["python3", "-c", source])
        body = json.dumps({"path": path, "payload": payload, "stream": stream}).encode(
            "utf-8"
        )
        result = subprocess.run(
            [*ssh, command],
            input=body,
            stdout=None if stream else subprocess.PIPE,
            check=True,
        )
        if not stream:
            return result.stdout.decode("utf-8")

    print("check health...", flush=True)
    call("/health")
    print("health: OK", flush=True)
    print("查询模型…", flush=True)
    response = json.loads(call("/v1/models"))
    models = [row["id"] for row in response["data"]]
    if not models or any(not isinstance(model, str) or not model for model in models):
        raise ValueError("模型列表无效或没有可用模型")
    for index, model in enumerate(models, 1):
        print(f"{index}) {model}")
    while True:
        choice = input("选择模型编号 [1，q 取消]：").strip() or "1"
        if choice.lower() == "q":
            return 0
        if choice.isascii() and choice.isdigit() and 1 <= int(choice) <= len(models):
            model = models[int(choice) - 1]
            break
        print("请输入有效编号")
    print(f"向 {model} 提问：写一首关于人工智能的诗（{'流式' if args.stream else '非流式'}输出）…", flush=True)
    result = call(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "写一首关于人工智能的诗"}],
            "max_tokens": 20,
            "stream": args.stream,
        },
        stream=args.stream,
    )
    if not args.stream:
        print(json.dumps(json.loads(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
        EOFError,
    ) as exc:
        print("error: " + str(exc), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
