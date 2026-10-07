#!/usr/bin/env python3
"""Small HF download probe. Run: python3 hf_speedtest.py [https://endpoint ...]
Defaults probe public endpoints, not endorsements. Never send credentials.
Requires curl >= 8.4 (streaming --max-filesize enforcement).
Each endpoint gets one 1 MiB Range request and a 12-second deadline.
Results measure this machine's current network (including proxy environment).
Default output is a Chinese table; speeds use MiB/s (1 MiB = 1024 KiB).
Failed probes have no comparable speed; the summary ranks overall speed.
"""
import concurrent.futures
import json
import re
import subprocess
import sys
import tempfile
import unicodedata

LIMIT = 1024 * 1024
DEFAULTS = ["https://huggingface.co", "https://hf-mirror.com", "https://hf-mirror.net", "https://d6108366.hf-mirror.com"]
FILE = "/gpt2/resolve/main/tokenizer.json"


def probe(endpoint):
    with tempfile.NamedTemporaryFile() as header_file:
        command = ["curl", "--silent", "--show-error", "--location", "--fail",
                   "--proto", "=https", "--proto-redir", "=https",
                   "--connect-timeout", "4", "--max-time", "12",
                   "--range", f"0-{LIMIT - 1}", "--max-filesize", str(LIMIT),
                   "--output", "/dev/null", "--dump-header", header_file.name,
                   "--write-out", "%{json}", endpoint.rstrip("/") + FILE]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=14)
            data = json.loads(result.stdout)
            header_file.seek(0)
            response_headers = header_file.read().decode(errors="replace")
        except (subprocess.TimeoutExpired, ValueError, OSError) as exc:
            return {"endpoint": endpoint, "status": "FAILED", "error": str(exc)}
    count = int(data.get("size_download", 0))
    code = data.get("http_code", 0)
    content_type = data.get("content_type") or ""
    valid_range = bool(re.search(r"(?im)^content-range:\s*bytes 0-1048575/", response_headers))
    ok = result.returncode == 0 and count == LIMIT and code == 206 and valid_range and "html" not in content_type.lower()
    total = data.get("time_total", 0)
    ttfb = data.get("time_starttransfer", 0)
    return {"endpoint": endpoint, "status": "OK" if ok else "FAILED",
            "http": code, "bytes": count, "ttfb_s": round(ttfb, 3),
            "total_s": round(total, 3),
            "overall_MiB_s": round(count / total / LIMIT, 3) if total else 0,
            "body_MiB_s": round(count / (total - ttfb) / LIMIT, 3) if total > ttfb else 0,
            "final_url": data.get("url_effective"),
            "error": result.stderr.strip() if result.returncode else ("" if ok else "响应不符合要求：需要完整的 1 MiB HTTP 206 分段数据，不能是 HTML 或错误页面")}

def print_row(values, widths):
    cells = []
    for value, width in zip(values, widths):
        text = str(value)
        display_width = sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)
        cells.append(text + " " * max(0, width - display_width))
    print("  ".join(cells).rstrip(), flush=True)



def main():
    version = subprocess.check_output(["curl", "--version"], text=True).split()[1]
    if tuple(map(int, version.split(".")[:2])) < (8, 4):
        sys.exit("需要 curl >= 8.4，才能安全限制流式下载大小")
    endpoints = sys.argv[1:] or DEFAULTS
    if any(not url.startswith("https://") for url in endpoints):
        sys.exit("仅支持 https:// 地址")
    print(f"HF 下载测速 · {len(endpoints)} 个站点（并发）", flush=True)
    print(f"每站请求一次，最多下载 1 MiB，超时 12 秒；文件：{FILE}", flush=True)
    print("速度单位：MiB/s；综合速度包含连接和首字节等待，传输速度仅计算正文下载。", flush=True)
    print("失败结果不参与速度比较；结果受当前网络及代理环境影响。\n", flush=True)
    widths = [max(4, *(len(endpoint) for endpoint in endpoints)), 4, 6, 12, 12, 10, 10, 10]
    print_row(["站点", "状态", "HTTP", "综合 MiB/s", "传输 MiB/s", "首字节(s)", "总耗时(s)", "下载(MiB)"], widths)
    print_row(["-" * width for width in widths], widths)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(endpoints))) as pool:
        futures = [pool.submit(probe, endpoint) for endpoint in endpoints]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            ok = result["status"] == "OK"
            print_row([
                result["endpoint"], "成功" if ok else "失败", result.get("http") or "—",
                f'{result["overall_MiB_s"]:.3f}' if ok else "—",
                f'{result["body_MiB_s"]:.3f}' if ok else "—",
                f'{result["ttfb_s"]:.3f}' if "ttfb_s" in result else "—",
                f'{result["total_s"]:.3f}' if "total_s" in result else "—",
                f'{result["bytes"] / LIMIT:.3f}' if "bytes" in result else "—",
            ], widths)
            if not ok:
                print(f'  失败原因：{result["error"]}', flush=True)
            final_url = result.get("final_url")
            if final_url and final_url != result["endpoint"].rstrip("/") + FILE:
                print(f"  重定向至：{final_url}", flush=True)
    successful = [result for result in results if result["status"] == "OK"]
    print(f"\n完成：{len(successful)}/{len(results)} 个站点成功。", flush=True)
    if successful:
        fastest = max(successful, key=lambda result: result["overall_MiB_s"])
        print(f'本次综合速度最快：{fastest["endpoint"]}（{fastest["overall_MiB_s"]:.3f} MiB/s）', flush=True)
    else:
        print("所有站点测速失败，请查看上方失败原因。", flush=True)


if __name__ == "__main__":
    main()
