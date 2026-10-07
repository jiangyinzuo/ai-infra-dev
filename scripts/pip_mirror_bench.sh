#!/usr/bin/env bash
# pip/uv 镜像测速：索引响应延迟 + 定量 wheel 吞吐（只取前几 MiB，快速出结果）
#
# 用法:
#   ./pip_mirror_bench.sh              # 默认测 numpy
#   ./pip_mirror_bench.sh numpy torch  # 指定候选包（取首个能解析到 wheel 的）
#   WHEEL_FILENAME=nvidia_cufft-12.0.0.61-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl ./pip_mirror_bench.sh nvidia-cufft
#
# 依赖: curl, getent, python3, awk, bc
set -uo pipefail

PKGS=("${@:-numpy}")
MIRRORS=(
  "pypi.org|https://pypi.org/simple/"
  "aliyun|https://mirrors.aliyun.com/pypi/simple/"
  "tsinghua|https://pypi.tuna.tsinghua.edu.cn/simple/"
  "ustc|https://mirrors.ustc.edu.cn/pypi/simple/"
  "tencent|https://mirrors.cloud.tencent.com/pypi/simple/"
  "huawei|https://repo.huaweicloud.com/repository/pypi/simple/"
)

PY_TAG="cp312"
URL_TIMEOUT=25
DL_TIMEOUT=8           # 单镜像最长下载时间（秒），到点即停
RANGE_MB=8             # 只取前 N MiB 定量测速，不下完整包
UA="pip/24.0"

# ---- 工具函数 ----------------------------------------------------------------
resolve_host() { echo "$1" | sed -E 's#https?://([^/]+).*#\1#'; }

fmt() { # 秒 -> 可读（>=1s 用秒，否则毫秒）
  awk -v v="$1" 'BEGIN{ if (v>=1) printf "%.3fs", v; else printf "%.0fms", v*1000 }'
}

# 从包索引页抓一个 manylinux x86_64 wheel 链接（相对或绝对）
find_wheel() { # args: index_url pkg
  local page code wheel
  page=$(curl -sSL -A "$UA" -m "$URL_TIMEOUT" \
    -w '\n%{time_starttransfer} %{http_code}' "$1$2/" 2>/dev/null) || return 1
  code=$(echo "$page" | tail -1 | awk '{print $2}')
  [ "$code" = "200" ] || return 1
  if [ -n "${WHEEL_FILENAME:-}" ]; then
    wheel=$(printf '%s' "$page" | python3 -c '
import os, sys
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
class Links(HTMLParser):
    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href", "")
            if unquote(urlsplit(href).path).rsplit("/", 1)[-1] == os.environ["WHEEL_FILENAME"]:
                print(href)
Links().feed(sys.stdin.read())
')
    [ -n "$wheel" ] || return 1
    printf '%s\n' "$wheel"
    return
  fi
  wheel=$(echo "$page" \
    | grep -oE "href=\"[^\"]*$2-[0-9][^\"]*${PY_TAG}-${PY_TAG}-[^\"]*manylinux[^\"]*x86_64\.whl" \
    | head -1 | sed 's/href="//;s/"$//')
  [ -n "$wheel" ] || return 1
  echo "$wheel"
}

# 相对链接 -> 绝对 URL
abs_url() { # args: index_url pkg rel
  python3 - "$1$2/" "$3" <<'PY'
import sys, urllib.parse
print(urllib.parse.urljoin(sys.argv[1], sys.argv[2]))
PY
}

# ---- 阶段 1: 索引响应延迟 ----------------------------------------------------
echo "=============================================================="
echo " 阶段 1 索引响应 (GET /simple/<pkg>/)   包: ${PKGS[*]}"
echo "=============================================================="
printf "%-10s %6s %8s %8s %8s %8s\n" MIRROR HTTP DNS TCP TLS TTFB
for entry in "${MIRRORS[@]}"; do
  name="${entry%%|*}"; url="${entry##*|}"
  host=$(resolve_host "$url")
  dns=$( { /usr/bin/time -f "%e" getent hosts "$host" >/dev/null; } 2>&1 )
  r=$(curl -sSL -A "$UA" -o /dev/null -m "$URL_TIMEOUT" \
      -w '%{time_namelookup} %{time_connect} %{time_appconnect} %{time_starttransfer} %{http_code}' \
      "${url}${PKGS[0]}/" 2>/dev/null)
  if [ -z "$r" ]; then
    printf "%-10s %6s %8s %8s %8s %8s\n" "$name" FAIL "-" "-" "-" "-"
    continue
  fi
  read -r nl tc ta tt code <<<"$r"
  printf "%-10s %6s %8s %8s %8s %8s\n" \
    "$name" "$code" "$(fmt "$nl")" "$(fmt "$tc")" "$(fmt "$ta")" "$(fmt "$tt")"
done

# ---- 阶段 2: 定量 wheel 吞吐（Range 前 N MiB，串行、实时逐行输出） ----------
echo
echo "=============================================================="
echo " 阶段 2 wheel 吞吐（取前 ${RANGE_MB}MiB / 最长 ${DL_TIMEOUT}s，逐行输出）"
echo "=============================================================="
MAXRANGE=$((RANGE_MB*1024*1024-1))
TARGET=$((RANGE_MB*1024*1024))
printf "%-10s %-16s %9s %8s %12s %5s\n" MIRROR WHEEL TIME MiB/s GOT_B CODE
for entry in "${MIRRORS[@]}"; do
  name="${entry%%|*}"; url="${entry##*|}"
  rel=""
  for p in "${PKGS[@]}"; do
    rel=$(find_wheel "$url" "$p") && { pkg="$p"; break; }
  done
  if [ -z "$rel" ]; then
    printf "%-10s %-16s %9s %8s %12s %5s\n" "$name" "n/a" "-" "-" "-" "-"
    continue
  fi
  abs=$(abs_url "$url" "$pkg" "$rel")
  wname=$(basename "${abs%%\?*}" | sed -E 's/\.whl$//; s/-cp[0-9].*//')
  r=$(curl -sSL -A "$UA" -r "0-$MAXRANGE" -o /dev/null -m "$DL_TIMEOUT" \
      -w '%{time_total} %{speed_download} %{http_code} %{size_download}' \
      "$abs" 2>/dev/null)
  if [ -z "$r" ]; then
    printf "%-10s %-16s %9s %8s %12s %5s\n" "$name" "$wname" "FAIL" "-" "-" "-"
    continue
  fi
  read -r tot sp code sz <<<"$r"
  mib=$(echo "scale=2; $sp/1048576" | bc -l 2>/dev/null)
  printf "%-10s %-16s %9s %8s %12s %5s\n" \
    "$name" "$wname" "$(fmt "$tot")" "$mib" "$sz" "$code"
done
echo
echo "注: GOT_B 达到 $TARGET 说明按时下完，MiB/s 有效；"
echo "    明显偏小说明 ${DL_TIMEOUT}s 内没下完（源慢）；CODE=403 为该镜像拒绝直连文件。"
