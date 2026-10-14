#!/usr/bin/env bash
set -euo pipefail

# Usage: ./download-a3-container.sh [YYYYMMDD] [VERSION]
# Example: ./download-a3-container.sh 20261013 26.2.0
if [[ ${1:-} == -h || ${1:-} == --help || $# -gt 2 ]]; then
  printf 'Usage: %s [YYYYMMDD] [VERSION]\nDefaults: 20261012 26.2.0\n' "$0"
  if [[ $# -gt 2 ]]; then
    exit 1
  fi
  exit 0
fi
image_date=${1:-20261012}
image_version=${2:-26.2.0}

# Keep the container image under images/a3/, separate from tool/plugin archives.
script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$script_dir"

image_tag="dev-${image_version}.day${image_date}-800I-A3-py311-openEuler24.03-lts-aarch64"
image="vllm-ascend:$image_tag"
archive="vllm-ascend_${image_tag}.tar.gz"
url="https://cmc-nkg-artifactory.cmc.tools.huawei.com/artifactory/cmc-nkg-inner/MindIE/vLLM%20Ascend/${image_version}.daily/${image_date:0:6}/${image_date}/vLLM-Ascend-images/$image.tar.gz"

# On WSL only check the archive; servers also check the exact Docker image tag.
if [[ -f $archive ]]; then
  printf 'skip %s (archive exists)\n' "$archive"
  exit 0
fi
if [[ $(uname -r) != *[Mm]icrosoft* ]]; then
  images=$(docker images vllm-ascend --format '{{.Repository}}:{{.Tag}}')
  while IFS= read -r existing_image; do
    if [[ $existing_image == "$image" ]]; then
      printf 'skip %s (Docker image exists)\n' "$image"
      exit 0
    fi
  done <<< "$images"
fi

temp=$(mktemp "${archive}.tmp.XXXXXX")
trap 'rm -f -- "$temp"' EXIT
wget --no-check-certificate -O "$temp" "$url"
mv -- "$temp" "$archive"
trap - EXIT

