#!/usr/bin/env bash
set -euo pipefail

# 必须在昇腾服务器上运行，禁止在本机 WSL 创建容器。
if [[ $(uname -r) == *[Mm]icrosoft* ]]; then
  echo 'error: 禁止在 WSL 中运行，请在远程昇腾服务器上执行。' >&2
  exit 1
fi

# Explicit environment settings bypass the corresponding interactive prompt.
if [[ -z ${IMAGE_NAME:-} ]]; then
  image_rows=$(docker images vllm-ascend --format '{{.CreatedAt}}\t{{.Repository}}:{{.Tag}}')
  [[ -n $image_rows ]] || { echo 'error: 没有可用的 vllm-ascend 镜像。' >&2; exit 1; }
  mapfile -t images < <(printf '%s\n' "$image_rows" | LC_ALL=C sort -r | cut -f2)
  for index in "${!images[@]}"; do
    printf '%d) %s\n' "$((index + 1))" "${images[index]}"
  done
  while true; do
    read -r -p '选择镜像编号 [1，最新]: ' choice || exit 1
    choice=${choice:-1}
    if [[ $choice =~ ^[1-9][0-9]{0,8}$ ]] && ((choice <= ${#images[@]})); then
      IMAGE_NAME=${images[choice - 1]}
      break
    fi
    echo '请输入列表中的有效编号。' >&2
  done
fi

if [[ -z ${CONTAINER_NAME:-} ]]; then
  container_names=$(docker ps -a --format '{{.Names}}')
  next=1
  while IFS= read -r name; do
    if [[ $name =~ ^jiangyinzuo-dev([0-9]{1,9})$ ]]; then
      number=$((10#${BASH_REMATCH[1]}))
      ((number < next)) || next=$((number + 1))
    fi
  done <<< "$container_names"
  default_name="jiangyinzuo-dev$next"
  read -r -p "容器名 [$default_name]: " CONTAINER_NAME || exit 1
  CONTAINER_NAME=${CONTAINER_NAME:-$default_name}
fi
printf '创建容器 %s，镜像 %s\n' "$CONTAINER_NAME" "$IMAGE_NAME"

# 在服务器上执行；仅挂载实际存在的额外数据目录。
extra_volumes=()
for directory in /data1 /data2 /data3; do
  if test -d "$directory"; then
    extra_volumes+=(-v "$directory:$directory")
  fi
done

# /etc/hccn.conf: 昇腾设备的 HCCN 网络配置文件
docker create \
  --name "$CONTAINER_NAME" \
  --net=host \
  --shm-size=500g \
  --privileged=true \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci6 \
  --device /dev/davinci7 \
  --device /dev/davinci8 \
  --device /dev/davinci9 \
  --device /dev/davinci10 \
  --device /dev/davinci11 \
  --device /dev/davinci12 \
  --device /dev/davinci13 \
  --device /dev/davinci14 \
  --device /dev/davinci15 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  --hostname "${HOSTNAME}-container" \
  -e PIP_INDEX_URL=https://mirrors.tools.huawei.com/pypi/simple \
  -e PIP_EXTRA_INDEX_URL= \
  -e PIP_TRUSTED_HOST=mirrors.tools.huawei.com \
  -v /home:/home \
  -v /data:/data \
  -v /mnt:/mnt \
  -v /tmp:/tmp \
  "${extra_volumes[@]}" \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -it "$IMAGE_NAME" bash
docker cp "$HOME/devtools/openeuler24.03-lts-aarch64/openEuler.repo" \
  "$CONTAINER_NAME:/etc/yum.repos.d/openEuler.repo"
