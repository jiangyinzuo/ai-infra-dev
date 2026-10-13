#!/usr/bin/env bash

# 在服务器上执行；仅挂载实际存在的额外数据目录。
extra_volumes=()
for directory in /data1 /data2 /data3; do
  if test -d "$directory"; then
    extra_volumes+=(-v "$directory:$directory")
  fi
done

# /etc/hccn.conf: 昇腾设备的 HCCN 网络配置文件
docker run \
  --name $CONTAINER_NAME \
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
  -it $IMAGE_NAME bash
