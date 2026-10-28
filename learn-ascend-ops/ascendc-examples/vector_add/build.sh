#!/usr/bin/env bash
set -xeuo pipefail

# Load the CANN environment before running this script, for example:
# source /usr/local/Ascend/ascend-toolkit/set_env.sh

ARCH="${CMAKE_ASC_ARCHITECTURES:-dav-2201}"
BISHENG="${BISHENG:-bisheng}"
BUILD_DIR="build"

mkdir -p "${BUILD_DIR}/add_static" "${BUILD_DIR}/add_tpipe_tque"

"${BISHENG}" -std=c++17 -O2 --npu-arch="${ARCH}" -c --asc-aicore-lang \
    add_static.asc -o "${BUILD_DIR}/add_static/add_static.o"
"${BISHENG}" --npu-arch="${ARCH}" \
    "${BUILD_DIR}/add_static/add_static.o" -o "${BUILD_DIR}/add_static/add_static"

"${BISHENG}" -std=c++17 -O2 --npu-arch="${ARCH}" -c --asc-aicore-lang \
    add_tpipe_tque.asc -o "${BUILD_DIR}/add_tpipe_tque/add_tpipe_tque.o"
"${BISHENG}" --npu-arch="${ARCH}" \
    "${BUILD_DIR}/add_tpipe_tque/add_tpipe_tque.o" -o "${BUILD_DIR}/add_tpipe_tque/add_tpipe_tque"

mkdir -p "${BUILD_DIR}/pipeline_static" "${BUILD_DIR}/pipeline_tpipe_tque"

"${BISHENG}" -std=c++17 -O2 --npu-arch="${ARCH}" -c --asc-aicore-lang \
    add_pipeline_static.asc -o "${BUILD_DIR}/pipeline_static/pipeline_static.o"
"${BISHENG}" --npu-arch="${ARCH}" \
    "${BUILD_DIR}/pipeline_static/pipeline_static.o" -o "${BUILD_DIR}/pipeline_static/pipeline_static"

"${BISHENG}" -std=c++17 -O2 --npu-arch="${ARCH}" -c --asc-aicore-lang \
    add_pipeline_tpipe_tque.asc -o "${BUILD_DIR}/pipeline_tpipe_tque/pipeline_tpipe_tque.o"
"${BISHENG}" --npu-arch="${ARCH}" \
    "${BUILD_DIR}/pipeline_tpipe_tque/pipeline_tpipe_tque.o" -o "${BUILD_DIR}/pipeline_tpipe_tque/pipeline_tpipe_tque"

printf 'Built:\n  %s\n  %s\n  %s\n  %s\n' \
    "${BUILD_DIR}/add_static/add_static" \
    "${BUILD_DIR}/add_tpipe_tque/add_tpipe_tque" \
    "${BUILD_DIR}/pipeline_static/pipeline_static" \
    "${BUILD_DIR}/pipeline_tpipe_tque/pipeline_tpipe_tque"
