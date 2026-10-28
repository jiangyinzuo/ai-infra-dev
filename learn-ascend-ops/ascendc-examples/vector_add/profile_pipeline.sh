#!/usr/bin/env bash
set -euo pipefail

# Run from the project directory after configuring CANN.
# source /usr/local/Ascend/ascend-toolkit/set_env.sh

REPS="${REPS:-5}"
PROFILE_ROOT="${PROFILE_ROOT:-profiles/pipeline_$(date +%Y%m%d_%H%M%S)}"

if ! [[ "${REPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPS must be a positive integer" >&2
    exit 1
fi
command -v msprof >/dev/null

python3 gen_pipeline_data.py
./build.sh
if [[ -e "${PROFILE_ROOT}" ]]; then
    echo "PROFILE_ROOT already exists: ${PROFILE_ROOT}" >&2
    exit 1
fi
mkdir -p "${PROFILE_ROOT}"

rm -f pipeline_output/static.bin pipeline_output/tpipe_tque.bin
./build/pipeline_static/pipeline_static
cmp pipeline_output/static.bin pipeline_output/golden.bin
./build/pipeline_tpipe_tque/pipeline_tpipe_tque
cmp pipeline_output/static.bin pipeline_output/tpipe_tque.bin

printf 'kernel\tround\tduration_us\tprofile_csv\n' > "${PROFILE_ROOT}/durations.tsv"

run_profile() {
    local kernel="$1"
    local round="$2"
    local executable="$3"
    local output_dir="${PROFILE_ROOT}/${kernel}_r${round}"
    local csv_files csv duration

    rm -f "pipeline_output/${kernel}.bin"
    msprof op --output="${output_dir}" "${executable}"
    cmp "pipeline_output/${kernel}.bin" pipeline_output/golden.bin
    csv_files=("${output_dir}"/OPPROF_*/OpBasicInfo.csv)
    if [[ ! -f "${csv_files[0]}" || "${#csv_files[@]}" -ne 1 ]]; then
        echo "Expected exactly one OpBasicInfo.csv under ${output_dir}" >&2
        exit 1
    fi
    csv="${csv_files[0]}"
    duration="$(awk -F, 'NR == 2 { print $3; exit }' "${csv}")"
    if [[ -z "${duration}" ]]; then
        echo "Could not read Task Duration(us) from ${csv}" >&2
        exit 1
    fi
    printf '%s\t%s\t%s\t%s\n' "${kernel}" "${round}" "${duration}" "${csv}" \
        | tee -a "${PROFILE_ROOT}/durations.tsv"
}

for ((round = 1; round <= REPS; ++round)); do
    if (( round % 2 == 1 )); then
        run_profile static "${round}" ./build/pipeline_static/pipeline_static
        run_profile tpipe_tque "${round}" ./build/pipeline_tpipe_tque/pipeline_tpipe_tque
    else
        run_profile tpipe_tque "${round}" ./build/pipeline_tpipe_tque/pipeline_tpipe_tque
        run_profile static "${round}" ./build/pipeline_static/pipeline_static
    fi
    cmp pipeline_output/static.bin pipeline_output/tpipe_tque.bin
done

awk -F '\t' '
NR == 1 { next }
{
    count[$1]++
    sum[$1] += $3
    if (!(($1) in min) || $3 < min[$1]) min[$1] = $3
    if (!(($1) in max) || $3 > max[$1]) max[$1] = $3
}
END {
    print "kernel\tcount\tmean_us\tmin_us\tmax_us"
    for (kernel in count) {
        printf "%s\t%d\t%.6f\t%.6f\t%.6f\n", kernel, count[kernel], \
            sum[kernel] / count[kernel], min[kernel], max[kernel]
    }
}' "${PROFILE_ROOT}/durations.tsv" | tee "${PROFILE_ROOT}/summary.tsv"

printf 'Profiles and summaries are under %s\n' "${PROFILE_ROOT}"
