#!/usr/bin/env bash
set -euo pipefail

# Run from the project directory. Configure CANN first:
# source /usr/local/Ascend/ascend-toolkit/set_env.sh

REPS="${REPS:-5}"
PROFILE_ROOT="${PROFILE_ROOT:-profiles/$(date +%Y%m%d_%H%M%S)}"

if ! [[ "${REPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPS must be a positive integer" >&2
    exit 1
fi
command -v msprof >/dev/null

python3 gen_data.py
./build.sh
mkdir -p "${PROFILE_ROOT}"
./build/add_static/add_static
cp output/output.bin "${PROFILE_ROOT}.static_reference.bin"
./build/add_tpipe_tque/add_tpipe_tque
cmp "${PROFILE_ROOT}.static_reference.bin" output/output.bin
cmp output/output.bin output/golden.bin

printf 'kernel\tround\tduration_us\tprofile_csv\n' > "${PROFILE_ROOT}/durations.tsv"

run_profile() {
    local kernel="$1"
    local round="$2"
    local executable="$3"
    local output_dir="${PROFILE_ROOT}/${kernel}_r${round}"
    local csv_files csv duration

    msprof op --output="${output_dir}" "${executable}"
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
        run_profile static "${round}" ./build/add_static/add_static
        cp output/output.bin "${PROFILE_ROOT}/static_output_r${round}.bin"
        run_profile tpipe_tque "${round}" ./build/add_tpipe_tque/add_tpipe_tque
        cmp "${PROFILE_ROOT}/static_output_r${round}.bin" output/output.bin
    else
        run_profile tpipe_tque "${round}" ./build/add_tpipe_tque/add_tpipe_tque
        cp output/output.bin "${PROFILE_ROOT}/tpipe_tque_output_r${round}.bin"
        run_profile static "${round}" ./build/add_static/add_static
        cmp "${PROFILE_ROOT}/tpipe_tque_output_r${round}.bin" output/output.bin
    fi
    cmp output/output.bin output/golden.bin
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

rm -f "${PROFILE_ROOT}.static_reference.bin"
printf 'Profiles and summaries are under %s\n' "${PROFILE_ROOT}"
