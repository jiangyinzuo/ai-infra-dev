#!/usr/bin/env bash
set -euo pipefail

# Usage: extract.sh [--host USER@HOST]; WSL requires a remote extraction target.
# Extract tar archives and single-file gzip archives beside this script only.
# Do not recurse; container images live separately under ../images/.
# Keep the archives; existing extracted files may be overwritten.
script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
remote_target=
while (($#)); do
    case $1 in
        --host)
            (($# >= 2)) || { echo 'error: --host requires USER@HOST' >&2; exit 2; }
            remote_target=$2
            shift 2
            ;;
        -h|--help)
            echo 'Usage: extract.sh [--host USER@HOST]'
            exit 0
            ;;
        *) echo "error: unknown argument '$1'" >&2; exit 2 ;;
    esac
done
source "$script_dir/../remote-exec.sh"
remote_extract_if_wsl "$remote_target" "${script_dir##*/}/extract.sh"
cd -- "$script_dir"

command -v tar >/dev/null || { echo "error: tar is required" >&2; exit 1; }
command -v gzip >/dev/null || { echo "error: gzip is required" >&2; exit 1; }
shopt -s nullglob
archives=(*.tar *.tar.gz *.tgz *.tar.xz *.tar.bz2 *.tar.zst)
gzip_archives=()
for archive in *.gz; do
    [[ $archive == *.tar.gz ]] || gzip_archives+=("$archive")
done

if ((${#archives[@]} == 0 && ${#gzip_archives[@]} == 0)); then
    echo 'No supported archives found.'
fi

for archive in "${archives[@]}"; do
    printf 'extract %s\n' "$archive"
    # 工具解压默认覆盖同名文件，保留已有目录的权限和时间，不恢复归档所有者，并忽略 `TAR_OPTIONS`（避免 `--keep-old-files` 阻止更新）。
    # 目标目录必须对执行用户可写；不会自动 sudo 或修正 root 所有的目录。解压会保留目标中的额外文件
    # 不应与正在使用这些工具的任务同时执行。
    env -u TAR_OPTIONS tar --extract --file="$archive" --directory="$script_dir" \
        --no-overwrite-dir --no-same-owner --no-same-permissions
done

for archive in "${gzip_archives[@]}"; do
    file=${archive%.gz}
    # The Tree-sitter release filename includes the platform; the CLI does not.
    [[ $archive != tree-sitter-linux-arm64.gz ]] || file=tree-sitter
    printf 'extract %s\n' "$archive"
    gzip -dc -- "$archive" > "$file"
    [[ $file != tree-sitter ]] || chmod +x -- "$file"
done

# 只添加一次；path.env 尚未同步时不影响 shell 启动。
bashrc="$HOME/.bashrc"
source_line='[[ ! -f "$HOME/devtools/openeuler24.03-lts-aarch64/path.env" ]] || source "$HOME/devtools/openeuler24.03-lts-aarch64/path.env"'
if [[ -f $bashrc ]] && grep -Fqx -- "$source_line" "$bashrc"; then
    printf 'skip %s (path.env already configured)\n' "$bashrc"
else
    printf '\n%s\n' "$source_line" >> "$bashrc"
    printf 'updated %s; run: source ~/.bashrc\n' "$bashrc"
fi
