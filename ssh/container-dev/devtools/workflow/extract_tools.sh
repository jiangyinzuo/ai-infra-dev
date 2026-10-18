#!/usr/bin/env bash
set -euo pipefail

# Internal extraction helper; the Python workflow owns CLI and remote dispatch.
# Extract platform archives only, preserving archives and unrelated files.
script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../openeuler24.03-lts-aarch64" && pwd)
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
    destination=$script_dir
    case $archive in
        fzf-*-linux_arm64.tar.gz)
            version=${archive#fzf-}
            version=${version%-linux_arm64.tar.gz}
            support="$script_dir/fzf-${version}-support"
            for script in fzf-preview.sh fzf-tmux; do
                [[ -f $support/$script ]] || { echo "error: missing $support/$script" >&2; exit 1; }
            done
            destination="$HOME/plugged/fzf/bin"
            mkdir -p "$destination"
            ;;
    esac
    # 工具解压默认覆盖同名文件，保留已有目录的权限和时间，不恢复归档所有者，并忽略 `TAR_OPTIONS`（避免 `--keep-old-files` 阻止更新）。
    # 目标目录必须对执行用户可写；不会自动 sudo 或修正 root 所有的目录。解压会保留目标中的额外文件
    # 不应与正在使用这些工具的任务同时执行。
    env -u TAR_OPTIONS tar --extract --file="$archive" --directory="$destination" \
        --no-overwrite-dir --no-same-owner --no-same-permissions
    case $archive in
        bindfs-*.tar.gz)
            bindfs_path=$(command -v bindfs || true)
            if [[ -z $bindfs_path && -x $HOME/.local/bin/bindfs ]]; then
                bindfs_path="$HOME/.local/bin/bindfs"
            fi
            if [[ -z $bindfs_path ]]; then
                builder="$HOME/vimrc/install/build_bindfs.sh"
                [[ -f $builder ]] || {
                    echo "error: missing $builder; update vimrc first" >&2
                    exit 1
                }
                sudo yum install -y gcc make pkgconf-pkg-config fuse-devel
                bash "$builder" "$script_dir/${archive%.tar.gz}"
            else
                printf 'skip bindfs build (already installed: %s)\n' "$bindfs_path"
            fi
            ;;
        fzf-*-linux_arm64.tar.gz)
            cp -- "$support/fzf-preview.sh" "$support/fzf-tmux" "$destination/"
            chmod +x -- "$destination/fzf" "$destination/fzf-preview.sh" "$destination/fzf-tmux"
            ;;
    esac
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
