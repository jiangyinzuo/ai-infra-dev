#!/usr/bin/env bash
# Usage: ./update-vllm.sh (may be invoked from any directory).
# Align the sibling vllm worktree with vllm-ascend's verified main commit.
# Requires bash and git. Fetches missing commits from origin; does not install packages.
# Leaves vllm in detached HEAD; refuses staged, unstaged, or untracked changes.
set -euo pipefail

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

[[ $# -eq 0 ]] || die 'Usage: update-vllm.sh'
workspace=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
vllm_dir="$workspace/vllm"
commit_file="$workspace/vllm-ascend/.github/vllm-main-verified.commit"

[[ -f "$commit_file" ]] || die "Missing dependency file: $commit_file"
commit=$(<"$commit_file")
commit=${commit//[[:space:]]/}
[[ "$commit" =~ ^[[:xdigit:]]{40}$ ]] || die "Expected a full Git commit SHA in $commit_file"
commit=${commit,,}

# Require a repository root here, not an enclosing repository discovered by Git.
[[ -e "$vllm_dir/.git" ]] || die "Not a Git worktree: $vllm_dir"
git -C "$vllm_dir" rev-parse --is-inside-work-tree >/dev/null ||
    die 'Cannot open vllm worktree; if moved, run git worktree repair from its main repository.'
[[ -z "$(git -C "$vllm_dir" status --porcelain --untracked-files=all)" ]] ||
    die 'vllm has local changes; commit or stash them (including untracked files) before updating.'

printf 'Dependency: %s\nTarget:     %s\n' "$commit_file" "$commit"
if ! git -C "$vllm_dir" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    git -C "$vllm_dir" fetch --no-tags origin "$commit"
fi
git -C "$vllm_dir" checkout --detach "$commit"
printf 'Updated %s to %s (detached HEAD).\n' "$vllm_dir" "$(git -C "$vllm_dir" rev-parse HEAD)"