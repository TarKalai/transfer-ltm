#!/usr/bin/env bash
# Move the code to a machine without GitHub (Domino), as one file.
#
#   bash scripts/codesync.sh pack [FILE]           # laptop: write the committed code to FILE
#   bash scripts/codesync.sh pull FILE [--force]   # Domino: make this code folder that version
#
# FILE is a git bundle (default: transfer-ltm.bundle next to the code folder). It holds
# the history of `main` and nothing that is not committed: no data, no local notes, no
# untracked files. Upload it the way you upload any file, then pull.
#
# pull replaces the code with the bundle's version: changed files are overwritten and
# files deleted from the repo are deleted. Everything that is not part of the repo
# (data/, runs, .venv, files of your own) is left alone.
# It refuses when the folder has edits of its own (or commits the bundle does not
# contain), unless --force, which first saves them under .codesync_backup/<time>/.
# The first pull turns the folder into a git working copy (the files it replaces are
# copied to .codesync_backup/ first); runs then record the code version they used
# (run_info.json: git_commit).
set -euo pipefail

BRANCH=main
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() { echo "codesync: $*" >&2; exit 1; }
abspath() { echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }
command -v git >/dev/null || die "git is not installed on this machine"

cmd="${1:-}"
shift || true
# FILE arguments are relative to where the command is typed; the work happens in ROOT.
args=()
for arg in "$@"; do
    case "$arg" in
        --*) args+=("$arg") ;;
        *) [ -e "$(dirname "$arg")" ] && arg="$(abspath "$arg")"; args+=("$arg") ;;
    esac
done
set -- "${args[@]+"${args[@]}"}"
cd "$ROOT"

case "$cmd" in
pack)
    out="${1:-$ROOT/../transfer-ltm.bundle}"
    git rev-parse --verify -q "$BRANCH" >/dev/null || die "no branch '$BRANCH' here"
    if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
        echo "NOTE: these uncommitted changes are NOT in the bundle (commit them first):"
        git status --short --untracked-files=no | sed 's/^/  /'
    fi
    git bundle create "$out" "$BRANCH" 2>/dev/null
    echo "Wrote $out ($(du -h "$out" | cut -f1)): $BRANCH at $(git log -1 --format='%h %s' "$BRANCH")"
    ;;

pull)
    file="" force=0
    for arg in "$@"; do
        case "$arg" in
            --force) force=1 ;;
            *) file="$arg" ;;
        esac
    done
    [ -n "$file" ] || die "usage: bash scripts/codesync.sh pull FILE [--force]"
    [ -f "$file" ] || die "$file not found"
    backup=".codesync_backup/$(date +%Y%m%d-%H%M%S)"

    first=0
    if [ ! -d .git ]; then
        first=1
        git init -q
        git symbolic-ref HEAD "refs/heads/$BRANCH"
    fi
    git bundle verify "$file" >/dev/null 2>&1 || die "$file is not a complete bundle (upload cut short?)"
    git fetch -q "$file" "$BRANCH" 2>/dev/null         || die "could not read branch '$BRANCH' from $file (upload cut short? pack it again)"
    new="$(git rev-parse FETCH_HEAD)"

    if [ "$first" = 1 ]; then
        # The bundle's files replace whatever copies are here: keep the current ones.
        n=0
        while IFS= read -r f; do
            if [ -f "$f" ]; then
                mkdir -p "$backup/$(dirname "$f")"
                cp -p "$f" "$backup/$f"
                n=$((n + 1))
            fi
        done < <(git ls-tree -r --name-only "$new")
        git reset -q --hard "$new"
        [ "$n" -gt 0 ] && echo "First pull: the $n files it replaced were copied to $backup/"
        echo "Code is now $(git log -1 --format='%h %s (%cd)' --date=short)"
    else
        old="$(git rev-parse HEAD)"
        dirty="$(git status --porcelain --untracked-files=no)"
        local_commits=0
        git merge-base --is-ancestor "$old" "$new" || local_commits=1
        if [ -n "$dirty" ] || [ "$local_commits" = 1 ]; then
            if [ "$force" = 0 ]; then
                [ -n "$dirty" ] && { echo "Edited here since the last pull:"; echo "$dirty" | sed 's/^/  /'; }
                [ "$local_commits" = 1 ] && echo "This folder has commits the bundle does not contain."
                die "nothing changed. Rerun with --force to save these to .codesync_backup/ and take the bundle's code"
            fi
            mkdir -p "$backup"
            [ -n "$dirty" ] && git diff HEAD > "$backup/uncommitted.patch"
            [ "$local_commits" = 1 ] && git bundle create "$backup/local-commits.bundle" HEAD 2>/dev/null
            echo "Saved this folder's own changes to $backup/"
        fi
        if [ "$old" = "$new" ] && [ -z "$dirty" ]; then
            echo "Already up to date: $(git log -1 --format='%h %s')"
            exit 0
        fi
        git reset -q --hard "$new"
        echo "Updated $(git rev-parse --short "$old") -> $(git log -1 --format='%h %s (%cd)' --date=short)"
        git diff --stat "$old" "$new" | tail -n 20
    fi

    extra="$(git ls-files --others --exclude-standard)"
    if [ -n "$extra" ]; then
        echo
        echo "Not part of the code (left alone; delete them if they are old copies):"
        echo "$extra" | head -n 25 | sed 's/^/  /'
        [ "$(echo "$extra" | wc -l)" -gt 25 ] && echo "  ..."
    fi
    ;;

*)
    sed -n '2,19p' "$0"
    exit 1
    ;;
esac
