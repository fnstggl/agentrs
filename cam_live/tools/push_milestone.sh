#!/bin/bash
# Push the current feat/cam-live commit to origin as an off-box backup (operator-authorized 2026-11-26:
# "push milestones"). Secret-scans every commit not yet on the remote; never force-pushes.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
[ "$(git rev-parse --abbrev-ref HEAD)" = "feat/cam-live" ] || { echo "not on feat/cam-live"; exit 2; }
HELPER=${CAM_LIVE_CRED_HELPER:-/root/.config/agentrs/cred_helper.sh}
SHA=$(git rev-parse HEAD)
BASE=$(git ls-remote --heads origin feat/cam-live | cut -f1)
BASE=${BASE:-3968583}
git cat-file -e "$BASE" 2>/dev/null || BASE=3968583
PAT='ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gho_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|xox[baprs]-[A-Za-z0-9-]{10,}|[MN][A-Za-z0-9_-]{23,25}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,}|hf_[A-Za-z0-9]{30,}'
HITS=$(git diff "$BASE..$SHA" | grep '^+' | grep -cE "$PAT" || true)
if [ "$HITS" != "0" ]; then echo "secret scan: $HITS suspicious added lines between $BASE and $SHA — not pushing"; exit 3; fi
BIG=$(git diff --stat=200 "$BASE..$SHA" --diff-filter=A --numstat | awk '$1=="-"{print $3}' | head -5)
[ -z "$BIG" ] || echo "note: binary files added: $BIG"
git -c credential.helper= -c credential.helper="$HELPER" push origin "$SHA:refs/heads/feat/cam-live" 2>&1 | sed -E 's#https://[^@/]*@#https://***@#g' | tail -3
echo "pushed $SHA"
