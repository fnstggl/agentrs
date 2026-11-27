#!/bin/bash
# Push the current feat/cam-live commit to origin as an off-box backup (operator-authorized 2026-11-26:
# "push milestones"). Secret-scans every commit not yet on the remote; never force-pushes.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
[ "$(git rev-parse --abbrev-ref HEAD)" = "feat/cam-live" ] || { echo "not on feat/cam-live"; exit 2; }
HELPER=${CAM_LIVE_CRED_HELPER:-/root/.config/camlive/gh_cred_helper.sh}
SHA=$(git rev-parse HEAD)
BASE=$(git ls-remote --heads origin feat/cam-live | cut -f1)
BASE=${BASE:-3968583}
git cat-file -e "$BASE" 2>/dev/null || BASE=3968583
PAT='ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gho_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|xox[baprs]-[A-Za-z0-9-]{10,}|[MN][A-Za-z0-9_-]{23,25}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,}|hf_[A-Za-z0-9]{30,}'
# 1) exact check against the real secrets held on this machine (never printed): any occurrence aborts the push
REAL=$(mktemp); trap 'rm -f "$REAL"' EXIT
for f in /root/.config/camlive/discord_bot_token /root/.config/camlive/gh_token /root/.config/agentrs/gh_token; do
  [ -r "$f" ] && tr -d '\n' < "$f" >> "$REAL" && echo >> "$REAL"
done
[ -r /root/.config/camlive/discord_account.json ] && python3 -c "import json;print(json.load(open('/root/.config/camlive/discord_account.json'))['password'])" >> "$REAL"
if [ -s "$REAL" ] && git diff "$BASE..$SHA" | grep -qF -f "$REAL"; then
  echo "secret scan: a REAL credential of this machine occurs in the commits $BASE..$SHA — not pushing"; exit 4
fi
# 2) pattern check; hits are tolerated only in the files listed in tools/secret_scan_allow.txt (test fixtures
#    that contain deliberately fake credentials for the scrubbing / detection tests)
ALLOW=cam_live/tools/secret_scan_allow.txt
HITS=$(git diff "$BASE..$SHA" | awk '/^\+\+\+ /{f=substr($2,3)} /^\+[^+]/{print f"\t"$0}' | grep -E "$PAT" | cut -f1 | sort -u \
       | { if [ -f "$ALLOW" ]; then grep -vxF -f <(grep -v '^#' "$ALLOW") || true; else cat; fi; } | tr '\n' ' ' || true)
if [ -n "${HITS// /}" ]; then echo "secret scan: credential-like text added in: $HITS— not pushing"; exit 3; fi
BIG=$(git diff --stat=200 "$BASE..$SHA" --diff-filter=A --numstat | awk '$1=="-"{print $3}' | head -5)
[ -z "$BIG" ] || echo "note: binary files added: $BIG"
git -c credential.helper= -c credential.helper="$HELPER" push origin "$SHA:refs/heads/feat/cam-live" 2>&1 | sed -E 's#https://[^@/]*@#https://***@#g' | tail -3
echo "pushed $SHA"
