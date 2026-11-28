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
# secret scan of every blob any commit of BASE..SHA introduces (not just the net diff), decompressed, binaries as
# bytes: the real credentials of this machine (exact, never printed) and credential-shaped patterns outside
# tools/secret_scan_allow.txt (G2-integration G2I-1 / G2I-2). Any hit aborts the push.
python3 cam_live/tools/secret_scan.py "$BASE" "$SHA" || { echo "not pushing"; exit 3; }
BIG=$(git diff --stat=200 "$BASE..$SHA" --diff-filter=A --numstat | awk '$1=="-"{print $3}' | head -5)
[ -z "$BIG" ] || echo "note: binary files added: $BIG"
git -c credential.helper= -c credential.helper="$HELPER" push origin "$SHA:refs/heads/feat/cam-live" 2>&1 | sed -E 's#https://[^@/]*@#https://***@#g' | tail -3
echo "pushed $SHA"
