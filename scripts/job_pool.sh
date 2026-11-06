#!/usr/bin/env bash
# Run shell jobs (one per line of a file) with at most MAX_JOBS running concurrently.
# Usage: MAX_JOBS=4 bash scripts/job_pool.sh jobs.txt
set -u
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
while IFS= read -r job || [ -n "$job" ]; do
  [ -z "$job" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "${MAX_JOBS:-4}" ]; do wait -n; done
  echo "$(date +%T) start: ${job:0:160}"
  bash -c "$job" &
done < "$1"
wait
echo "$(date +%T) pool finished"
