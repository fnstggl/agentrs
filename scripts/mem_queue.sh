#!/usr/bin/env bash
# Run queued shell jobs one after another as memory allows (MemAvailable >= MIN_GB).
# Usage: MIN_GB=3.2 bash scripts/mem_queue.sh jobs.txt   (one shell command per line)
set -u
MIN_KB=$(awk "BEGIN{print int(${MIN_GB:-3.0}*1024*1024)}")
while IFS= read -r job || [ -n "$job" ]; do
  [ -z "$job" ] && continue
  while true; do
    avail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    [ "$avail" -ge "$MIN_KB" ] && break
    sleep 30
  done
  echo "$(date +%T) start: $job"
  bash -c "$job" &
  sleep ${SETTLE_S:-240}
done < "$1"
wait
echo "$(date +%T) queue finished"
