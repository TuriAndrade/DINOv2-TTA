#!/usr/bin/env bash

# Usage: ./script_executor.sh runs.txt
# Each line in runs.txt must be a full command (e.g., "python3 main.py ...")

RUNS_FILE="$1"

if [[ ! -f "$RUNS_FILE" ]]; then
  echo "File not found: $RUNS_FILE"
  exit 1
fi

i=0
while IFS= read -r CMD || [[ -n "$CMD" ]]; do
  # Skip empty lines and comments
  [[ -z "$CMD" || "$CMD" =~ ^# ]] && continue

  i=$((i+1))
  echo "[$i] Running: $CMD"
  eval "$CMD"
done < "$RUNS_FILE"