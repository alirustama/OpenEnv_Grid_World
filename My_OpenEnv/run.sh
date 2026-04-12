#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -f .env.submission ]; then
  set -a
  # shellcheck disable=SC1091
  source .env.submission
  set +a
fi

echo "Starting local OpenEnv inference..."
python inference.py
