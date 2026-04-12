#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
HTML_DIR="${SCRIPT_DIR}/html"
PORT="${1:-8000}"

if [[ ! -d "${HTML_DIR}" ]]; then
  echo "HTML directory not found: ${HTML_DIR}" >&2
  exit 1
fi

if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=python
else
  echo "Python is required to host the HTML demo." >&2
  exit 1
fi

echo "Serving ${HTML_DIR} at http://0.0.0.0:${PORT}/"
cd "${HTML_DIR}"
exec "${PYTHON_CMD}" -m http.server "${PORT}"