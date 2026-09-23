#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ! command -v xcodegen >/dev/null 2>&1; then
    echo "xcodegen is required on the Mac. Install it with: brew install xcodegen" >&2
    exit 1
fi
cd "${SCRIPT_DIR}"
xcodegen generate --spec project.yml
echo "Generated ${SCRIPT_DIR}/Scan.xcodeproj"
