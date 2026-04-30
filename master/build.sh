#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$SCRIPT_DIR/main.py"
TARGET_DIR="${TARGET_DIR:-/Volumes/NO NAME}"

if [ ! -f "$SOURCE" ]; then
  echo "未找到 $SOURCE" >&2
  exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
  echo "未找到目标目录 $TARGET_DIR" >&2
  exit 1
fi

cp "$SOURCE" "$TARGET_DIR/main.py"
echo "已复制到 $TARGET_DIR/main.py"
