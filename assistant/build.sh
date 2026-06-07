#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE="$SCRIPT_DIR/main.py"
RULES_PATH="$ROOT_DIR/calibration/chromaforge-rules.json"
TARGET_DIR="${TARGET_DIR:-/Volumes/NO NAME}"
TARGET_PATH="$TARGET_DIR/main.py"

if [ ! -f "$SOURCE" ]; then
  echo "未找到 $SOURCE" >&2
  exit 1
fi

if [ ! -f "$RULES_PATH" ]; then
  echo "未找到标定文件 $RULES_PATH" >&2
  exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
  echo "未找到目标目录 $TARGET_DIR" >&2
  exit 1
fi

cd "$ROOT_DIR"

uv run python -m calibration.chromaforge_export_adapter \
  "$RULES_PATH" \
  --source "$SOURCE" \
  --output "$SOURCE" \
  --task-constant-name OBJECT_TASKS

cp "$SOURCE" "$TARGET_PATH"

echo "已更新辅车入口: $SOURCE"
echo "已上传辅车入口: $TARGET_PATH"
