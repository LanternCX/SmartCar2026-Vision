#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE="$SCRIPT_DIR/main_v2.py"
RULES_PATH="$ROOT_DIR/calibration/chromaforge-rules.json"
MODEL_SOURCE="$ROOT_DIR/yolo/yolo.tflite"
TARGET_DIR="${TARGET_DIR:-/Volumes/NO NAME}"
TARGET_PATH="$TARGET_DIR/main.py"
INCLUDE_YOLO="${1:-}"

if [ "$INCLUDE_YOLO" != "" ] && [ "$INCLUDE_YOLO" != "yolo" ]; then
  echo "用法: $0 [yolo]" >&2
  exit 1
fi

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
if [ "$INCLUDE_YOLO" = "yolo" ]; then
  if [ ! -f "$MODEL_SOURCE" ]; then
    echo "未找到模型文件 $MODEL_SOURCE" >&2
    exit 1
  fi
  cp "$MODEL_SOURCE" "$TARGET_DIR/yolo.tflite"
fi

echo "已更新主车 v2 入口: $SOURCE"
echo "已上传主车 v2 入口: $TARGET_PATH"
if [ "$INCLUDE_YOLO" = "yolo" ]; then
  echo "已上传 YOLO 模型: $TARGET_DIR/yolo.tflite"
fi
