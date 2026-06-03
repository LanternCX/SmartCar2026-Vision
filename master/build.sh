#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$SCRIPT_DIR/main.py"
MODEL_SOURCE="$SCRIPT_DIR/../yolo/yolo.tflite"
TARGET_DIR="${TARGET_DIR:-/Volumes/NO NAME}"
INCLUDE_YOLO="${1:-}"

if [ "$INCLUDE_YOLO" != "" ] && [ "$INCLUDE_YOLO" != "yolo" ]; then
  echo "用法: $0 [yolo]" >&2
  exit 1
fi

if [ ! -f "$SOURCE" ]; then
  echo "未找到 $SOURCE" >&2
  exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
  echo "未找到目标目录 $TARGET_DIR" >&2
  exit 1
fi

cp "$SOURCE" "$TARGET_DIR/main.py"
if [ "$INCLUDE_YOLO" = "yolo" ]; then
  if [ ! -f "$MODEL_SOURCE" ]; then
    echo "未找到 $MODEL_SOURCE" >&2
    exit 1
  fi
  cp "$MODEL_SOURCE" "$TARGET_DIR/yolo.tflite"
  echo "已复制到 $TARGET_DIR/main.py 和 YOLO 模型"
else
  echo "已复制到 $TARGET_DIR/main.py"
fi
