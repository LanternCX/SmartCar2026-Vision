#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$SCRIPT_DIR/main.py"
MODEL_SOURCE="$SCRIPT_DIR/yolo.tflite"
TARGET_DIR="${TARGET_DIR:-/Volumes/NO NAME}"

if [ ! -f "$SOURCE" ]; then
  echo "未找到 $SOURCE" >&2
  exit 1
fi

if [ ! -f "$MODEL_SOURCE" ]; then
  echo "未找到 $MODEL_SOURCE" >&2
  exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
  echo "未找到目标目录 $TARGET_DIR" >&2
  exit 1
fi

cp "$SOURCE" "$TARGET_DIR/main.py"
cp "$MODEL_SOURCE" "$TARGET_DIR/yolo.tflite"
echo "已复制到 $TARGET_DIR/main.py 和 YOLO 模型"
