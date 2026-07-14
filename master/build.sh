#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BOOT_SOURCE="$SCRIPT_DIR/main.py"
RUN_SOURCE="$SCRIPT_DIR/run.py"
MODEL_SOURCE="$ROOT_DIR/yolo/yolo.tflite"
TARGET_DIR="${TARGET_DIR:-/Volumes/NO NAME}"
BOOT_TARGET_PATH="$TARGET_DIR/main.py"
RUN_TARGET_PATH="$TARGET_DIR/run.py"
INCLUDE_YOLO="${1:-}"

if [ "$INCLUDE_YOLO" != "" ] && [ "$INCLUDE_YOLO" != "yolo" ]; then
  echo "用法: $0 [yolo]" >&2
  exit 1
fi

if [ ! -f "$BOOT_SOURCE" ] || [ ! -f "$RUN_SOURCE" ]; then
  echo "未找到主车启动入口或正式运行脚本" >&2
  exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
  echo "未找到目标目录 $TARGET_DIR" >&2
  exit 1
fi

cp "$BOOT_SOURCE" "$BOOT_TARGET_PATH"
cp "$RUN_SOURCE" "$RUN_TARGET_PATH"
if [ "$INCLUDE_YOLO" = "yolo" ]; then
  if [ ! -f "$MODEL_SOURCE" ]; then
    echo "未找到模型文件 $MODEL_SOURCE" >&2
    exit 1
  fi
  cp "$MODEL_SOURCE" "$TARGET_DIR/yolo.tflite"
fi

echo "已上传主车启动入口: $BOOT_TARGET_PATH"
echo "已上传主车正式运行脚本: $RUN_TARGET_PATH"
if [ "$INCLUDE_YOLO" = "yolo" ]; then
  echo "已上传 YOLO 模型: $TARGET_DIR/yolo.tflite"
fi
