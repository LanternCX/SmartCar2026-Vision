#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/build}"
TOOL_MODULE="tools.chromaforge_export_adapter"

mkdir -p "$BUILD_DIR/master" "$BUILD_DIR/assistant"

cd "$SCRIPT_DIR"

uv run python -m "$TOOL_MODULE" \
  --source master/main.py \
  --output "$BUILD_DIR/master/main.py" \
  --task-constant-name TASKS

uv run python -m "$TOOL_MODULE" \
  --source assistant/main.py \
  --output "$BUILD_DIR/assistant/main.py" \
  --task-constant-name OBJECT_TASKS

echo "已构建主车入口: $BUILD_DIR/master/main.py"
echo "已构建辅车入口: $BUILD_DIR/assistant/main.py"
