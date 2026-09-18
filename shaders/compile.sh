#!/usr/bin/env bash

set -euo pipefail

SHADER_DIR="$(dirname "$0")"
OUT_DIR="${1:-$SHADER_DIR}"

echo "Compiling shaders..."

glslc "$SHADER_DIR/triangle.vert" -o "$OUT_DIR/triangle.vert.spv"
glslc "$SHADER_DIR/triangle.frag" -o "$OUT_DIR/triangle.frag.spv"

echo "Done."
