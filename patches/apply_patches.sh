#!/bin/bash
# Apply xrfeitoria patches to the active Python environment (xrfeitoria).
# Run this once after creating/activating the xrfeitoria environment.
set -e

XRF_ROOT=$(python -c "import xrfeitoria, os; print(os.path.dirname(xrfeitoria.__file__))")
if [ -z "$XRF_ROOT" ]; then
    echo "ERROR: xrfeitoria not found in the current Python environment." >&2
    exit 1
fi
echo "xrfeitoria found at: $XRF_ROOT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cp "$SCRIPT_DIR/xrfeitoria_motion.py" "$XRF_ROOT/utils/anim/motion.py"
cp "$SCRIPT_DIR/xrfeitoria_motion_utils.py" "$XRF_ROOT/utils/anim/utils.py"
cp "$SCRIPT_DIR/xrfeitoria_render.py" "$XRF_ROOT/renderer/renderer_blender.py"

echo "Patches applied successfully."
