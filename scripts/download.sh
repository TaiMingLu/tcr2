#!/bin/bash
# Download all data and models needed for this environment.
#
# Estimated total download size: ~4 GB
# Estimated disk usage after extraction: ~4 GB
#
# Models used:
#   - Qwen2.5-1.5B-Instruct: ~3 GB (already in /home/user/shared/models/)
#   - Qwen2.5-0.5B-Instruct: ~1 GB (head selector, paper uses this)
#
# Datasets are synthetically generated - no external downloads needed.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=== TCR Environment Downloads ==="

SHARED_MODELS="/home/user/shared/models"
WORKSPACE_MODELS="/home/user/models"
mkdir -p "$WORKSPACE_MODELS"

# Qwen2.5-1.5B-Instruct (main model - likely already in shared)
if [ -d "$SHARED_MODELS/Qwen2.5-1.5B-Instruct" ]; then
    echo "Qwen2.5-1.5B-Instruct found at $SHARED_MODELS/Qwen2.5-1.5B-Instruct"
    echo "  (Will use this as the main model)"
else
    echo "Downloading Qwen2.5-1.5B-Instruct..."
    huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
        --local-dir "$WORKSPACE_MODELS/Qwen2.5-1.5B-Instruct" \
        --local-dir-use-symlinks False \
        2>&1 | tail -5
fi

# Qwen2.5-0.5B-Instruct (head selector, paper uses this)
if [ -d "$WORKSPACE_MODELS/Qwen2.5-0.5B-Instruct" ]; then
    echo "Qwen2.5-0.5B-Instruct found at $WORKSPACE_MODELS/Qwen2.5-0.5B-Instruct"
elif [ -d "$SHARED_MODELS/Qwen2.5-0.5B-Instruct" ]; then
    echo "Qwen2.5-0.5B-Instruct found at $SHARED_MODELS/Qwen2.5-0.5B-Instruct"
else
    echo "Downloading Qwen2.5-0.5B-Instruct..."
    huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
        --local-dir "$WORKSPACE_MODELS/Qwen2.5-0.5B-Instruct" \
        --local-dir-use-symlinks False \
        2>&1 | tail -5
fi

echo "=== Downloads complete ==="
echo "Model locations:"
ls -d "$SHARED_MODELS"/Qwen2.5-* 2>/dev/null || echo "  (none in shared)"
ls -d "$WORKSPACE_MODELS"/Qwen2.5-* 2>/dev/null || echo "  (none in workspace)"
