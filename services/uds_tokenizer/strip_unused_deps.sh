#!/bin/bash
# Post-install fixup for Konflux builds: register torch stub and strip GPU deps.
set -euo pipefail

SITE=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")

# Register torch stub so vllm imports resolve without PyTorch
cp /app/torch_stub.py "$SITE/torch_stub.py"
echo "import torch_stub" > "$SITE/torch_stub.pth"

# vllm CUDA kernels (~1.1 GB)
find "$SITE" -path '*/vllm/*.so' -delete

# Native-only packages replaced by torch_stub.py import hook
rm -rf "$SITE/opencv_python_headless.libs" "$SITE/cv2"
rm -rf "$SITE/xgrammar"
rm -rf "$SITE/torchvision" "$SITE/torchaudio"

# Unused at runtime by the tokenizer service
rm -rf "$SITE/pip" "$SITE/setuptools"
rm -rf "$SITE/uvloop" "$SITE/cryptography" "$SITE/hf_xet"
rm -rf "$SITE/outlines_core" "$SITE/sentry_sdk" "$SITE/supervisor"
rm -rf "$SITE/compressed_tensors" "$SITE/anthropic"
rm -rf "$SITE/httptools" "$SITE/watchfiles" "$SITE/websockets"
rm -rf "$SITE/fastar" "$SITE/rignore" "$SITE/dill"

# transformers model implementations, keep only the 5 used at import time
MODELS="$SITE/transformers/models"
if [ -d "$MODELS" ]; then
    find "$MODELS" -mindepth 1 -maxdepth 1 -type d \
        ! -name auto ! -name encoder_decoder ! -name gemma3 ! -name siglip ! -name whisper \
        -exec rm -rf {} +
fi

