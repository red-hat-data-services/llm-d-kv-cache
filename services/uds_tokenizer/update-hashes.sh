#!/usr/bin/env bash
set -euo pipefail

# Recreate venv
uv venv --python 3.12 --clear
source .venv/bin/activate

# Lock and export production requirements
uv lock
uv export --format requirements-txt --no-extra test --output-file requirements.txt
../../hack/remove-bad-AIPCC-jiter-hash.sh

# Export test requirements (production + test extras)
uv export --format requirements-txt --extra test --output-file requirements-test.txt
../../hack/remove-bad-AIPCC-jiter-hash.sh --file requirements-test.txt

deactivate
