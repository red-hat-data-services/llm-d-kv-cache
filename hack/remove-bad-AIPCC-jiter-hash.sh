#!/usr/bin/env bash
# Removes the bad AIPCC jiter hash from the uds_tokenizer requirements.txt file.
# This is a temporary workaround until AIPCC fixes the jiter source dist.
# Tracking ticket: https://redhat.atlassian.net/browse/AIPCC-13850
# Usage: hack/remove-bad-AIPCC-jiter-hash.sh [--file <path>]
# Deps: git, perl

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
REQUIREMENTS_FILE="${REPO_ROOT}/services/uds_tokenizer/requirements.txt"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --file)
            REQUIREMENTS_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: hack/remove-bad-AIPCC-jiter-hash.sh [--file <path>]" >&2
            exit 1
            ;;
    esac
done
HASH="f2839f9c2c7e2dffc1bc5929a510e14ce0a946be9365fd1219e7ef342dae14f4"

if ! grep -q "sha256:${HASH}" "${REQUIREMENTS_FILE}"; then
    echo "Hash not found in ${REQUIREMENTS_FILE}, nothing to do."
    exit 0
fi

perl -0777 -pi -e "s/ \\\\\n    --hash=sha256:${HASH}//g" "${REQUIREMENTS_FILE}"

echo "Removed hash ${HASH} from ${REQUIREMENTS_FILE}"
