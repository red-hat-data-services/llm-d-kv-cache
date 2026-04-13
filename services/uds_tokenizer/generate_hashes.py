#!/usr/bin/env python3
"""Replace hashes in requirements.txt with hashes from the rhoai PEP 503 index.

uv export produces hashes from whichever index it resolved each package from
(often PyPI). But hermeto downloads from the rhoai index, which has rebuilt
wheels with different hashes. This script replaces ALL hashes with rhoai
hashes so hermeto's checksum verification passes.

Also prepends --index-url so hermeto/cachi2 downloads from the rhoai index.

Usage:
    python generate_hashes.py
"""

import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# PEP 503 simple API for the internal rhoai-wheels index (CPU variant).
RHOAI_INDEX = (
    "https://packages.redhat.com/api/pypi/public-rhai/rhoai/3.4/cpu-ubi9/simple"
)

# Prepended to requirements.txt so hermeto downloads from the internal index.
INDEX_URL_HEADER = f"--index-url {RHOAI_INDEX}/\n"


def fetch_pep503_hashes(
    index_url: str, pkg: str, version: str
) -> list[str]:
    """Fetch sha256 hashes for a package version from a PEP 503 simple index.

    Queries the JSON variant (PEP 691) of the simple API to get wheel/sdist
    file entries with their hashes.
    """
    # Normalize package name for PEP 503 (lowercase, replace - and _ with -)
    normalized = re.sub(r"[-_.]+", "-", pkg).lower()
    url = f"{index_url}/{normalized}/?format=json"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception:
        return []

    hashes = []
    for file_info in data.get("files", []):
        filename = file_info.get("filename", "")
        if not _file_matches_version(filename, normalized, version):
            continue
        file_hashes = file_info.get("hashes", {})
        if "sha256" in file_hashes:
            hashes.append(file_hashes["sha256"])

    return hashes


def _file_matches_version(
    filename: str, normalized_pkg: str, version: str
) -> bool:
    """Check if a wheel/sdist filename matches a specific package version.

    Handles both standard filenames and build-tagged filenames (e.g. -2-)
    used by the internal rhoai index.
    """
    pkg_prefix = normalized_pkg.replace("-", "_")
    prefix = f"{pkg_prefix}-{version}"
    # Wheel: {pkg}-{version}-... or {pkg}-{version}-{build}-...
    if filename.endswith(".whl") and filename.startswith(prefix + "-"):
        return True
    # Sdist: {pkg}-{version}.tar.gz
    if filename == prefix + ".tar.gz":
        return True
    return False


def get_rhoai_hashes(pkg: str, version: str) -> tuple[str, list[str]]:
    """Get hashes from the rhoai index for a package version."""
    hashes = fetch_pep503_hashes(RHOAI_INDEX, pkg, version)
    if not hashes:
        print(f"  WARNING: No hashes found for {pkg}=={version}", file=sys.stderr)
    return pkg, sorted(set(hashes))


def parse_requirements(req_path: str) -> list[tuple[str, str]]:
    """Parse requirements.txt, returning (pkg_name, version) tuples."""
    with open(req_path) as f:
        content = f.read()

    results = []
    for line in content.split("\n"):
        m = re.match(r"^([a-zA-Z][a-zA-Z0-9_.-]+)==([^\s;\\]+)", line)
        if m:
            pkg = m.group(1).lower().replace("_", "-")
            version = m.group(2)
            results.append((pkg, version))
    return results


def replace_hashes(req_path: str, pkg_hashes: dict[str, list[str]]) -> int:
    """Replace all hashes in requirements.txt with rhoai index hashes."""
    with open(req_path) as f:
        lines = f.read().split("\n")

    new_lines = []
    replaced = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^([a-zA-Z][a-zA-Z0-9_.-]+)==([^\s;\\]+)", line)
        if m:
            pkg = m.group(1).lower().replace("_", "-")

            # Skip existing hash continuation lines
            # The requirement line may end with " \" if hashes follow
            base_line = line.rstrip(" \\").rstrip()
            j = i + 1
            while j < len(lines):
                stripped = lines[j].strip()
                if stripped.startswith("--hash=") or stripped.startswith("# via"):
                    j += 1
                else:
                    break

            if pkg in pkg_hashes and pkg_hashes[pkg]:
                hash_lines = [
                    f"    --hash=sha256:{h}" for h in pkg_hashes[pkg]
                ]
                new_lines.append(base_line + " \\")
                new_lines.append(" \\\n".join(hash_lines))
                replaced += 1
            else:
                # No rhoai hashes — keep original lines unchanged
                for k in range(i, j):
                    new_lines.append(lines[k])

            i = j
            continue

        new_lines.append(line)
        i += 1

    with open(req_path, "w") as f:
        f.write("\n".join(new_lines))

    return replaced


def prepend_index_url(req_path: str) -> None:
    """Prepend --index-url to requirements.txt for hermeto/cachi2."""
    with open(req_path) as f:
        content = f.read()
    if "--index-url" in content:
        return
    with open(req_path, "w") as f:
        f.write(INDEX_URL_HEADER + content)
    print(f"Prepended --index-url to {req_path}")


def main():
    req_path = "requirements.txt"

    print("Parsing requirements.txt...")
    req_pkgs = parse_requirements(req_path)
    print(f"Found {len(req_pkgs)} packages.")

    print("Fetching rhoai hashes for all packages...")
    pkg_hashes: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(get_rhoai_hashes, pkg, version): pkg
            for pkg, version in req_pkgs
        }
        done = 0
        for future in as_completed(futures):
            pkg, hashes = future.result()
            pkg_hashes[pkg] = hashes
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(req_pkgs)} packages fetched...")

    print(f"Replacing hashes in {req_path}...")
    replaced = replace_hashes(req_path, pkg_hashes)
    print(f"Replaced hashes for {replaced} packages.")

    missing = [pkg for pkg, hashes in pkg_hashes.items() if not hashes]
    if missing:
        print(f"WARNING: {len(missing)} packages have no rhoai hashes: "
              f"{', '.join(missing)}", file=sys.stderr)

    prepend_index_url(req_path)
    print("Done.")


if __name__ == "__main__":
    main()
