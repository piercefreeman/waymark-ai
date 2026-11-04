#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Emit a GitHub Actions matrix covering every supported Pydantic AI minor release.

Each entry pins the latest stable patch of one minor line, from ``--minimum`` up to the
newest release on PyPI, so CI exercises new Pydantic AI versions as soon as they ship.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

STABLE_RELEASE = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="pydantic-ai-slim")
    parser.add_argument(
        "--minimum",
        default="2.21",
        help="Oldest supported MAJOR.MINOR line. Default: 2.21",
    )
    parser.add_argument(
        "--releases-file",
        type=Path,
        help="Load PyPI JSON release metadata from a local file instead of PyPI.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON output.")
    return parser.parse_args()


def parse_stable_version(version: str) -> tuple[int, int, int] | None:
    match = STABLE_RELEASE.fullmatch(version)
    if match is None:
        return None
    return int(match["major"]), int(match["minor"]), int(match["patch"])


def is_yanked_release(files: Any) -> bool:
    if not isinstance(files, list) or not files:
        return False
    return all(isinstance(file, dict) and file.get("yanked", False) for file in files)


def load_release_map(package: str, releases_file: Path | None) -> dict[str, Any]:
    if releases_file is not None:
        payload = json.loads(releases_file.read_text())
    else:
        with urlopen(f"https://pypi.org/pypi/{quote(package)}/json", timeout=30) as response:
            payload = json.load(response)
    releases = payload.get("releases", payload) if isinstance(payload, dict) else payload
    if not isinstance(releases, dict):
        raise ValueError("Expected a JSON object containing a 'releases' mapping.")
    return releases


def latest_patch_by_minor(
    release_map: dict[str, Any],
    minimum: tuple[int, int],
) -> list[tuple[int, int, int]]:
    latest: dict[tuple[int, int], tuple[int, int, int]] = {}
    for version, files in release_map.items():
        parsed = parse_stable_version(version)
        if parsed is None or parsed[:2] < minimum or is_yanked_release(files):
            continue
        key = parsed[:2]
        if key not in latest or parsed > latest[key]:
            latest[key] = parsed
    return sorted(latest.values(), reverse=True)


def main() -> int:
    args = parse_args()
    try:
        major, minor = (int(part) for part in args.minimum.split("."))
    except ValueError as exc:
        raise ValueError(f"--minimum must be MAJOR.MINOR, got {args.minimum!r}") from exc

    versions = latest_patch_by_minor(
        load_release_map(args.package, args.releases_file),
        (major, minor),
    )
    if not versions:
        raise ValueError(f"No stable {args.package} releases found at or above {args.minimum}.")

    matrix = {
        "include": [
            {
                "label": f"{major}.{minor}",
                "package": args.package,
                "version": f"{major}.{minor}.{patch}",
            }
            for major, minor, patch in versions
        ]
    }
    json.dump(matrix, sys.stdout, indent=2 if args.pretty else None)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from None
