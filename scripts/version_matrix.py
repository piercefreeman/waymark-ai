#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["click>=8.1"]
# ///
"""Emit a GitHub Actions matrix covering the supported minor releases of a PyPI package.

Each entry pins the latest stable patch of one minor line, from ``--minimum`` up to the
newest release on PyPI (optionally only the newest ``--limit`` lines), so CI exercises new
dependency versions as soon as they ship.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

import click

STABLE_RELEASE = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")
MINOR_LINE = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)$")


def parse_stable_version(version: str) -> tuple[int, int, int] | None:
    match = STABLE_RELEASE.fullmatch(version)
    if match is None:
        return None
    return int(match["major"]), int(match["minor"]), int(match["patch"])


def parse_minor_line(
    _ctx: click.Context,
    _param: click.Parameter,
    value: str,
) -> tuple[int, int]:
    match = MINOR_LINE.fullmatch(value)
    if match is None:
        raise click.BadParameter(f"expected MAJOR.MINOR, got {value!r}")
    return int(match["major"]), int(match["minor"])


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
        raise click.ClickException("Expected a JSON object containing a 'releases' mapping.")
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


@click.command(help=__doc__)
@click.option("--package", required=True, help="PyPI distribution name.")
@click.option(
    "--minimum",
    required=True,
    callback=parse_minor_line,
    help="Oldest supported MAJOR.MINOR line.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    help="Only include the newest N minor lines.",
)
@click.option(
    "--releases-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Load PyPI JSON release metadata from a local file instead of PyPI.",
)
@click.option("--pretty", is_flag=True, help="Pretty-print the JSON output.")
def main(
    package: str,
    minimum: tuple[int, int],
    limit: int | None,
    releases_file: Path | None,
    pretty: bool,
) -> None:
    versions = latest_patch_by_minor(load_release_map(package, releases_file), minimum)[:limit]
    if not versions:
        raise click.ClickException(
            f"No stable {package} releases found at or above {minimum[0]}.{minimum[1]}."
        )

    matrix = {
        "include": [
            {
                "label": f"{major}.{minor}",
                "package": package,
                "version": f"{major}.{minor}.{patch}",
            }
            for major, minor, patch in versions
        ]
    }
    click.echo(json.dumps(matrix, indent=2 if pretty else None))


if __name__ == "__main__":
    main()
