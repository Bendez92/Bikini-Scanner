"""Check that the dependency lockfile satisfies the user-facing requirements.

The lockfile is intentionally more specific than ``requirements.txt``. This
check keeps every direct requirement present and within its declared version
range without requiring the full, heavyweight application environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
LOCKFILE_PATH = REPO_ROOT / "requirements.lock.txt"


def _read_requirements(path: Path) -> list[Requirement]:
    requirements: list[Requirement] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith(("--index-url", "--extra-index-url")):
            continue
        try:
            requirements.append(Requirement(line))
        except InvalidRequirement as error:
            raise ValueError(f"{path.name}:{line_number}: invalid requirement: {raw_line}") from error
    return requirements


def _read_lockfile(path: Path) -> dict[str, Requirement]:
    pins: dict[str, Requirement] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith(("--index-url", "--extra-index-url")):
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement as error:
            raise ValueError(f"{path.name}:{line_number}: invalid requirement: {raw_line}") from error
        specifiers = list(requirement.specifier)
        if len(specifiers) != 1 or specifiers[0].operator != "==":
            raise ValueError(f"{path.name}:{line_number}: expected an exact pin: {raw_line}")
        pins[canonicalize_name(requirement.name)] = requirement
    return pins


def check(requirements_path: Path = REQUIREMENTS_PATH, lockfile_path: Path = LOCKFILE_PATH) -> list[str]:
    requirements = _read_requirements(requirements_path)
    pins = _read_lockfile(lockfile_path)
    mismatches: list[str] = []
    for requirement in requirements:
        name = canonicalize_name(requirement.name)
        pin = pins.get(name)
        if pin is None:
            mismatches.append(f"- {requirement.name}: required {requirement.specifier}; lockfile pin is missing")
            continue
        locked_specifier = next(iter(pin.specifier))
        try:
            locked_version = Version(locked_specifier.version).public
            satisfies = requirement.specifier.contains(locked_version, prereleases=True)
        except InvalidVersion:
            satisfies = False
        if not satisfies:
            mismatches.append(
                f"- {requirement.name}=={locked_specifier.version}\n+ {requirement.name}{requirement.specifier}"
            )
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that requirements.lock.txt satisfies requirements.txt.")
    parser.add_argument(
        "--requirements",
        type=Path,
        default=REQUIREMENTS_PATH,
        help="path to the unpinned requirements file (default: requirements.txt)",
    )
    parser.add_argument(
        "--lockfile",
        type=Path,
        default=LOCKFILE_PATH,
        help="path to the pinned lockfile (default: requirements.lock.txt)",
    )
    args = parser.parse_args()
    try:
        mismatches = check(args.requirements, args.lockfile)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if mismatches:
        print("requirements.lock.txt does not satisfy requirements.txt:")
        print("\n".join(mismatches))
        return 1
    print("OK: requirements.lock.txt satisfies requirements.txt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
