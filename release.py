#!/usr/bin/env python3
"""
Release helper script for AI Audio Detector

This script helps with version management and release preparation.
"""

import re
import sys
import subprocess
from pathlib import Path
from typing import Optional


def get_current_version() -> str:
    """Get current version from setup.py"""
    setup_py = Path("setup.py")
    if not setup_py.exists():
        raise FileNotFoundError("setup.py not found")

    content = setup_py.read_text()
    match = re.search(r'version="([^"]+)"', content)
    if not match:
        raise ValueError("Version not found in setup.py")

    return match.group(1)


def update_version(new_version: str) -> None:
    """Update version in setup.py and pyproject.toml"""

    # Update setup.py
    setup_py = Path("setup.py")
    content = setup_py.read_text()
    content = re.sub(r'version="[^"]+"', f'version="{new_version}"', content)
    setup_py.write_text(content)
    print(f"✅ Updated setup.py to version {new_version}")

    # Update pyproject.toml
    pyproject_toml = Path("pyproject.toml")
    if pyproject_toml.exists():
        content = pyproject_toml.read_text()
        content = re.sub(r'version = "[^"]+"', f'version = "{new_version}"', content)
        pyproject_toml.write_text(content)
        print(f"✅ Updated pyproject.toml to version {new_version}")


def validate_version(version: str) -> bool:
    """Validate semantic version format"""
    pattern = r"^\d+\.\d+\.\d+(?:-[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*)?$"
    return bool(re.match(pattern, version))


def run_command(cmd: str) -> bool:
    """Run shell command and return success status"""
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        print(f"Error: {e.stderr}")
        return False


def create_release(version: str, push: bool = False) -> None:
    """Create a new release"""

    if not validate_version(version):
        print(f"❌ Invalid version format: {version}")
        print("Version must follow semantic versioning (e.g., 1.0.0, 1.2.3-beta.1)")
        sys.exit(1)

    current_version = get_current_version()
    print(f"Current version: {current_version}")
    print(f"New version: {version}")

    if current_version == version:
        print("❌ New version is the same as current version")
        sys.exit(1)

    # Update version files
    update_version(version)

    # Run tests
    print("\n🧪 Running tests...")
    if not run_command(
        "/Users/ajpri/Summer/AI_Audio/.venv/bin/python -m pytest tests/ -v"
    ):
        print("❌ Tests failed. Please fix before releasing.")
        sys.exit(1)
    print("✅ All tests passed")

    # Run linting
    print("\n🔍 Running linting...")
    if not run_command(
        "flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics"
    ):
        print("❌ Linting failed. Please fix before releasing.")
        sys.exit(1)
    print("✅ Linting passed")

    # Build package
    print("\n📦 Building package...")
    if not run_command("/Users/ajpri/Summer/AI_Audio/.venv/bin/python -m build"):
        print("❌ Build failed")
        sys.exit(1)
    print("✅ Package built successfully")

    # Check package
    print("\n🔍 Checking package...")
    if not run_command(
        "/Users/ajpri/Summer/AI_Audio/.venv/bin/python -m twine check dist/*"
    ):
        print("❌ Package check failed")
        sys.exit(1)
    print("✅ Package check passed")

    # Git operations
    print(f"\n📝 Creating git commit and tag...")
    if not run_command("git add ."):
        print("❌ Git add failed")
        sys.exit(1)

    if not run_command(f'git commit -m "Release version {version}"'):
        print("❌ Git commit failed")
        sys.exit(1)

    if not run_command(f"git tag -a v{version} -m 'Version {version}'"):
        print("❌ Git tag failed")
        sys.exit(1)

    print(f"✅ Created git tag v{version}")

    if push:
        print("\n🚀 Pushing to remote...")
        if not run_command("git push origin main"):
            print("❌ Git push failed")
            sys.exit(1)

        if not run_command(f"git push origin v{version}"):
            print("❌ Tag push failed")
            sys.exit(1)

        print("✅ Pushed to remote. GitHub Actions will handle PyPI publishing.")
    else:
        print(f"\n✅ Release {version} prepared locally.")
        print("To publish, run:")
        print("  git push origin main")
        print(f"  git push origin v{version}")
        print(
            "\nGitHub Actions will automatically publish to PyPI when the tag is pushed."
        )


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python release.py <version> [--push]")
        print("Example: python release.py 1.0.1")
        print("Example: python release.py 1.1.0 --push")
        sys.exit(1)

    version = sys.argv[1]
    push = "--push" in sys.argv

    create_release(version, push)


if __name__ == "__main__":
    main()
