#!/usr/bin/env python3
"""
Setup script for AI Audio Detector testing environment
"""
import subprocess
import sys
from pathlib import Path


def run_command(command, description=""):
    """Run a shell command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description or command}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def main():
    """Main setup and test execution"""
    print("AI Audio Detector - Testing Environment Setup")
    print("=" * 60)

    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)

    print(f"Python version: {sys.version}")

    # Install development dependencies
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing core dependencies"),
        ("pip install -r requirements-dev.txt", "Installing development dependencies"),
    ]

    for command, description in commands:
        if not run_command(command, description):
            print(f"Failed to execute: {description}")
            sys.exit(1)

    # Install pre-commit hooks
    print("\nSetting up pre-commit hooks...")
    if not run_command("pre-commit install", "Installing pre-commit hooks"):
        print("Warning: Failed to install pre-commit hooks")

    # Run initial tests
    print("\nRunning initial test suite...")
    test_commands = [
        (
            "python -c \"from ai_audio_detector import AIAudioDetector; print('Import successful')\"",
            "Testing imports",
        ),
        (
            "flake8 ai_audio_detector.py --count --select=E9,F63,F7,F82 --show-source",
            "Basic syntax check",
        ),
        (
            "pytest tests/test_feature_extractor.py -v",
            "Running feature extractor tests",
        ),
        ("pytest tests/test_detector.py -v", "Running detector tests"),
    ]

    for command, description in test_commands:
        success = run_command(command, description)
        if not success and "pytest" in command:
            print(f"Warning: {description} failed, but continuing...")

    # Final setup verification
    print("\n" + "=" * 60)
    print("SETUP VERIFICATION")
    print("=" * 60)

    # Check if required files exist
    required_files = [
        "requirements.txt",
        "requirements-dev.txt",
        ".pre-commit-config.yaml",
        "pytest.ini",
        ".flake8",
        "tests/conftest.py",
        "Makefile",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"Warning: Missing files: {', '.join(missing_files)}")
    else:
        print("✓ All configuration files present")

    # Test make commands
    make_commands = [
        "make help",
        "make lint",
        "make format",
    ]

    for cmd in make_commands:
        success = run_command(cmd, f"Testing {cmd}")
        if success:
            print(f"✓ {cmd} working")
        else:
            print(f"✗ {cmd} failed")

    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("Next steps:")
    print("1. Run 'make test' to run the full test suite")
    print("2. Run 'make quick-check' for fast development checks")
    print("3. Run 'make full-check' before committing code")
    print("4. Check '.github/workflows/ci.yml' for CI configuration")
    print("\nFor more information, see TESTING.md")


if __name__ == "__main__":
    main()
