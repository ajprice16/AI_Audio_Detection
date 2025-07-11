#!/usr/bin/env python3
"""Fix E231 formatting issues in Python files."""

import re
import os
import glob


def fix_f_string_formatting(content):
    """Fix missing whitespace after colon in f-strings."""
    # Pattern to match f-strings with format specifiers that need space after colon
    pattern = r"\{([^}]+):([^}. ][^}]*)\}"

    def replace_func(match):
        var_part = match.group(1)
        format_part = match.group(2)
        return f"{{{var_part}: {format_part}}}"

    return re.sub(pattern, replace_func, content)


def process_file(filepath):
    """Process a single Python file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content
        content = fix_f_string_formatting(content)

        if content != original_content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Fixed formatting in: {filepath}")
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """Main function to process all Python files."""
    python_files = []

    # Find all Python files, excluding .venv
    for pattern in ["*.py", "tests/*.py"]:
        python_files.extend(glob.glob(pattern))

    # Also add example_usage.py specifically
    if os.path.exists("example_usage.py"):
        python_files.append("example_usage.py")

    # Remove any files in .venv and remove duplicates
    python_files = list(set([f for f in python_files if not f.startswith(".venv/")]))

    fixed_count = 0
    for filepath in python_files:
        if process_file(filepath):
            fixed_count += 1

    print(f"\nProcessed {len(python_files)} files, fixed {fixed_count} files.")


if __name__ == "__main__":
    main()
