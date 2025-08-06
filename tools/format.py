#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

try:
    root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
except subprocess.CalledProcessError:
    print("Not in a git repository", file=sys.stderr)
    sys.exit(1)

result = subprocess.run(
    ["git", "ls-files", "--cached", "--exclude-standard", "--", root],
    capture_output=True, text=True, check=True
)

files = [f for f in result.stdout.splitlines() if f.endswith(('.c', '.cpp', '.h'))]

if files:
    subprocess.run(["clang-format", "-i"] + files, check=True)
