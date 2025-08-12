#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
from pathlib import Path

def main():
    output_dir = Path("./example/output")
    backup_dir = Path("./example/output_backup")
    script_path = Path("./tools/make.sh")

    if not output_dir.exists():
        print(f"Error: Output directory {output_dir} does not exist")
        sys.exit(1)

    if not script_path.exists():
        print(f"Error: Script {script_path} does not exist")
        sys.exit(1)

    print("Creating backup of output directory...")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(output_dir, backup_dir)
    shutil.rmtree(output_dir)

    print(f"Running script: {script_path}")
    try:
        subprocess.run([str(script_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Script execution failed with exit code {e.returncode}")
        sys.exit(1)

    print("Comparing output directories using 'diff -r'...")
    try:
        diff_result = subprocess.run(
            ["diff", "-r", str(backup_dir), str(output_dir)],
            capture_output=True,
            text=True,
            check=False
        )

        if diff_result.returncode == 0:
            print("SUCCESS: No differences found between runs - output is consistent")
            print("Cleaning up backup directory...")
            shutil.rmtree(backup_dir)
            sys.exit(0)
        elif diff_result.returncode == 1:
            print("FAILURE: Differences found between runs:")
            print(diff_result.stdout)
            if diff_result.stderr:
                print("\nDiff stderr:")
                print(diff_result.stderr)
            print(f"\nBackup directory preserved at: {backup_dir}")
            print(f"New output directory at: {output_dir}")
            sys.exit(1)
        else:
            print(f"ERROR: 'diff' command failed with exit code {diff_result.returncode}")
            if diff_result.stdout:
                print("Diff stdout:")
                print(diff_result.stdout)
            if diff_result.stderr:
                print("Diff stderr:")
                print(diff_result.stderr)
            sys.exit(1)

    except FileNotFoundError:
        print("Error: 'diff' command not found. Please ensure it is installed and in your PATH.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during diff comparison: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
