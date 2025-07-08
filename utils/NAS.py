#!/usr/bin/env python3
"""
Recursive rsync script that only syncs files created after July 1st, 2025
"""

import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


def get_file_creation_time(file_path):
    """
    Get file creation time. On Linux, this uses the birth time if available,
    otherwise falls back to modification time.
    """
    try:
        file_stat = os.stat(file_path)
        # Try to get birth time (creation time) - available on some filesystems
        if hasattr(file_stat, "st_birthtime"):
            return datetime.fromtimestamp(file_stat.st_birthtime)
        # Fall back to modification time on Linux systems
        else:
            return datetime.fromtimestamp(file_stat.st_mtime)
    except (OSError, ValueError) as e:
        print(f"Warning: Could not get creation time for {file_path}: {e}")
        return None


def find_files_after_date(source_dir, cutoff_date):
    """
    Find all files in source_dir (recursively) that were created after cutoff_date
    Returns a list of relative paths from source_dir
    """
    source_path = Path(source_dir)
    matching_files = []

    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' does not exist")
        return []

    print(
        f"Scanning {source_dir} for files created after {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}..."
    )

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            creation_time = get_file_creation_time(file_path)

            # make sur the file doesnt end witt .dat or .mat
            if (
                creation_time
                and creation_time > cutoff_date
                and not (
                    file_path.endswith(".dat")
                    or file_path.endswith(".mat")
                    and not "dataset" in file_path
                    and not file_path.endswith(".tfrec")
                )
            ):
                # Get relative path from source directory
                rel_path = os.path.relpath(file_path, source_dir)
                matching_files.append(rel_path)
                # print(
                #     f"Found: {rel_path} (created: {creation_time.strftime('%Y-%m-%d %H:%M:%S')})"
                # )

    return matching_files


def run_rsync_with_file_list(source_dir, dest_dir, file_list, dry_run=False):
    """
    Run rsync with a specific list of files
    """
    if not file_list:
        print("No files to sync.")
        return True

    # Create temporary file with list of files to sync
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".txt"
    ) as temp_file:
        for file_path in file_list:
            temp_file.write(file_path + "\n")
        temp_file_path = temp_file.name

    try:
        # Build rsync command
        rsync_cmd = [
            "rsync",
            "-atP",  # archive mode, verbose
            "--relative",  # preserve directory structure
            "--files-from=" + temp_file_path,
            source_dir,
            dest_dir,
        ]

        if dry_run:
            rsync_cmd.insert(1, "--dry-run")
            print("DRY RUN - No files will actually be copied")

        print("\nRunning rsync command:")
        print(" ".join(rsync_cmd))
        print()

        # Execute rsync
        result = subprocess.run(rsync_cmd, capture_output=True, text=True)

        # print(result.stdout)
        if result.stderr:
            print("Errors/Warnings:", result.stderr)

        return result.returncode == 0

    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)


def run_rsync():
    """
    Main function to handle command line arguments and orchestrate the sync
    """
    # Configuration
    CUTOFF_DATE = datetime(2025, 7, 5, 0, 0, 0)  # July 1st, 2025 00:00:00

    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python rsync_conditional.py SOURCE DESTINATION [--dry-run]")
        print("Example: python rsync_conditional.py /home/user/docs/ /backup/docs/")
        sys.exit(1)

    source_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    dry_run = "--dry-run" in sys.argv
    force = "--force" in sys.argv

    # Validate directories
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist")
        sys.exit(1)

    if not os.path.isdir(source_dir):
        print(f"Error: Source path '{source_dir}' is not a directory")
        sys.exit(1)

    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    print(f"Cutoff Date: {CUTOFF_DATE.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dry Run: {dry_run}")
    print("-" * 50)

    # Find files created after cutoff date
    matching_files = find_files_after_date(source_dir, CUTOFF_DATE)

    if not matching_files:
        print(f"\nNo files found created after {CUTOFF_DATE.strftime('%Y-%m-%d')}")
        return

    print(f"\nFound {len(matching_files)} file(s) to sync:")
    # for file_path in matching_files:
    #     print(f"  {file_path}")

    # Confirm before proceeding (unless dry run)
    if not dry_run and not force:
        response = input(
            f"\nProceed with syncing {len(matching_files)} file(s)? (y/N): "
        )
        if response.lower() not in ["y", "yes"]:
            print("Sync cancelled by user")
            return

    # Run rsync
    success = run_rsync_with_file_list(source_dir, dest_dir, matching_files, dry_run)

    if success:
        print(f"\n{'Dry run completed' if dry_run else 'Sync completed successfully'}")
    else:
        print("\nSync failed - check error messages above")
        sys.exit(1)


if __name__ == "__main__":
    run_rsync()
