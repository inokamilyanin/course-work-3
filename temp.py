import os
import argparse
from pathlib import Path

def get_files_set(directory):
    """Get a set of all files in a directory and its subdirectories."""
    files = set()
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            # Get relative path from the input directory
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, directory)
            files.add(rel_path)
    return files

def compare_directories(dir1, dir2):
    """Compare two directories and return the differences."""
    # Get sets of files from both directories
    files1 = get_files_set(dir1)
    files2 = get_files_set(dir2)
    
    # Find differences
    only_in_dir1 = files1 - files2
    only_in_dir2 = files2 - files1
    
    # Print results
    print(f"\nFiles only in {dir1}:")
    for file in sorted(only_in_dir1):
        print(f"  {file}")
    
    print(f"\nFiles only in {dir2}:")
    for file in sorted(only_in_dir2):
        print(f"  {file}")

def main():
    parser = argparse.ArgumentParser(description='Compare files between two directories')
    parser.add_argument('dir1', help='First directory path')
    parser.add_argument('dir2', help='Second directory path')
    
    args = parser.parse_args()
    
    # Convert to absolute paths and check if directories exist
    dir1 = os.path.abspath(args.dir1)
    dir2 = os.path.abspath(args.dir2)
    
    if not os.path.isdir(dir1):
        print(f"Error: Directory not found: {dir1}")
        return
    
    if not os.path.isdir(dir2):
        print(f"Error: Directory not found: {dir2}")
        return
    
    compare_directories(dir1, dir2)

if __name__ == "__main__":
    main()