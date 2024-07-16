#!/bin/python3
import os
from pathlib import Path

"""
symlink_setup.py
Created: Nathan McAllister - 07/01/2024
Updated: Cameron Wolfe - 07/15/2024

Creates symlinks in all subdirectories of experiments, testing, and setup.

In each directory, the file "symlinks.txt" contains all of the libraries needed
to run the scripts in that directory.  This is then compared with the libraries
in the python folder, and creates a symlink between them.  Run as administrator
on Windows.
"""

# Directories where symlinks should be created
dir_names = ["../experiments/", "../testing/", "../setup/"]


def update_symlinks(directory_path):
    # Get a string of the current directory for printing
    dir_str = "/".join(directory_path.parts[-2:])

    # Folder has symlinks.txt
    if (directory_path / "symlinks.txt").exists():

        # Get files from symlinks.txt
        with open(directory_path / "symlinks.txt") as file:
            symlinks = [symlink.strip() for symlink in file.readlines()]

        # Check if symlinks.txt is empty
        if symlinks and symlinks[0]:

            # Console output containing all symlinks listed
            symlinks_str = "\n    ".join(symlinks)
            print(f"{dir_str}:\n    \033[96m{symlinks_str}\033[0m")

            for symlink in symlinks:
                # Find the location of the symlink in the python folder (source)
                symlink_glob = list(Path.cwd().glob(f"**/{symlink}"))

                # If it is found, create a symlink
                if symlink_glob:
                    if (directory_path / symlink).exists():
                        os.remove(directory_path / symlink)

                    os.symlink(symlink_glob[0], directory_path / symlink)

                else:
                    print(f"\033[91mInvalid file ({symlink}) in symlinks.txt\033[0m")

        else:
            print(f"{dir_str}:\033[93m Empty symlinks.txt\033[0m")

    else:
        print(f"{dir_str}:\033[91m No symlinks.txt\033[0m")


if __name__ == "__main__":

    dir_paths = [Path(x) for x in dir_names]

    subdirectories = []
    for dir_path in dir_paths:
        subdirectories.extend([x for x in dir_path.iterdir() if x.is_dir()])

    for subdirectory in subdirectories:
        update_symlinks(subdirectory)
