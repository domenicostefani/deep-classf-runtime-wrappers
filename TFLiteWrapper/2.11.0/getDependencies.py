#!/usr/bin/env python3

from glob import glob
import os

script_dir = os.path.dirname(os.path.realpath(__file__))

# Do this for all build-* folders
for build_dir in glob(script_dir + "/build-*"):
    print("For build dir: " + build_dir)

    # Look for all .a and .so in any subfolder at any depth
    libpaths = []
    for lib in glob(build_dir + "/**/lib*.a", recursive=True) + glob(build_dir + "/**/lib*.so", recursive=True):
        libpaths.append(lib)

    print("---\nTo Include")
    print("\n".join([str(os.path.basename(l).replace(".a", "").replace(".so", "")[3:]) for l in libpaths]))

    paths = set()
    for lib in libpaths:
        # Get the folder path
        paths.add(os.path.dirname(lib))
    paths = sorted(list(paths))
    print ("---\nTo Link")
    print("\n".join([str(os.path.abspath(l)) for l in paths]))
