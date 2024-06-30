import os
from pathlib import Path


#home_dir = Path.cwd(),
home_dir = Path(__file__)
print(f'home dir: {home_dir}'),
print(f"parent dir: {home_dir.parent}"),
print(f"parert of parent: {home_dir.parent.parent}")
print(f"new path: {home_dir.parent.parent.joinpath("continuum_nathan", "experiments")}")

os.symlink(__file__, Path(__file__).parent.joinpath("utils", "simlink_test.py"))

