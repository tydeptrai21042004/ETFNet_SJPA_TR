"""Setuptools mapping for the split source tree.

The built wheel still exposes the original ``ultralytics.*`` package names.
"""
from pathlib import Path
from setuptools import setup

ROOT = Path(__file__).resolve().parent
SOURCE_ROOTS = (ROOT / "ultralytics", ROOT / "ultra_modeling", ROOT / "ultra_runtime", ROOT / "ultra_services")

packages = []
package_dir = {}
for source_root in SOURCE_ROOTS:
    if source_root.name == "ultralytics":
        init_files = [source_root / "__init__.py", *source_root.rglob("__init__.py")]
        for init_file in init_files:
            if not init_file.exists():
                continue
            rel = init_file.parent.relative_to(source_root)
            name = "ultralytics" + ("." + ".".join(rel.parts) if rel.parts else "")
            packages.append(name)
            package_dir[name] = init_file.parent.relative_to(ROOT).as_posix()
    else:
        for init_file in source_root.rglob("__init__.py"):
            rel = init_file.parent.relative_to(source_root)
            name = "ultralytics." + ".".join(rel.parts)
            packages.append(name)
            package_dir[name] = init_file.parent.relative_to(ROOT).as_posix()

setup(
    packages=sorted(set(packages)),
    package_dir=package_dir,
    py_modules=["etfnet_cli", "repository_layout"],
    include_package_data=True,
    package_data={"ultralytics": ["**/*.yaml"]},
)
