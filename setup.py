from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _read_requirements(path: Path):
    reqs = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        reqs.append(line)
    return reqs


setup(
    name="radlab-ml-utils",
    version="1.0.1",
    description="Lightweight utilities to streamline machine learning workflows.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="RadLab team",
    author_email="hello@radlab.dev",
    license="Apache-2.0",
    packages=find_packages(exclude=("tests", "examples")),
    python_requires=">=3.10",
    # No hard dependencies by default: `pip install .` stays lightweight.
    # Optional deps (from requirements.txt) via: `pip install .[deps]`
    extras_require={
        "deps": _read_requirements(ROOT / "requirements.txt"),
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
