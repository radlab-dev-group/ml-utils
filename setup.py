from pathlib import Path
from setuptools import setup, find_packages

README = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="radlab-ml-utils",
    version="0.0.1",
    description="Lightweight utilities to streamline machine learning workflows.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="RadLab team",
    license="Apache-2.0",
    author_email="pawel@radlab.dev",
    packages=find_packages(exclude=("tests", "examples")),
    python_requires=">=3.10",
    install_requires=[
        "wandb>=0.15",
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
