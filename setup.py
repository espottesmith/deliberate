#!/usr/bin/env python

from pathlib import Path

from setuptools import find_packages, setup

module_dir = Path(__file__).resolve().parent
base_dir = module_dir.parent.parent

setup(
    name="deliberate",
    setup_requires=["setuptools"],
    description="",
    long_description=""" """,
    long_description_content_type="text/markdown",
    url="https://github.com/espottesmith/deliberate",
    author="Evan Spotte-Smith",
    author_email="espottesmith@gmail.com",
    license="MIT",
    packages=find_packages("deliberate"),
    package_dir={"": "deliberate"},
    zip_safe=False,
    install_requires=[
        "setuptools",
    ],
    python_requires=">=3.6",
    version='1.0'
)
