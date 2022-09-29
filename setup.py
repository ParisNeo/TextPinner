from pathlib import Path
from typing import Union

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


def read_requirements(path: Union[str, Path]):
    with open(path, "r") as file:
        return file.read().splitlines()


requirements = read_requirements("requirements.txt")
requirements_dev = read_requirements("requirements_dev.txt")

setuptools.setup(
    name="TextPinner",
    version="0.0.7",
    author="Saifeddine ALOUI",
    author_email="aloui.saifeddine@gmail.com",
    description="A python library for pinning a text to one of texts list. Useful for natural commands parsing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ParisNeo/TextPinner",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    extras_require={"dev": requirements_dev},
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
