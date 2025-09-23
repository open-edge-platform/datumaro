# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

# ruff: noqa: E501

import os
import os.path as osp
import re
from distutils.util import strtobool

import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools_rust import Binding, RustExtension


def find_version(project_dir=None):
    if not project_dir:
        project_dir = osp.dirname(osp.abspath(__file__))

    file_path = osp.join(project_dir, "datumaro", "version.py")

    with open(file_path, "r", encoding="utf-8") as version_file:
        version_text = version_file.read()

    # PEP440:
    # https://www.python.org/dev/peps/pep-0440/#appendix-b-parsing-version-strings-with-regular-expressions
    pep_regex = r"([1-9]\d*!)?(0|[1-9]\d*)(\.(0|[1-9]\d*))*((a|b|rc)(0|[1-9]\d*))?(\.post(0|[1-9]\d*))?(\.dev(0|[1-9]\d*))?"
    version_regex = r"__version__\s*=\s*.(" + pep_regex + ")."
    match = re.match(version_regex, version_text)
    if not match:
        raise RuntimeError("Failed to find version string in '%s'" % file_path)

    version = version_text[match.start(1) : match.end(1)]
    return version


CORE_REQUIREMENTS_FILE = "requirements-core.txt"


def parse_requirements(filename=CORE_REQUIREMENTS_FILE):
    with open(filename, "r", encoding="utf-8") as fh:
        return fh.readlines()


CORE_REQUIREMENTS = parse_requirements(CORE_REQUIREMENTS_FILE)
if strtobool(os.getenv("DATUMARO_HEADLESS", "0").lower()):
    CORE_REQUIREMENTS.append("opencv-python-headless")
else:
    CORE_REQUIREMENTS.append("opencv-python")


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

ext_modules = [
    Pybind11Extension(
        "datumaro._capi",
        ["src/datumaro/capi/pybind.cpp"],
        define_macros=[("VERSION_INFO", find_version("./src"))],
        extra_compile_args=["-O3"],
    ),
]

setuptools.setup(
    name="datumaro",
    version=find_version("./src"),
    author="Intel",
    author_email="emily.chun@intel.com",
    description="Dataset Management Framework (Datumaro)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/open-edge-platform/datumaro",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src", include=["datumaro*"]),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=CORE_REQUIREMENTS,
    extras_require={
        # Tensorflow support
        "tf": ["tensorflow"],
        "tfds": ["tensorflow-datasets<4.9.3", "absl-py>=0.12.0"],
        # PyTorch support
        "torch": ["torch", "torchvision"],
        # Kaggle download support
        "kaggle": ["kaggle"],
        # NDR, mpii file format, missing value detection in tabular data
        "scipy": ["scipy"],
        # Tabular data processing
        "nlp": ["nltk", "tokenizers", "portalocker"],
        # Comparator
        "tabulate": ["tabulate"],
        # Any dependencies needed by the CLI
        # 2.3 has an unlisted dependency on PyTorch, which we don't need
        "cli": ["tensorboardX>=1.8,!=2.3", "tabulate", "scipy", "matplotlib>=3.3.1"],
        # Visualizer
        "visualizer": ["matplotlib>=3.3.1"],
        # NYU Depth Dataset v2 file format
        "h5py": ["h5py>=2.10.0"],
        # BraTS file format
        "nibabel": ["nibabel>=3.2.1"],
        # AVA dataset
        "protobuf": ["protobuf"],
        "experimental": ["polars>=1.31.0,<2.0.0"],
    },
    ext_modules=ext_modules,
    entry_points={
        "console_scripts": [
            "datum=datumaro.cli.__main__:main",
        ],
    },
    cmdclass={"build_ext": build_ext},
    include_package_data=True,
    rust_extensions=[
        RustExtension(
            "datumaro.rust_api",
            path=osp.join("rust", "Cargo.toml"),
            debug=False,
            binding=Binding.PyO3,
        ),
    ],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)
