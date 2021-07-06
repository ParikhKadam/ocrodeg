#!/bin/env python
# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

from __future__ import print_function

from distutils.core import setup  # , Extension, Command

PREREQS = """
numpy
matplotlib
scipy
pyyaml
typer
numpy
matplotlib
scipy
webdataset@git+git://github.com/tmbdev/webdataset.git
""".split()

scripts = """
""".split()

setup(
    name='ocrodeg',
    version='v0.0',
    author="Thomas Breuel",
    description="Document image degradation and augmentation for OCR.",
    packages=["ocrodeg"],
    scripts=scripts,
    install_requires=PREREQS
)
