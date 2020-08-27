# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================

import subprocess
import os

from setuptools import setup
from setuptools import find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

from codesearch import __version__

def get_requirements():
    requirements = [
        'Click',
        'tree-sitter',
        'spacy',
        'dill', 
        'scikit-learn==0.22.1',
        # this fastText version fixes a memory leak in the current release
        'fastText @ git+https://github.com/facebookresearch/fastText@02c61efaa6d60d6bb17e6341b790fa199dfb8c83',
        'rank-bm25',
        'ipywidgets',
        'pandas',
        "tqdm"
    ]

    if os.environ.get("INSTALL_TF", True):
        requirements.append('tensorflow>=2.0')
        requirements.append('tensorflow_hub')
    if os.environ.get("INSTALL_TORCH", True):
        requirements.append('torch')
    if os.environ.get("INSTALL_FASTAI", True):
        requirements.append('fastai')
    return requirements



setup(
    name='codesearch',
    version=__version__,
    python_requires='>=3.6',
    packages=find_packages(),
    install_requires=get_requirements(),
    entry_points="""
        [console_scripts]
        codesearch=codesearch.cli:cli
    """
)






