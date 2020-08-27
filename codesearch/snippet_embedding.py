# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================


from pathlib import Path

from codesearch.utils import Saveable
from codesearch.data_config import DESCRIPTION_FIELD

class SnippetTitleEmbedder(Saveable):
    
    def __init__(self, title_embedder):
        self.title_embedder = title_embedder
        self.dim = title_embedder("dummy").shape[0]
        
    def __call__(self, snippets):
        titles = [s[DESCRIPTION_FIELD] for s in snippets]
        return self.title_embedder(titles)
        
    def save(self, directory):
        super().save(directory)
        directory = Path(directory)
        self.title_embedder.save(directory/"title_embedder")
        
    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        title_embedder = Saveable.load(directory/"title_embedder")
        return cls(title_embedder)
        
    
