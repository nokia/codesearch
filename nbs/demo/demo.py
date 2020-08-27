# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================


from ipywidgets import VBox, HBox, HTML
from ipywidgets import GridspecLayout
from IPython.core.magic import (Magics, magics_class, line_magic,
                                cell_magic, line_cell_magic)

from codesearch.download import download_model
from codesearch.data import load_snippet_collection
from codesearch.embedding_retrieval import EmbeddingRetrievalModel
from codesearch.utils import load_model
from codesearch.data_config import MODELS, SNIPPET_COLLECTIONS

def create_embedding_retrieval_model(modelname, snippets):
    download_model(modelname)
    embedder = load_model(modelname)
    retrieval_model = EmbeddingRetrievalModel(embedder)
    retrieval_model.add_snippets(snippets)
    return retrieval_model


def preprocess_code(code):
    lines = code.split("\n")
    if len(lines) > 20:
        lines = lines[:20]
        lines.append("\t\t...")
    return "\n".join(lines)

def render_snippet(idx, snippet):
    description = snippet["description"]
    code = snippet["code"]
    
    url = snippet["attribution"][1]
    return VBox([HTML(f"<b>{idx}. {description}</b>"), 
                 HTML(f"<pre><code>{code}</code></pre>"),
                 HTML(f"<a>{url}</a>")
                ])


def render_snippet_tuple(idx, snippets):
    widgets = [render_snippet(idx, snippet) for snippet in snippets]
    return HBox(widgets)

def render_modelnames(modelnames):
    grid = GridspecLayout(1, len(modelnames), height='50px')
    for i, n in enumerate(modelnames):
        grid[0, i] = HTML(f"<h2>{n}</h2>")
    return grid

def render_snippet_results(modelnames, results):
    rows = [render_modelnames(modelnames)]
    num_results = len(results[modelnames[0]])
    header_grid = render_modelnames(modelnames)
    grid = GridspecLayout(num_results, len(modelnames), height='1700px')

    for i in range(num_results):
        for j, name in enumerate(modelnames):
            grid[i, j] = render_snippet(i + 1, results[name][i])
        
    return VBox([header_grid, grid])

@magics_class
class DemoQueryEngine(Magics):
    
    def __init__(self, shell):
        super().__init__(shell)
        self._modelnames = []
        self._retrieval_models = []
        self._snippets = []
        
    @line_magic
    def list_models(self, line):
         return sorted(MODELS)
        
    @line_magic
    def list_snippet_collections(self, line):
        return sorted(SNIPPET_COLLECTIONS)
        
    @line_magic
    def load_snippets(self, snippet_collection):
        self._snippets = load_snippet_collection(snippet_collection)
    
    @line_magic
    def load_model(self, modelname):
        if not self._snippets:
            return HTML("First load a snippet collections with %load_snippets")
        print("Loading model")
        model = create_embedding_retrieval_model(modelname, self._snippets)
        self._retrieval_models.append(model)
        self._modelnames.append(modelname)
        
    @line_magic
    def query(self, *args):
        query = " ".join(args)
        results = {name: m.query(query) for name, m in zip(self._modelnames, self._retrieval_models)}
        return render_snippet_results(self._modelnames, results)
    
    
def load_ipython_extension(ipython):
    """
    Any module file that define a function named `load_ipython_extension`
    can be loaded via `%load_ext module.path` or be configured to be
    autoloaded by IPython at startup time.
    """
    demo = DemoQueryEngine(ipython)
    ipython.register_magics(demo)