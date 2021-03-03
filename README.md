# Code search

This project contains the code to reproduce the experiments in the paper [Neural Code Search Revisited: Enhancing Code Snippet Retrieval through Natural Language Intent](https://arxiv.org/abs/2008.12193). It implements retrieval systems for annotated code snippets: pairs of a code snippet and a short natural language description. Our pretrained models and datasets are hosted on Zenodo (https://zenodo.org/record/4001602). The models and datasets will be downloaded automatically when calling `load_model`, `load_snippet_collection`, etc. (see the code examples below).

In addition, the project also implements some *code-only* retrieval models (BM25, NCS, UNIF) for snippet collections that do not come with descriptions.

The experiments in the paper are done on Python snippets, but the code preprocessing currently also supports java, javascript, and bash.

The project is developed by a research team in the [Application Platforms and Software Systems Lab](https://www.bell-labs.com/our-research/areas/applications-and-platforms/) of [Nokia Bell Labs](https://www.bell-labs.com/). 

## Installation

1. Install the codesearch library: `pip install .`
2. Install the tree-sitter parsers (for preprocessing the code snippets): e.g., `codesearch install_parsers python java` or simply `codesearch install_parsers` to install parsers for all supported languages. By default, parsers are installed under the `codesearch/parsers` directory this can be customized by setting the `TREE_SITTER_DIR` variable.
3. Install spacy (for preprocessing descriptions/code comments): `python -m spacy download en_core_web_md`


## Code structure

```
codesearch
├── codesearch          // Contains the library modules: model code, utilities to download and evaluate models, etc.
├── nbs                 // Contains examples notebooks and notebooks to reproduce the experiments
├── tests               // Contains some unit tests, mostly for verifying the code preprocessing
```

## Models

We provide some pretrained embedding models to create a retrieval system. The pretrained models also expose a consistent interface to embed snippets and queries:

#### Example: Query a snippet collection with a pretrained embedding model

```python
from codesearch.utils import load_model
from codesearch.embedding_retrieval import EmbeddingRetrievalModel

query = "plot a bar chart"
snippets = [{                           # a dummy snippet collection with 1 snippet
    "id": "1",
    "description": "Hello world",
    "code": "print('hello world')",
    "language": "python"
    }]

embedding_model = load_model("use-embedder-pacs")
retrieval_model = EmbeddingRetrievalModel(embedding_model)
retrieval_model.add_snippets(snippets)
retrieval_model.query(query)
```

#### Example: Embed snippets or queries with a pre-trained embedding model

```python
from codesearch.utils import load_model

model_name = "use-embedder-pacs"
queries = ["plot a bar chart"]
snippets = [{
    "description": "Hello world",
    "code": "print('hello world')",
    "language": "python"
    }]

embedding_model = load_model(model_name)
query_embs = embedding_model.embed_queries(queries)
snippet_embs = embedding_model.embed_snippets(snippets)
```

### Available models

Below you find a table with the pretrained models. For each model, we mention based on what information it computes a snippet embedding: the description and/or the code. 

| name                       | inputs             | training data                                          | notebook                    |
|----------------------------|--------------------|--------------------------------------------------------|-----------------------------|
| ncs-embedder-so-ds-feb20      | code               | so-ds-feb20                                            | nbs/ncs/ncs.ipynb           |
| ncs-embedder-staqc-py      | code               | staqc-py-cleaned                              | nbs/ncs/ncs.ipynb           |
| tnbow-embedder-so-ds-feb20 | description        | so-python-question-titles-feb20                        | nbs/tnbow/tnbow.ipynb       |
| use-embedder-pacs          | description        | so-duplicates-pacsv1-train                             | nbs/tuse/tuse_tuned.ipynb   |
| ensemble-embedder-pacs     | description + code | staqc-py-cleaned + so-duplicates-pacs-train | nbs/ensemble/ensemble.ipynb |

## Datasets

This project provides a consistent interface to download and load datasets related to code search.

### Snippet collections

####  Example: Load a snippet collection

```python
from codesearch.data import load_snippet_collection
collection_name = "so-ds-feb20"
snippets = load_snippet_collection(collection_name)
```

#### Available snippet collections
In the table below you find which snippet collections can be loaded. The staqc-py-cleaned, conala-curated, and codesearchnet collections are derived from existing datasets. For staqc-py and conala-curated we did some additional processing, for the codesearchnet collections we merely load the original dataset in a format that is consistent with our code. 

If you were to use any of these datasets in your research, please make sure to cite the respective works.

| name                                          | description                                                                                                                  |
|-----------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| so-ds-feb20                                   | Mined from Python Stack Overflow posts related to data science. Stack Overflow dumps can be found here: https://archive.org/details/stackexchange, [LICENSE](https://creativecommons.org/licenses/by-sa/4.0/)                                                             |
| staqc-py-cleaned                     | Derived from the Python StaQC snippets (additional cleaning was done as decribed in the paper). See https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset, [LICENSE](https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset/blob/master/LICENSE.txt)                               |
| conala-curated                                | Derived from the curated snippets of the CoNaLa benchmark. See https://conala-corpus.github.io/ , [LICENSE](https://creativecommons.org/licenses/by-sa/4.0/)                                                                                         |
| codesearchnet-{language}-{train\|valid\|test} | The CodeSearchNet snippet collections used for training/MRR validation/MRR testing. See https://github.com/github/CodeSearchNet. Licenses of the individial snippets can be found in pkl files.                                           |
| codesearchnet-{language}                      | The CodeSearchNet snippet collections used for the weights and biases benchmark. See https://github.com/github/CodeSearchNet. Licenses of the individial snippets can be found in pkl files. **Note**: not all of these snippets have descriptions |

### Evaluation data
Evaluation datasets link queries to relevant snippets in one of the above snippet collections.


#### Example: load an evaluation dataset
```python
from codesearch.data import load_eval_dataset
queries, query2ids = load_eval_dataset("so-ds-feb20-valid")
```

#### Available evaluation datasets
| name                           | description                                                                     |
|--------------------------------|---------------------------------------------------------------------------------|
| so-ds-feb20-{valid\|test}      | Queries paired to relevant snippets in the so-ds-feb20 snippet collection.      |
| staqc-py-cleaned-{valid\|test} | Queries paired to relevant snippets in the staqc-py-cleaned snippet collection. |
| conala-curated-0.5-test        | Queries paired to relevant snippets in the CoNaLa benchmark                     |


It is also possible to load a snippet collection as evaluation data. The descriptions will be used as queries. Note that this only makes sense to evaluate code-only models (i.e., models that do not use the description field).

#### Example: load a snippet collection as evaluation data
```python
queries, query2ids = load_eval_dataset("codesearchnet-python-valid")
```


### Training data

The different models we implement use different kinds of training data. Code-only models are trained on pairs of code snippets and descriptions. For these models, the snippet collections are used as training data (of course you should never train on a snippet collection when you intent to use that load that collection as evaluation data as well). The USE model is fine-tuned on titles of duplicate Stack Overflow posts. You can take a look our notebooks (e.g., nbs/ncs/ncs.ipynb, nbs/tuse/tuse_tuned) to find out how the training is done/how the training data is loaded.

To download and load the title pairs from Stack Overflow duplicate posts run:

```python
from codesearch.data import load_train_dataset
duplicate_records = load_train_dataset("so-duplicates-pacs-train")
```

These duplicate records have been filtered to ensure that there is no overlap with the `so-ds-feb20` and `staqc-py` evaluation datasets.

To download a text file with Stack Overflow post titles tagged with Python (used for the TNBOW baseline) run: 

```python
from codesearch.data import load_train_dataset
filename = load_train_dataset("so-python-question-titles-feb20")
```

## Demo notebook

 You can run the demo notebook `nbs/demo/demo.ipynb` to quickly try out any of the pretrained models on one of the snippet collections.

## Benchmark on PACS

To replicate the results of our paper or evaluate your own model on the PACS benchmark, have a look at `nbs/evaluate.ipynb` and `codesearch/benchmark.ipynb`. A custom embedding model class should implement the `embed_snippets` and `embed_queries` functions (similar to `codesearch/tuse/tuse_embedder.py`, `codesearch/tnbow/tnbow_embedder.py`, `codesearch/ncs/ncs_embedder.py` etc.).

#### Example: Benchmark a model on PACS

```python
from codesearch.benchmark import benchmark_on_pacs

benchmark_on_pacs(
    model_path=model_path, # one of the pretrained model names or a path to a model that can be loaded with `codesearch.utils.load_model`
    output_dir=output_dir
)
```
