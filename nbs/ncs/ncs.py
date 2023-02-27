#!/usr/bin/env python
# coding: utf-8

# © 2020 Nokia
# 
# Licensed under the BSD 3 Clause license
# 
# SPDX-License-Identifier: BSD-3-Clause

# ## Setup

# In[ ]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

import os
import json
import time
import numpy as np
import sys

from codesearch.encoders import BasicEncoder
from codesearch import embedding_pretraining
from codesearch.embedding_pretraining import train_fasttext_model_from_snippets, load_fasttext_model
from codesearch.utils import SaveableFunction
from codesearch.data import load_snippet_collection, EVAL_DATASETS, SNIPPET_COLLECTIONS, eval_datasets_from_regex
from codesearch.ncs.ncs_embedder import TfidfCodeEmbedder, NcsEmbedder
from codesearch.evaluation import evaluate_and_dump 
from codesearch.embedding_retrieval import EmbeddingRetrievalModel
start = time.time()
print(f"start={start}")

# Read configuration parameters from environment variables (when this notebook is run as a script).

# In[ ]:


fast_text_checkpoint = 'ncs_pls_work'# os.environ.get("fast_text_checkpoint", None)
model_filename = "ncs_work_more" #os.environ.get("model_filename", None)

snippets_collection = os.environ.get("snippets_collection", "so-ds-feb20")
train_snippets_collection = os.environ.get("train_snippets_collection", "so-ds-feb20")
valid_dataset = os.environ.get("valid_dataset", None)
test_dataset = os.environ.get("test_dataset", "so-ds-feb20-test")

text_overrides = json.loads(os.environ.get("text_overrides", "{}"))
code_overrides = json.loads(os.environ.get("code_overrides", "{}"))
fast_text_overrides = json.loads(os.environ.get("fast_text_overrides", "{}"))
zip_fn_name = os.environ.get("zip_fn", "zip_descr_end")
output_dir = os.environ.get("output_dir", ".")


# In[ ]:


print(f"model_filename={model_filename} fast_text_checkpoint={fast_text_checkpoint}")


# In[ ]:


print(f"snippets_collection={snippets_collection}")


# In[ ]:


print(f"text_overrides={text_overrides}\n code_overrides={code_overrides}\n fast_text_overrides={fast_text_overrides} zip_fn_name={zip_fn_name}")


# ## Load data

# In[ ]:


if valid_dataset and valid_dataset not in EVAL_DATASETS and valid_dataset not in SNIPPET_COLLECTIONS:
    raise ValueError()
test_datasets = eval_datasets_from_regex(test_dataset)
snippets = load_snippet_collection(snippets_collection)
train_snippets = load_snippet_collection(train_snippets_collection) 


# ## Train or load embedding model

# In[ ]:


if fast_text_checkpoint:
    model, enc = load_fasttext_model(fast_text_checkpoint)
    print("Loaded fast text checkpoint")

else:
    enc = BasicEncoder(text_preprocessing_params=text_overrides, code_preprocessing_params=code_overrides)
    zip_fn = getattr(sys.modules[embedding_pretraining.__name__], zip_fn_name)
    model = train_fasttext_model_from_snippets(train_snippets, enc, zip_fn, fast_text_overrides, "./", save=True)


# ## Unsupervised retrieval baseline

# A first baseline that computes a representation a snippet representation as a tfidf weighted average of their embeddings and a query representation by averaging all terms.

# ### Embedding code & queries

# In[ ]:


tfidf_model = TfidfCodeEmbedder.create_tfidf_model(enc, model, snippets)
embedder = NcsEmbedder(model, enc, tfidf_model)


# ### Create retrieval model

# In[ ]:


retrieval_model = EmbeddingRetrievalModel(embedder)
retrieval_model.add_snippets(snippets)
#
#
## In[ ]:
#
#
#if model_filename: embedder.save(model_filename)


# ## Evaluation

# In[ ]:

#retrieval_model =
sample_queries = ["plot x vs y, label them using x-y in the legend"]
config = {"text": text_overrides, "code": code_overrides, "fasttext": fast_text_overrides}
evaluate_and_dump(
    retrieval_model, 
    config, 
    output_dir, 
    valid_dataset, 
    test_datasets,
    sample_queries=sample_queries
)


# In[ ]:


duration = time.time() - start
f"Running the notebook took {duration} seconds"

