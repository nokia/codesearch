# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================

import unittest

from codesearch.utils import load_model
from codesearch.embedding_retrieval import EmbeddingRetrievalModel
from codesearch.data import load_snippet_collection, load_eval_dataset, load_train_dataset

class TestPretrainedModel(unittest.TestCase):

    def test_query_retrieval_model(self):
        query = "plot a bar chart"
        snippets = [{
            "id": "1",
            "description": "Hello world",
            "code": "print('hello world')",
            "language": "python"
            }]

        embedding_model = load_model("ensemble-embedder-pacs")
        retrieval_model = EmbeddingRetrievalModel(embedding_model)
        retrieval_model.add_snippets(snippets)
        retrieval_model.query(query)


    def test_embedding(self):
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
        

class TestLoadData(unittest.TestCase):

    def test_load_snippet_collection(self):
        collection_name = "so-ds-feb20"
        snippets = load_snippet_collection(collection_name)

    def test_load_evaluation_data(self):
        queries, query2ids = load_eval_dataset("so-ds-feb20-valid")
        queries, query2ids = load_eval_dataset("codesearchnet-python-valid")


    def test_load_training_data(self):
        duplicate_records = load_train_dataset("so-duplicates-pacs-train")
        filename = load_train_dataset("so-python-question-titles-feb20")

if __name__ == '__main__':
    unittest.main()
