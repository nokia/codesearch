# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================

import logging
from pathlib import Path

import tensorflow as tf
import numpy as np
from tqdm import trange

from codesearch.utils import Saveable
from codesearch.embedding_retrieval import RetrievalEmbedder
from codesearch.data_config import DESCRIPTION_FIELD

logger = logging.getLogger(__name__)


class TuseEmbedder(RetrievalEmbedder, Saveable):
    
    def __init__(self, use_embedder, batch_size=10, qa_interface=False):
        self.use_embedder = use_embedder
        self.qa_interface = qa_interface
        self.batch_size = batch_size
        self.dim = self.embed_sequences(['test']).shape[1]
        
    def embed_sequences(self, sequences):
        inputs = tf.constant(sequences)
        if not self.qa_interface:
            embedding = self.use_embedder(inputs)
        else:
            embedding = self.use_embedder(inputs)["outputs"]
        return embedding.numpy()
        
    def embed_queries(self, queries):
        logger.info(f"Embedding {len(queries)} queries")
        embs = np.zeros(shape=(len(queries), self.dim), dtype=np.float32)
        for i in trange(0, len(queries), self.batch_size, desc="Embedding queries"):
            embs[i:i + self.batch_size, :] = self.embed_sequences(queries[i:i + self.batch_size])
        return embs
    
    def embed_snippets(self, snippets):
        logger.info(f"Embedding {len(snippets)} snippets")
        embs = np.zeros(shape=(len(snippets), self.dim), dtype=np.float32)
        titles = [s[DESCRIPTION_FIELD].strip() for s in snippets]
        titles = np.array(titles, dtype=np.object)
        for i in trange(0, titles.shape[0], self.batch_size, desc="Embedding snippets"):
            embs[i:i + self.batch_size, :] = self.embed_sequences(titles[i: i + self.batch_size])
        return embs
        
    def save(self, directory):
        super().save(directory)
        directory = Path(directory)
        tf.saved_model.save(self.use_embedder, str(directory/"use"))
        
    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        embedder = tf.saved_model.load(str(directory/"use"))
        return cls(embedder)
    
    @classmethod
    def from_use_checkpoint(cls, use_checkpoint_path):
        embed = tf.saved_model.load(use_checkpoint_path)
        return TuseEmbedder(embed)
            
    