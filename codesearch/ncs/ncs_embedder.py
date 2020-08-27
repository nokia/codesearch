# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================

import logging
import pickle
from pathlib import Path
from collections import Counter
from math import log

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tqdm import tqdm

from codesearch.utils import Saveable, SaveableFunction
from codesearch.embedding_retrieval import RetrievalEmbedder, nan_logger
from codesearch.embedding_pretraining import load_fasttext_model
from codesearch.sequence_embedding import BoWEmbedder
from codesearch.data_config import CODE_FIELD, LANGUAGE_FIELD

logger = logging.getLogger(__name__)

def tfidf_preprocessor(s):
    return s    
    
def tfidf_tokenizer(s):
    return s.split(" ")
    

class TfidfCodeEmbedder(Saveable):
    
    def __init__(self, embedding_model, encoder, tfidf_model ):
        self.embedding_model = embedding_model
        self.encoder = encoder
        self.tfidf_model = tfidf_model
        
    def __call__(self, snippets, encode=True):
        logger.info(f"Embedding {len(snippets)} code snippets")
        embs = np.zeros(shape=(len(snippets), self.embedding_model.get_dimension()), dtype=np.float32)
        
        for i, snippet in enumerate(tqdm(snippets, desc="Embedding snippets")):
            language = snippet[LANGUAGE_FIELD]
            code = snippet[CODE_FIELD]
            embs[i, :] = self.embed_code(code, language, encode)
        return embs
    
    def embed_code(self, code, language, encode):
        code_enc = self.encoder.encode_code(code, language)[0] if encode else code
    
        if not code_enc:
            return np.zeros(self.embedding_model.get_dimension())
        else:
            embedder, tfidfer = self.embedding_model, self.tfidf_model 
            tfidf = tfidfer.transform([" ".join(code_enc)])
            code_tokens = list(set(code_enc))
            
            embs_snippet = np.array([embedder.get_word_vector(t) for t in code_tokens])
            tfidf_weights = []
            for t in code_tokens:
                if t in tfidfer.vocabulary_:
                    w = tfidf[0, tfidfer.vocabulary_[t]]
                else:
                    w = code_enc.count(t) * log(20000) # approximate tfidf for OOV token
                tfidf_weights.append(w)
            
            tfidf_weights = np.array(tfidf_weights)
            tfidf_weights = np.expand_dims(tfidf_weights, axis=1)
            embs_snippet_tfidf = embs_snippet * tfidf_weights/ tfidf_weights.sum()
            return embs_snippet_tfidf.sum(axis=0)
        
    def save(self, directory):
        super().save(directory)
        directory = Path(directory)
        self.encoder.save(directory/"encoder")
        self.embedding_model.save_model(str(directory/"fasttext_model.bin"))

        with open(directory/"tfidf", "wb") as f:
            pickle.dump(self.tfidf_model, f)
            
    @classmethod
    def load(cls, directory):
        directory = Path(directory)

        with open(directory/"tfidf", "rb") as f:
            tfidf_model = pickle.load(f)
        
        ft_model, encoder = load_fasttext_model(str(directory))
        return cls(ft_model, encoder, tfidf_model)

    
    @classmethod
    def create_tfidf_model(cls, encoder, embedding_model, snippets):
        core_snippets_enc = []
        for s in tqdm(snippets):
            s_enc = encoder.encode_code(s[CODE_FIELD], s[LANGUAGE_FIELD])[0]
            core_snippets_enc.append(s_enc)
        tfidfer = TfidfVectorizer(token_pattern=".*",
                                  analyzer="word", 
                                  preprocessor=tfidf_preprocessor, 
                                  tokenizer=tfidf_tokenizer)
        
        def docs():
            for core_snippet in core_snippets_enc:
                yield " ".join(core_snippet)
                
        tfidfer = tfidfer.fit(docs())
        
        return tfidfer
        
        
class NcsEmbedder(RetrievalEmbedder, Saveable):
    
    def __init__(self, fasttext_model, encoder, tfidf_model):
        self._ft_model = fasttext_model
        self._enc = encoder
        self._tfidf_model = tfidf_model
        self._query_embedder = BoWEmbedder(SaveableFunction(encoder.encode_description), fasttext_model)
        self._snippet_embedder = TfidfCodeEmbedder(fasttext_model, encoder, tfidf_model)
    @nan_logger   
    def embed_queries(self, queries):
        return self._query_embedder(queries)
    
    @nan_logger
    def embed_snippets(self, snippets):
        return self._snippet_embedder(snippets)
    
    def save(self, directory):
        super().save(directory)
        directory = Path(directory)
        self._ft_model.save_model(str(directory/"fasttext_model.bin"))
        self._enc.save(directory/"encoder")
        with open(directory/"tfidf", "wb") as f:
            pickle.dump(self._tfidf_model, f)
        
    @classmethod
    def load(cls, directory):
        directory = Path(directory)

        with open(directory/"tfidf", "rb") as f:
            tfidf_model = pickle.load(f)
        
        ft_model, encoder = load_fasttext_model(str(directory))
        return cls(ft_model, encoder, tfidf_model)