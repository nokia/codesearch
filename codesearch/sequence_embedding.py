# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================


import numpy as np
from pathlib import Path
import os

from tqdm import tqdm

from codesearch.utils import Saveable
from codesearch.embedding_pretraining import load_fasttext_model
    
    
class BoWEmbedder(Saveable):
    
    def __init__(self, encoder, ft_model):
        self.encoder = encoder
        self.ft_model = ft_model
        
    def __call__(self, sequences):
        embs = np.zeros(shape=(len(sequences), self.ft_model.get_dimension()), dtype=np.float32)
        for i, sequence in enumerate(tqdm(sequences, desc="Embedding sequences")):
            sequence_enc = self.encoder(sequence)
            if not sequence_enc: continue
            embs_seq = np.array([self.ft_model.get_word_vector(t) for t in sequence_enc])
            embs[i, :] = embs_seq.sum(axis=0)/len(sequence_enc)  
            
        return embs  
    
    def save(self, path):
        super().save(path)
        path = Path(path)
        self.encoder.save(path/"encoder")
        self.ft_model.save_model(str(path/"fasttext_model.bin"))
    
    @classmethod
    def load(cls, path):
        ft_model, encoder = load_fasttext_model(path)
        return cls(encoder, ft_model)
    
    
