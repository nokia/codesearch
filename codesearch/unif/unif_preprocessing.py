# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================

import copy

import torch
import numpy as np

from codesearch.utils import get_best_device

def pad_sequences(sequences, max_len, padding_symbol ='<pad>'):
    max_len_batch = max(len(s) for s in sequences)
    max_len = min(max_len, max_len_batch)
    
    sequences = [list(s) for s in sequences]
    for s in sequences:
        while len(s) < max_len:
            s.append(padding_symbol)
        while len(s) > max_len:
            s.pop(-1)
    return sequences


def extract_mask(padded_sequences, padding_symbol='<pad>'):
    batch_size = len(padded_sequences)
    seq_len = len(padded_sequences[0])
    mask = np.ones(shape=(batch_size, seq_len))
    for i in range(batch_size):
        for j in range(seq_len):
            if padded_sequences[i][j] == padding_symbol:
                mask[i, j] = 0.
    return torch.tensor(mask, dtype=torch.float32)#, device=get_best_device())
 

class Padder(object):
    
    def __init__(self, max_code_len, max_description_len, padding_symbol="<pad>", ft_model=None):
        self.max_code_len = max_code_len
        self.max_descr_len = max_description_len
        self.padding_symbol = padding_symbol
        self.ft_model = ft_model
        
    def _pad_and_mask(self, sequences, max_len):        
        sequences_padded = pad_sequences(sequences, max_len, padding_symbol=self.padding_symbol)
        sequence_masks = extract_mask(sequences_padded, self.padding_symbol)
        if self.ft_model:
            # tuple of subwords and indices
            # should only be done when FastTextEmbedder is used
            sequences_padded = fasttext_preprocess([t for s in sequences_padded for t in s], self.ft_model)
        return sequences_padded, sequence_masks
    
    def _valid_sample(self, s):
        """
        Filter out samples that for which all code/description tokens 
        were removed after preprocessing.
        """
        return s["code"] and all(s["descriptions"])
    
    def __call__(self, samples):            
        code = [x["code"] for x, _ in samples if self._valid_sample(x)]
        code_padded, code_mask = self._pad_and_mask(code, self.max_code_len)
        descriptions = [d for x, _ in samples if self._valid_sample(x) for d in x["descriptions"]]
        descriptions_padded, descriptions_mask = self._pad_and_mask(descriptions, self.max_descr_len)
        x = {
            "code": (code_padded, code_mask),
            "descriptions": (descriptions_padded, descriptions_mask)
            }
        y = torch.stack([y for x, y in samples if self._valid_sample(x)])
        return x, y
        
    
def fasttext_preprocess(words, ft_model):
    word_subinds = np.empty([0], dtype=np.int64)
    word_offsets = [0]
    for word in words:
        _, subinds = ft_model.get_subwords(word)
        word_subinds = np.concatenate((word_subinds, subinds))
        word_offsets.append(word_offsets[-1] + len(subinds))
    word_offsets = word_offsets[:-1]
    ind = torch.tensor(word_subinds, dtype=torch.long)#, device=get_best_device())        
    offsets = torch.tensor(word_offsets, dtype=torch.long)#, device=get_best_device())
    return ind, offsets
