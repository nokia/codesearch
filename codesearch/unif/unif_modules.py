# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================

import math 
from pathlib import Path 

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.sparse import EmbeddingBag
import numpy as np
from fasttext import load_model

from codesearch.utils import get_best_device, Saveable


class FastTextEmbeddingBag(EmbeddingBag):
    
    def __init__(self, ft_model, random_init=False):
        self.model = ft_model
        input_matrix = self.model.get_input_matrix()
        input_matrix_shape = input_matrix.shape
        super().__init__(input_matrix_shape[0], input_matrix_shape[1])
        if not random_init:
            # initialize weights
            print("Initializing the weights with fast text matrix")
            self.weight.data.copy_(torch.FloatTensor(input_matrix))
        self.dim = input_matrix_shape[1]

    def forward(self, ind, offsets):
        ind = ind.to(get_best_device())
        offsets = offsets.to(get_best_device())
        return super().forward(ind, offsets)
    
    @classmethod
    def from_file(cls, model_path):
        ft_model = load_model(str(model_path))
        return cls(ft_model)


class Attention1D(nn.Module):
    """
    Attention where the query is a single vector instead of a sequence.
    """
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.minus_inf = torch.tensor([[-float('inf')]], device=get_best_device()) # 1 x 1
        self.to(get_best_device())
        
    def forward(self, query, context, mask):
        """
        query: batch_size x dims
        context: batch_size x time x dims
        mask: batch_size x time
        """
        mask = mask.to(get_best_device())
        batch_size = context.size(0)
        # context: batch_size x time x dims
        # query: batch_size x dims => batch_size x dims x 1
        query = query.unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
        # attn_scores: batch_size x time x 1
        
        attn_scores = torch.where(mask.view(batch_size, -1) != 0., torch.bmm(context, query).squeeze(-1), self.minus_inf)
        
        # attn_weights: batch_size x time
        attn_weights = self.softmax(attn_scores.squeeze(-1))
        # output: batch_size x dims
        output = (attn_weights.unsqueeze(-1) * context).sum(1)
        return output, attn_weights
        
    
class SelfAttention1D(nn.Module):
    """
    Self attention where the time dimension of the query is 1.
    """
    def __init__(self, dim):
        super().__init__()
        self.attn = Attention1D()
        self.query = nn.Parameter(torch.Tensor(dim))
        bound = 1 / math.sqrt(dim)
        nn.init.uniform_(self.query, -bound, bound)
        self.to(get_best_device())

    def forward(self, context, mask):
        """
        context: batch_size x time x dims
        mask: batch_size x time
        """
        return self.attn(self.query, context, mask)


class SelfAttnEmbedder(nn.Module):
    
    def __init__(self, fastTextModel, random_init=False):
        super().__init__()
        self.embedding_bag = FastTextEmbeddingBag(fastTextModel, random_init)
        self.attn = SelfAttention1D(self.embedding_bag.dim)
        self.to(get_best_device())
        
    def forward(self, tokens, mask):
        """
        tokens: list of list of strings: batch_size x time
        mask: batch_size x time
        """
        batch_size, time = mask.size()
        # batch_size * time x dims
        token_embeddings = self.embedding_bag(*tokens)
        # batch_size x time x dims
        token_embeddings = token_embeddings.view(batch_size, time, -1)
        attn_output, attn_weights = self.attn(token_embeddings, mask)
        return attn_output
        
    @property
    def layer_groups(self):
        return [self.embedding_bag, self.attn]
        
    @classmethod
    def init_from_fasttext_file(cls, model_path):
        ft_model = load_model(str(model_path))
        return cls(ft_model)
    
class AverageEmbedder(nn.Module):
    
    def __init__(self, fastTextModel, random_init=False):
        super().__init__()
        self.embedding_bag = FastTextEmbeddingBag(fastTextModel, random_init)
        self.to(get_best_device())
        
    def forward(self, tokens, mask):
        """
        tokens: list of list of strings: batch_size x time
        mask: batch_size x time
        """
        mask = mask.to(get_best_device())
        batch_size, time = mask.size()
        # batch_size * time x dims
        token_embeddings = self.embedding_bag(*tokens)
        token_embeddings = token_embeddings.view(batch_size, time, -1)
        token_embeddings *= mask.unsqueeze(-1)
        return token_embeddings.sum(axis=1)/mask.sum(axis=1).unsqueeze(-1)

    @property
    def layer_groups(self):
        return [self.embedding_bag]
        
    @classmethod
    def init_from_fasttext_file(cls, model_path):
        ft_model = load_model(str(model_path))
        return cls(ft_model)
    
    
class SimilarityModel(nn.Module, Saveable):
    
    def __init__(self, fastTextModel, random_init=False):
        super().__init__()
        self.model = fastTextModel
        self.code_embedder = SelfAttnEmbedder(fastTextModel, random_init)
        self.description_embedder = AverageEmbedder(fastTextModel, random_init)
        self.cosine = nn.CosineSimilarity()
        self.to(get_best_device())

    
    def forward(self, sample):
        code_emb = self.code_embedder(*sample["code"])
        batch_size, emb_dim = code_emb.size()
        code_emb = F.normalize(code_emb, p=2)
        # batch_size * num_descr/code x emb_dim
        descr_emb = self.description_embedder(*sample["descriptions"])
        descr_emb = F.normalize(descr_emb, p=2)
        cosine_sims = torch.einsum('bi,bji->bj', code_emb, descr_emb.view(batch_size, -1, emb_dim))
        return cosine_sims
    
    @property
    def layer_groups(self):
        descr_groups = self.description_embedder.layer_groups
        code_groups = self.code_embedder.layer_groups
        return [ [descr_groups[0], code_groups[0]], code_groups[1:]]

    @classmethod
    def init_from_fasttext_file(cls, model_path):
        ft_model = load_model(str(model_path))
        return cls(ft_model)
    
    
    def save(self, directory):
        super().save(directory)
        directory = Path(directory)
        self.model.save_model(str(directory/"fasttext_model.bin"))
        torch.save(self.state_dict(), str(directory/"similarity_model_state"))
        
    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        model = cls.init_from_fasttext_file(directory/"fasttext_model.bin")
        model.load_state_dict(torch.load(str(directory/"similarity_model_state")))
        return model
    
    
    
    
class MarginRankingLoss(object):
    
    def __init__(self, margin=0.05):
        self.zeros = torch.zeros(1, device=get_best_device())
        self.margin = margin

    def __call__(self, sims, labels):
        labels = labels.type(torch.float32)
        sims = sims.view(labels.size())
        pos_sims = (sims * labels).sum(axis=1)/labels.sum(axis=1)
        neg_sims = (sims * (1.-labels)).sum(axis=1)/(1.-labels).sum(axis=1)
        losses = torch.max(self.zeros, - pos_sims + neg_sims + self.margin)
        return losses.mean()
