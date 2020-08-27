# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================


import tempfile
import dill 
import os
from pathlib import Path
import json
import sys
from functools import partial

from codesearch.utils import Saveable 
from codesearch.text_preprocessing import preprocess_text
from codesearch.code_preprocessing import code_tokenization
    
class BasicEncoder(Saveable):
    """
    Only does preprocessing. Use this when mapping tokens to vocabulary indices is
    done by the model itself. This is the case when using fasttext.
    """
    def __init__(self, description_preprocessor=None, code_preprocessor=None, text_preprocessing_params={}, code_preprocessing_params={}):
        self._custom_preprocessors = bool(description_preprocessor or code_preprocessor)
        self._descr_preprocessor = description_preprocessor or partial(preprocess_text, **text_preprocessing_params)
        self._code_preprocessor = code_preprocessor or partial(code_tokenization, **code_preprocessing_params)
        self._text_preprocessing_params = text_preprocessing_params
        self._code_preprocessing_params = code_preprocessing_params
    
    def encode(self, description, code, language):
        return self.encode_description(description), self.encode_code(code, language)
    
    def encode_description(self, description):
        return self._descr_preprocessor(description)
    
    def encode_code(self, code, language):
        return self._code_preprocessor(code, language)
    
    def save(self, directory):
        super().save(directory)
        directory = Path(directory)
        try :
            custom_preprocessors = self._custom_preprocessors
        except:
            custom_preprocessors = True
        if custom_preprocessors: # for backwards compatibility
            with open(directory/"basicencoder.pkl", "bw") as f:
                dill.dump(self, f)
        else:
            kwargs = {
                "text_preprocessing_params": self._text_preprocessing_params, 
                "code_preprocessing_params": self._code_preprocessing_params
            }
            with open(directory/"kwargs.json", "w") as f:
                json.dump(kwargs, f)
        
    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        if (directory/"basicencoder.pkl").exists(): # for backwards compatibility
            with open(directory/"basicencoder.pkl", "rb") as f:
                return dill.load(f)
        with open(directory/"kwargs.json") as f:
            kwargs = json.load(f)
            return cls(**kwargs)
        
    @classmethod
    def from_default_config(cls):
        return cls(preprocess_text, code_tokenization)
        