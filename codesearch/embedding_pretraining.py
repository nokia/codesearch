# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================


import os
from pathlib import Path
import logging

from fasttext import load_model
import fasttext
from tqdm import tqdm

from codesearch.utils import Saveable
from codesearch.data_config import DESCRIPTION_FIELD, CODE_FIELD, LANGUAGE_FIELD


logger = logging.getLogger(__name__)

def load_fasttext_model(model_dir):
    model_dir = Path(model_dir)
    enc_path = model_dir/"encoder"
    enc = Saveable.load(enc_path)
    ft_model_file = model_dir/"fasttext_model.bin"
    ft_model = load_model(str(ft_model_file))
    return ft_model, enc


def train_fasttext_model_from_snippets(snippets, encoder, zip_fn, hyperparams, model_dir, save=True):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        os.makedirs(model_dir)
    descriptions_and_code = [(s[DESCRIPTION_FIELD], s[CODE_FIELD], s[LANGUAGE_FIELD]) for s in snippets]
    input_file = str(model_dir/"input.txt")
    create_input_file_from_snippets(input_file, descriptions_and_code, encoder, zip_fn)
    return train_fasttext_model_from_text(input_file, encoder, hyperparams, model_dir, save)

def train_fasttext_model_from_text(input_file, encoder, hyperparams, model_dir, save=True):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        os.makedirs(model_dir)
    logger.info(f"Training skipgram fasttext model with the following hyper-param overrides {hyperparams}")
    model = fasttext.train_unsupervised(str(input_file), model='skipgram', **hyperparams)
    if save:
        encoder.save(model_dir/"encoder")
        model.save_model(str(model_dir/"fasttext_model.bin"))
    return model


def create_input_file_from_text(output_file, text_file, encoder):
    with open(text_file, "r") as f1:
        num_lines = sum(1 for l in f1)
        
    with open(text_file, "r") as f1:
        logger.info(f"Preprocessing {text_file} and writing output to {output_file}.")
        with open(output_file, "w") as f2:
            for l in tqdm(f1, total=num_lines): 
                descr_tokens = encoder.encode_description(l)
                f2.write(" ".join(descr_tokens))
                f2.write("\n")
    

def create_input_file_from_snippets(filename, descriptions_and_code, encoder, zip_fn):
    num_encoding_errors = 0
    with open(filename, "w") as f:
        i = 0
        for descr, code, language in descriptions_and_code:
            try:
                descr_tokens, (code_tokens, _) = encoder.encode(descr, code, language)
            except Exception as e:
                logger.error(f"Error when encoding snippet with description {descr}")
                logger.error(e)
                num_encoding_errors += 1
                continue
            
            f.write(zip_fn(descr_tokens, code_tokens))
            i += 1
    logger.info(f"Finished dumping tokens. {num_encoding_errors} pairs triggered an encoding error.")
    
    
def zip_descr_end(descr_tokens, code_tokens):
    return f"{' '.join(code_tokens)} {' '.join(descr_tokens)}\n"


def zip_descr_start(descr_tokens, code_tokens):
    return f"{' '.join(descr_tokens)} {' '.join(code_tokens)}\n"


def zip_descr_start_end(descr_tokens, code_tokens):
    return zip_descr_start(descr_tokens, code_tokens) + zip_descr_end(descr_tokens, code_tokens)


def zip_descr_middle(descr_tokens, code_tokens):
    middle = len(code_tokens)//2
    return f"{' '.join(code_tokens[:middle])} {' '.join(descr_tokens)} {' '.join(code_tokens[middle:])}\n"


def zip_descr_middle_and_start_end(descr_tokens, code_tokens):
    middle_zip = zip_descr_middle(descr_tokens, code_tokens)
    start_end_zip = zip_descr_start_end(descr_tokens, code_tokens)
    return middle_zip + start_end_zip