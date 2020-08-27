# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================


import re
import time 
from threading import Thread
from traceback import print_exc
import spacy

from codesearch.stopwords import stopwords

def async_fn(f):
    def wrapper(*args, **kwargs):
        thr = Thread(target = f, args = args, kwargs = kwargs)
        thr.setDaemon(True)
        thr.start()
    return wrapper

_nlp = None
_initializing_nlp = False

@async_fn
def init_nlp():
    try:
        global _nlp, _initializing_nlp
        if _initializing_nlp: return
        _initializing_nlp = True
        nlp = spacy.load("en_core_web_md", vectors=False, disable=['parser', 'ner', 'tagger'])
        for word in nlp.vocab:
            word.is_stop = False

        for word in stopwords:
            t = nlp.vocab[word]
            t.is_stop = True
            t = nlp.vocab[word.capitalize()]
            t.is_stop = True
        _nlp = nlp
        print("\nInitialized spacy nlp")
    except:
        print_exc()
        _nlp = -1
    

def nlp():
    i = 0
    progress_symbols = ["|", "/", "─", "\\"]
    while _nlp is None:
        if not _initializing_nlp:
            init_nlp()
        print(f"\rInitializing spacy nlp {progress_symbols[i % 4]}", end="")
        time.sleep(1)
        i += 1
    return _nlp

def preprocess_text(text, lemmatize=True, remove_stop=True, clean_howto=True):
    if clean_howto:
        text = clean_how_to(text)
    tokens = nlp()(text)
    
    if remove_stop:
        tokens = [t for t in tokens if not t.is_stop and str(t) not in  {"#", "//", "/**", "*/"}]
    else:
        tokens = [t for t in tokens if str(t) not in  {"#", "//", "/**", "*/"} ]
    tokens = [t for t in tokens if not str(t).isspace()]
    if lemmatize:
        tokens = [t.lemma_.lower().strip() for t in tokens]
    else:
        tokens = [str(t).lower().strip() for t in tokens]
    return tokens

def compute_overlap(q, d):
    q_toks = set(t.lemma_.lower() for t in nlp()(q) if not t.is_stop)
    d_toks = set(t.lemma_.lower() for t in nlp()(d) if not t.is_stop)    
    return len(q_toks & d_toks), len(q_toks & d_toks)/(len(q_toks))

how_to_pattern = "^([hH]ow to |[hH]ow do ([Ii] |you )|[Hh]ow does one |([tT]he )?[Bb]est way to |([Hh]ow )?[Cc]an (you |[Ii] ))"
def is_how_to(t):
    return re.match(how_to_pattern, t)


def clean_how_to(t):
    t = re.sub(how_to_pattern, "", t)
    if t.endswith("?"):
        t = t[:-1]
    return t[0].capitalize() + t[1:]