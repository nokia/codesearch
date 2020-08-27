# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================


import random

import numpy as np

from codesearch.data import load_train_dataset 


def load_duplicates(duplicates_dataset, seed=42):
    duplicates = load_train_dataset(duplicates_dataset)
    random.seed(seed)
    random.shuffle(duplicates)
    origs = []
    dupls = []
    # descriptions that are duplicates are mapped to the same number
    duplicate_hash = {} 
    for record_id, duplicates_group in enumerate(duplicates):
        orig = duplicates_group[0]
        i = 0
        duplicate_hash[orig] = record_id
        for dupl in duplicates_group[1:]: 
            if orig == dupl: continue
            duplicate_hash[dupl] = record_id
            if i == 0: 
                origs.append(orig)
                dupls.append(dupl)
            i += 1
        
    return origs, dupls, duplicate_hash

def generate_negative_samples(_orig, _dupl, duplicate_hash, all_titles, num_negative, seed):
    t1s, t2s = [], []
    for i in range(num_negative):
        t1 = random.choice(all_titles)
        t2 = random.choice(all_titles)
        while duplicate_hash[t1] == duplicate_hash[t2]:
            t2 = random.choice(all_titles)
        t1s.append(t1)
        t2s.append(t2)
    return zip(t1s, t2s)
    
    
def add_negative_examples(origs, dupls, duplicate_hash, num_negative, seed):
    origs_, dupls_, labels_ = [], [], []
    all_titles = origs + dupls
    for orig, dup in zip(origs, dupls):
        origs_.append(orig)
        dupls_.append(dup)
        labels_.append(1)
        for orig_n, dupl_n in generate_negative_samples(orig, dup, duplicate_hash, all_titles, num_negative, seed):
            origs_.append(orig_n)
            dupls_.append(dupl_n)
            labels_.append(0)
    return np.array(origs_, dtype=np.object), np.array(dupls_, dtype=np.object), np.array(labels_, dtype=np.int64)


def create_data(origs, dupls, duplicate_hash, num_negative, seed=1, split_ratio=0.1):
    random.seed(seed)
    split = int(len(origs) * 0.1)
    
    origs_train, dupls_train = origs[:-split], dupls[:-split]
    origs_valid, dupls_valid = origs[-split:], dupls[-split:]
        
    origs_train, dupls_train, labels_train = add_negative_examples(origs_train, dupls_train, duplicate_hash, num_negative, seed)
    origs_valid, dupls_valid, labels_valid  = add_negative_examples(origs_valid, dupls_valid, duplicate_hash, num_negative, seed)
        
    return ((origs_train, dupls_train), labels_train), ((origs_valid, dupls_valid), labels_valid)