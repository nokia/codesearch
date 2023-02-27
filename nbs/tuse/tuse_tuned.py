#!/usr/bin/env python
# coding: utf-8

# # Model based on the universal sentence encoder

# © 2020 Nokia
# 
# Licensed under the BSD 3 Clause license
# 
# SPDX-License-Identifier: BSD-3-Clause

# In[1]:


#%load_ext autoreload
#%autoreload 2

from pathlib import Path
import time
import os 
import json
import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import Model, layers, Input, optimizers
from tensorflow.keras.layers import InputLayer
from tensorflow.keras import models
import tensorflow_hub as hub

from codesearch.tuse.tuse_embedder import TuseEmbedder
from codesearch.embedding_retrieval import EmbeddingRetrievalModel
from codesearch.evaluation import evaluate, evaluate_and_dump
from codesearch.data import load_snippet_collection, load_eval_dataset, EVAL_DATASETS, eval_datasets_from_regex
from codesearch.data_config import DESCRIPTION_FIELD
from codesearch.duplicates import load_duplicates, create_data

start = time.time()

# In[2]:


duplicate_titles_file = os.environ.get("duplicate_titles", "so-duplicates-pacsv1-train")
snippets_collection = os.environ.get("snippet_collection", "so-ds-feb20")
valid_dataset_name = os.environ.get("valid_dataset", "so-ds-feb20-valid")
test_dataset_pattern = os.environ.get("test_dataset", "so-ds-feb20-test")


neg_samples = int(os.environ.get("num_neg", 5))

#head = json.loads(os.environ.get("head", '{"sim":"dot", "activations": ["relu", "sigmoid"], "loss": "xent"}'))
head = json.loads(os.environ.get("head", '{"sim":"cosine", "activations": ["linear", "sigmoid"], "loss": "xent"}'))
#head = json.loads(os.environ.get("head", '{"sim":"cosine", "activations": ["relu", "sigmoid"], "loss": "xent" }'))
#head = json.loads(os.environ.get("head", '{"sim": "cosine", "activations": ["relu"], "loss": "mse"}'))
output_dir = os.environ.get("output_dir", "pacsv1")
if not Path(output_dir).exists():
    Path(output_dir).mkdir()
dropout = float(os.environ.get("dropout", 0))
lr = float(os.environ.get("lr", 1e-4))

# ## Load snippets

# In[3]:


if valid_dataset_name and valid_dataset_name not in EVAL_DATASETS:
    raise ValueError()
    
test_dataset_names = eval_datasets_from_regex(test_dataset_pattern)
snippets = load_snippet_collection(snippets_collection)


valid_dataset = load_eval_dataset(valid_dataset_name)
test_datasets = [ load_eval_dataset(ds_name) for ds_name in test_dataset_names]

# ## Load duplicate post titles

# In[4]:


origs, dupls, duplicate_hash = load_duplicates(duplicate_titles_file)
print(len(origs), len(dupls))
data_train, data_valid = create_data(origs, dupls, duplicate_hash, neg_samples, seed=42)

# inspect data
(origs_train, dupls_train), labels_train = data_train
list(zip(origs_train[:5], dupls_train[:5], labels_train[:5]))

# ## Training

# In[5]:


now = datetime.datetime.now()
month, day = now.month, now.day
model_name = f"use5-act={'_'.join(head['activations'])}-sim={head['sim']}-negsamples={neg_samples}-lr={lr}-dropout={dropout}-date={month}{day}"
model_dir = Path(output_dir)/model_name
if not model_dir.exists():
    model_dir.mkdir()

# In[6]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) < 1:
    print("No gpus available.")
else: # Set memory growth for gpu 0.
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ### Evaluation

# In[7]:


def get_retrieval_model(embedder, snippets=snippets):
    retrieval_model = EmbeddingRetrievalModel(embedder)
    retrieval_model.add_snippets(snippets)
    return retrieval_model 
    
def eval_embedder(embedder, steps, snippets=snippets, valid_dataset=valid_dataset_name, test_dataset_names=test_dataset_names):
    embedder = TuseEmbedder(embedder, batch_size=512)
    retrieval_model = get_retrieval_model(embedder, snippets)
    config = {"model": model_name, "steps": steps} 
    results = evaluate_and_dump(retrieval_model, config, output_dir, valid_dataset_name, test_dataset_names)
    return results[valid_dataset_name]

# In[8]:



class EvalCallback(callbacks.Callback):
    def __init__(self, steps, embed, i=0, score_to_beat=0.25):
        self.i = i
        self.steps = steps
        self.embed = embed
        self.max_score = score_to_beat
    
    def on_train_batch_begin(self, batch, logs):
        import sys
        if self.i % self.steps == 0:
            results = eval_embedder(self.embed, self.i)
            mrr_score = results["mrr"]
            if mrr_score > self.max_score:
                TuseEmbedder(embed).save(model_dir/f'use_steps={self.i}')
                self.max_score = mrr_score
        self.i += 1

# In[9]:


embed = tf.keras.Sequential(
    hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-large/5", input_shape=[], dtype=tf.string, trainable=True),
    name="embed"
)

# In[10]:


if head["loss"] == "mse":
    loss = tf.keras.losses.MeanSquaredError()
else:
    loss =  tf.keras.losses.BinaryCrossentropy()
    
original = Input(shape=[], dtype=tf.string)
duplicate = Input(shape=[], dtype=tf.string)

orig_emb = embed(original)
duplicate_emb = embed(duplicate)
if dropout:
    dropout_layer = tf.keras.layers.Dropout(dropout)
    orig_emb = dropout_layer(orig_emb)
    duplicate_emb = dropout_layer(duplicate_emb)
sims = tf.keras.layers.dot([orig_emb, duplicate_emb], 1, normalize=(head["sim"] == "cosine"))
classification_head = tf.keras.Sequential()
if len(head["activations"]) == 2:
    classification_head.add(tf.keras.layers.Dense(1, activation=head["activations"][0],input_shape=[1]))
    classification_head.add(tf.keras.layers.Activation(head["activations"][1], input_shape=[1]))
else:
    classification_head.add(tf.keras.layers.Activation(head["activations"][0], input_shape=[1]))
output = classification_head(sims)

model = Model((original, duplicate), output)
model.summary()

model.compile(
    loss=loss,
    optimizer=optimizers.Adam(learning_rate=1e-4),
    metrics=["accuracy"])

# In[11]:


classification_head.set_weights([np.array([[15.]]), np.array([-5.])])

# In[ ]:


csv_logger = CSVLogger(str(model_dir/'log.csv'), append=True, separator=';')
eval_callback = EvalCallback(100, embed)
for i in range(15):
    data_train, data_valid = create_data(origs, dupls, duplicate_hash, neg_samples, seed=i)
    model.fit(data_train[0], data_train[1], epochs=1, batch_size=512, shuffle=False, validation_data=data_valid, callbacks=[csv_logger, eval_callback])

