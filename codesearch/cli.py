# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================

import click

from codesearch.download import download_model
from codesearch.download import download_dataset
from codesearch.data_config import register_model, register_training_dataset, register_snippet_collection, register_eval_dataset
from codesearch.install_parsers import install_parsers


@click.group()
def cli():
    pass

@click.group()
def download():
    pass

@click.command(name="model")
@click.argument("model_name")
def download_model_(model_name):
    download_model(model_name)

@click.command(name="dataset")
@click.argument("dataset_name")
def download_dataset_(dataset_name):
    download_dataset(dataset_name)


@click.group()
def register():
    pass

@click.command(name="model")
@click.argument("model_name")
@click.argument("filename")
@click.argument("model_url")
def register_model_(model_name, filename, model_url):
    register_model(model_name, filename, model_url)

@click.command(name="training_dataset")
@click.argument("dataset_name")
@click.argument("filename")
@click.argument("model_url")
def register_training_dataset_(dataset_name, filename, model_url):
    register_training_dataset(dataset_name, filename, model_url)

@click.command(name="eval_dataset")
@click.argument("dataset_name")
@click.argument("filename")
@click.argument("model_url")
def register_training_dataset_(dataset_name, filename, model_url):
    register_eval_dataset(dataset_name, filename, model_url)

@click.command(name="snippet_collection")
@click.argument("dataset_name")
@click.argument("filename")
@click.argument("model_url")
def register_snippet_collection_(collection_name, filename, model_url):
    register_snippet_collection(collection_name, filename, model_url)

@click.command(name="install_parsers")
@click.argument("languages", nargs=-1)

def install_parsers_(languages):
    install_parsers(languages)

cli.add_command(download)
download.add_command(download_model_)
download.add_command(download_dataset_)
cli.add_command(register)
register.add_command(register_model_)
cli.add_command(install_parsers_)