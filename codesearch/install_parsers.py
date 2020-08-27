# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================


from pathlib import Path
import subprocess
import os

from tree_sitter import Parser, Language

supported_languages = {"python", "javascript", "java", "bash"}

MODULE_DIR = Path(__file__).parent

PARSER_DIR = Path(os.environ.get("TREE_SITTER_DIR", str(MODULE_DIR/"parsers")))
if not PARSER_DIR.exists():
    PARSER_DIR.mkdir()

tree_sitter_build = str(PARSER_DIR/".treesitter/build/my-languages.so")


def git_clone(repo):
  print(f"Cloning {repo}")
  out = subprocess.Popen(['git', 'clone', repo], 
    stdout=subprocess.PIPE, 
    stderr=subprocess.STDOUT)
  stdout, stderr = out.communicate()
  if stderr:
    raise ValueError(f"Error when cloning {repo}")
  print(stdout)

PARSERS = {}

LANGUAGE_ALIASES = {"shell": "bash"}
def get_parser(language):
  language = LANGUAGE_ALIASES.get(language, language)
  if language in PARSERS:
    return PARSERS[language]
  LANGUAGE = Language(tree_sitter_build, language)
  parser = Parser()
  parser.set_language(LANGUAGE)
  PARSERS[language] = parser
  return parser

def language_installed(language):
    if not Path(tree_sitter_build).exists():
        return False
    try:
        Language(tree_sitter_build, language)
        return True
    except:
        return False

def install_parsers(languages=None):
    if not languages:
        languages = supported_languages
    if all(language_installed(lang) for lang in languages):
        print(f"Parsers for languages {languages} already installed.")
        return
    wd = os.getcwd()
    os.chdir(PARSER_DIR)
    for lang in languages:
        if lang not in supported_languages: 
            raise ValueError(f"{lang} not supported. The supported languages are: {', '.join(sorted(supported_languages))}.")
        repo = f"tree-sitter-{lang}"
        git_clone(f"https://github.com/tree-sitter/{repo}")
    Language.build_library(tree_sitter_build, [str(PARSER_DIR/f"tree-sitter-{lang}") for lang in supported_languages])
    os.chdir(wd)

if __name__ == "__main__":
  install_parsers()
