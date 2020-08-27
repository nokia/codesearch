# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================

import unittest

import spacy

class TestSpacy(unittest.TestCase):


    def test_spacy(self):
        spacy.load("en_core_web_md")