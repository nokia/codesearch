# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================

import unittest
import textwrap

from codesearch.install_parsers import install_parsers
from codesearch.code_preprocessing import calls_and_imports, code_tokenization, print_nodes, extract_codechunks

install_parsers()

class TestCodeTokenization(unittest.TestCase):

    def _test_code_tokenization(self, code, expected_tokens, expected_token_types=None):
        tokens, token_types = code_tokenization(code, self.language)
        self.assertEqual(tokens, expected_tokens)
        if expected_token_types:
            self.assertEqual(token_types, expected_token_types)

class TestCodeTokenizationPython(TestCodeTokenization):

    def setUp(self):
        super().setUp()
        self.language = "python"

    def test_default_tokenization1(self):
        code = """
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.random.normal(100, 20, size=1000)
        plt.hist(x, bins='auto')
        plt.show()
        """
        self._test_code_tokenization(code, 
            expected_tokens=('matplotlib', 'pyplot', 'plt', 'numpy', 'np', 'x', 'np', 'random', 'normal', 'size', 'plt', 'hist', 'x', 'bins', 'plt', 'show')
        )

    def test_default_tokenization2(self):
        code = "foo.bar(a, 20, b=1000)"
        self._test_code_tokenization(code,
            expected_tokens=('foo', 'bar', 'a', 'b')
        )

    def test_default_tokenization_comment(self):
        code = "# this is a remark\na = 1"
        self._test_code_tokenization(code,
            expected_tokens=('remark', 'a'),
            expected_token_types=('comment', 'code')
        )

class TestCodeTokenizationJava(TestCodeTokenization):

    def setUp(self):
        super().setUp()
        self.language = "java"

    def test_default_tokenization1(self):
        code = """
        import java.io.File;

        String a = String.valueOf(2); 
        int i = Integer.parseInt(a); 
        """
        self._test_code_tokenization(code, 
            expected_tokens=('java', 'io', 'file', 'a', 'string', 'value', 'of', 'i', 'integer', 'parse', 'int', 'a')
        )

    def test_default_tokenization_comment(self):
        code = "// this is a remark\nint a = 1"
        self._test_code_tokenization(code,
            expected_tokens=('remark', 'a'),
            expected_token_types=('comment', 'code')
        )

class TestCodeTokenizationJavascript(TestCodeTokenization):

    def setUp(self):
        super().setUp()
        self.language = "javascript"

    def test_default_tokenization1(self):
        code = """
        var pathArray = window.location.pathname.split('/');

        var newPathname = "";
        for (i = 0; i < pathArray.length; i++) {
        }
        """
        self._test_code_tokenization(code, 
            expected_tokens=(
                'path', 'array', 'window', 'location', 'pathname', 'split', 'new', 'pathname', 
                'for',  'i', 'i', '<', 'path', 'array', 'length', 'i')
        )

    def test_default_tokenization_comment(self):
        code = "// this is a remark\nlet a = 1"
        self._test_code_tokenization(code,
            expected_tokens=('remark', 'a'),
            expected_token_types=('comment', 'code')
        )

class TestCodeTokenizationBash(TestCodeTokenization):

    def setUp(self):
        super().setUp()
        self.language = "bash"

    def test_default_tokenization1(self):
        code = '# this is a remark\necho "$ATOM_APP"'
        self._test_code_tokenization(code, 
            expected_tokens=('remark', 'echo', 'atom', 'app'),
            expected_token_types=('comment', 'code', 'code', 'code')
        )


if __name__ == '__main__':
    unittest.main()