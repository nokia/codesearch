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

class TestCallsAndImports(unittest.TestCase):

    def _test_calls_and_imports(self, code, expected_calls, expected_imports):
        code = textwrap.dedent(code)
        try:
            calls, imports = calls_and_imports(code, self.language)
            self.assertEqual(set(calls), set(expected_calls))
            self.assertEqual(len(calls), len(expected_calls))
            self.assertEqual(set(imports), set(expected_imports))
            self.assertEqual(len(imports), len(expected_imports))
        except Exception as e:
            print_nodes(code, self.language)
            raise e


class TestCallsAndImportsPython(TestCallsAndImports):

    def setUp(self):
        super().setUp()
        self.language = "python"

    def test_simple_snippet(self):
        code = """
        import numpy

        a = numpy.array([1, 2, 3])
        """
        self._test_calls_and_imports(code, expected_calls=["array"], expected_imports=["numpy"])

    def test_alias_import(self):
        code = """
        import numpy as np

        a = np.array([1, 2, 3])
        """
        self._test_calls_and_imports(code, expected_calls=["array"], expected_imports=["numpy", "np"])

    def test_from_import(self):
        code = "from os import path"
        self._test_calls_and_imports(code, expected_calls=[], expected_imports=["path", "os"])

    def test_dotted_import(self):
        code = "import os.path"
        self._test_calls_and_imports(code, expected_calls=[], expected_imports=["os", "path"])

    def test_dotted_from_import(self):
        code = "from os.path import exists"
        self._test_calls_and_imports(code, expected_calls=[], expected_imports=["os", "path", "exists"])

    def test_from_import_star(self):
        code = "from os import *"
        self._test_calls_and_imports(code, expected_calls=[], expected_imports=["os"])

    def test_chained_calls(self):
        code = "foo().bar()"
        self._test_calls_and_imports(code, expected_calls=["foo", "bar"], expected_imports=[])

    def test_no_calls_or_imports(self):
        code = """
        a = 1
        b = "import os"
        c = b.not_a_call
        """
        self._test_calls_and_imports(code, expected_calls=[], expected_imports=[])


class TestCallsAndImportsJava(TestCallsAndImports):

    def setUp(self):
        super().setUp()
        self.language = "java"

    def test_simple_snippet(self):
        code = """
        import java.io.File;

        String a = String.valueOf(2); 
        int i = Integer.parseInt(a);  
        """
        self._test_calls_and_imports(code, expected_calls=["valueOf", "parseInt"], expected_imports=["java", "io", "File"])

    def test_import_star(self):
        code = "import java.io.*;"
        self._test_calls_and_imports(code, expected_calls=[], expected_imports=["java", "io"])
    
    def test_chained_calls(self):
        code = "foo(a).bar(b);"
        self._test_calls_and_imports(code, expected_calls=["foo", "bar"], expected_imports=[])

    def test_no_calls_or_imports(self):
        code = """
        int a = 1;
        String b = "import java.io.File;";
        int c = b.not_a_call;
        """
        self._test_calls_and_imports(code, expected_calls=[], expected_imports=[])

   
class TestCallsAndImportsJavascript(TestCallsAndImports):

    def setUp(self):
        super().setUp()
        self.language = "javascript"

    def test_simple_snippet(self):
        code = """
        import {bar} from './foo.js';

        let a = bar()
        """
        self._test_calls_and_imports(code, expected_calls=["bar"], expected_imports=["bar", "foo", "./foo.js"])

    def test_import_curly(self):
        code = "import {bar1, bar2} from './foo.js';"        
        self._test_calls_and_imports(code, expected_calls=[], expected_imports=["bar1", "bar2", "foo", "./foo.js"])

    def test_import_no_curly(self):
        code = "import bar from './foo.js';"        
        self._test_calls_and_imports(code, expected_calls=[], expected_imports=["bar", "foo", "./foo.js"])

    def test_import_star_as1(self):
        code = "import * as bar from './foo.js';"
        self._test_calls_and_imports(code, expected_calls=[], expected_imports=["foo", "./foo.js", "bar"])

    def test_import_as_curly(self):
        code = "import {bar1 as bar1as, bar2 as bar2as} from './foo.js';"
        self._test_calls_and_imports(code, expected_calls=[], expected_imports=["foo", "./foo.js", "bar1", "bar2", "bar1as", "bar2as"])

    def test_import_at(self):
        code = "import { IMainMenu } from '@jupyterlab/mainmenu';"
        self._test_calls_and_imports(code, expected_calls=[], expected_imports=["IMainMenu", "@jupyterlab", "mainmenu", "@jupyterlab/mainmenu"])

    def test_nested_call(self):
        code = "foo(a).bar(b)';"
        self._test_calls_and_imports(code, expected_calls=["foo", "bar"], expected_imports=[])


class TestCallsAndImportsBash(TestCallsAndImports):
    
    def setUp(self):
        super().setUp()
        self.language = "bash"

    def test_calls(self):
        # Every command is considered a call
        code = "grep -v ShipIt | head -1"
        self._test_calls_and_imports(code, expected_calls=["grep", "head"], expected_imports=[])

    def test_calls_2(self):
        code = """
        echo "The script you are running has basename `basename "$0"`, dirname `dirname "$0"`"
        """
        self._test_calls_and_imports(code, expected_calls=["echo", "basename", "dirname"], expected_imports=[])

    def test_duplicate_calls(self):
        code = """
        grep | grep 
        """
        self._test_calls_and_imports(code, expected_calls=["grep"], expected_imports=[])

    def test_calls_sudo(self):
        # Note: grep is here a higher order call and is not detected
        code = """
        sudo grep a
        """
        self._test_calls_and_imports(code, expected_calls=["sudo"], expected_imports=[])

    def test_command_expression(self):
        code = "$(echo hello)"
        self._test_calls_and_imports(code, expected_calls=["echo"], expected_imports=[])

    def test_no_calls(self):
        code = "[matt@server1 ~]$ pwd"
        self._test_calls_and_imports(code, expected_calls=[], expected_imports=[])

    def test_break(self):
        code = """
        while true; do
            break;
        done
        """
        self._test_calls_and_imports(code, expected_calls=[], expected_imports=[])

    # def test_no_calls_2(self):
    #     code = "Black        0;30     Dark Gray     1;30"
    #     self._test_calls_and_imports(code, expected_calls=[], expected_imports=[])

    # def test_no_calls_3(self):
    #     code = "/home/matt"
    #     self._test_calls_and_imports(code, expected_calls=[], expected_imports=[])
