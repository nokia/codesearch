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

class TestExtractCodeChunks(unittest.TestCase):
    
    def _test_extract_codechunks(self, code, expected_code_chunks):
        code = textwrap.dedent(code)
        code_chunks = extract_codechunks(code, self.language)
        try:
            self.assertEqual(code_chunks, expected_code_chunks)
        except Exception as e:
            print('test raised an exception')
            print('tree-sitter nodes for the given code:')
            print_nodes(code, self.language)
            raise e
 

class TestExtractCodeChunksPython(TestExtractCodeChunks):

    def setUp(self):
        super().setUp()
        self.language = "python"

    def test_simple_import(self):
        code = "import numpy"
        self._test_extract_codechunks(code, expected_code_chunks=[("import identifier", "numpy")])

    def test_call_expr(self):
        code = "foo.some_attribute.bar(a, 20, b=1000)"
        self._test_extract_codechunks(code, 
            expected_code_chunks=[("generic identifier", "foo"), ("attribute identifier", "some_attribute"), ("call identifier", "bar"), ("argument identifier", "a"), ("keyword_argument identifier", "b")]
            )

    def test_chained_calls(self):
        code = "foo().property1._property.bar().call()"
        self._test_extract_codechunks(code, expected_code_chunks=[
            ("call identifier", "foo"), 
            ("attribute identifier", "property1"), 
            ("attribute identifier", "_property"),
            ("call identifier", "bar"), 
            ("call identifier", "call"), 
            ])

    def test_alias_import(self):
        code = "import numpy as np"
        self._test_extract_codechunks(code, expected_code_chunks=[("import identifier", "numpy"), ("import identifier", "np")])

    def test_from_import(self):
        code = "from os import path"
        self._test_extract_codechunks(code, expected_code_chunks=[("import identifier", "os"), ("import identifier", "path")])

    
    def test_dotted_import(self):
        code = "import os.path"
        self._test_extract_codechunks(code, expected_code_chunks=[("import identifier", "os"), ("import identifier", "path")])

    def test_dotted_from_import(self):
        code = "from os.path import exists"
        self._test_extract_codechunks(code, expected_code_chunks=[("import identifier", "os"), ("import identifier", "path"), ("import identifier", "exists")])

    def test_from_import_star(self):
        code = "from os import *"
        self._test_extract_codechunks(code, expected_code_chunks=[("import identifier", "os")])



    def test_inline_comment(self):
        code = "# this is a comment\nfoo()"
        self._test_extract_codechunks(code, expected_code_chunks=[
            ("comment", "# this is a comment"), 
            ("call identifier", "foo"), 
            ])

    def test_docstring_comment(self):
        # doc string comments are not kept for python at the moment
        # the tree-sitter parse tree considers them as strings
        code = """  
        def foo():
            \"\"\" this is a docstring \"\"\"
            pass
        """
        self._test_extract_codechunks(code, expected_code_chunks=[("generic identifier", "foo")])


class TestExtractCodeChunksJava(TestExtractCodeChunks):

    def setUp(self):
        super().setUp()
        self.language = "java"

    def test_simple_import(self):
        code = "import java.io.File;"
        self._test_extract_codechunks(code, expected_code_chunks=[("import identifier", "java"), ("import identifier", "io"), ("import identifier", "File")])
        

    def test_call_expr(self):
        code = "foo.some_attribute.bar(a, 20)"
        self._test_extract_codechunks(code, 
            expected_code_chunks=[("generic identifier", "foo"), ("attribute identifier", "some_attribute"), ("call identifier", "bar"), ("argument identifier", "a")]
            )

    def test_chained_calls(self):
        code = "foo().property1._property.bar().call()"
        self._test_extract_codechunks(code, expected_code_chunks=[
            ("call identifier", "foo"), 
            ("attribute identifier", "property1"), 
            ("attribute identifier", "_property"),
            ("call identifier", "bar"), 
            ("call identifier", "call"), 
            ])

    def test_from_import_star(self):
        code = "import java.io.*;"
        self._test_extract_codechunks(code, expected_code_chunks=[("import identifier", "java"), ("import identifier", "io")])


    def test_inline_comment(self):
        code = "// this is a comment\nfoo()"
        self._test_extract_codechunks(code, expected_code_chunks=[
            ("comment", "// this is a comment"), 
            ("call identifier", "foo"), 
            ])

    def test_docstring_comment(self):
        code = """ 
        /**
        This is a docstring.
        */ 
        void foo(){
        }
        """
        self._test_extract_codechunks(code, expected_code_chunks=[("comment", "/**\nThis is a docstring.\n*/"),("generic identifier", "foo")])   

    def test_type_identifier(self):
        code = """ 
        void foo(String a){
            int b;
            SomeType c;
        }
        """
        self._test_extract_codechunks(code, 
            expected_code_chunks=[
                ("generic identifier", "foo"), 
                ("type identifier", "String"), 
                ("generic identifier", "a"),
                ("generic identifier", "b"),
                ("type identifier", "SomeType"),
                ("generic identifier", "c")
                ])   


class TestExtractCodeChunksJavascript(TestExtractCodeChunks):

    def setUp(self):
        super().setUp()
        self.language = "javascript"
        
    def test_call_expr(self):
        code = "foo.some_attribute.bar(a, 20)"
        self._test_extract_codechunks(code, 
            expected_code_chunks=[("generic identifier", "foo"), ("attribute identifier", "some_attribute"), ("call identifier", "bar"), ("argument identifier", "a")]
            )

    def test_chained_calls(self):
        code = "foo().property1._property.bar().call()"
        self._test_extract_codechunks(code, expected_code_chunks=[
            ("call identifier", "foo"), 
            ("attribute identifier", "property1"), 
            ("attribute identifier", "_property"),
            ("call identifier", "bar"), 
            ("call identifier", "call"), 
            ])

    def test_inline_comment(self):
        code = "// this is a comment\nfoo()"
        self._test_extract_codechunks(code, expected_code_chunks=[
            ("comment", "// this is a comment"), 
            ("call identifier", "foo"), 
            ])

    def test_docstring_comment(self):
        code = """ 
        /**
        This is a docstring.
        */ 
        function foo(){
        }
        """
        self._test_extract_codechunks(code, expected_code_chunks=[("comment", "/**\nThis is a docstring.\n*/"),("generic identifier", "foo")])   

    def test_import_curly(self):
        code = "import {bar1, bar2} from './foo.js';"        
        self._test_extract_codechunks(code,expected_code_chunks=[("import identifier", "bar1"), ("import identifier", "bar2"), ("import identifier", "./foo.js")]) 

    def test_import_no_curly(self):
        code = "import bar from './foo.js';"        
        self._test_extract_codechunks(code, expected_code_chunks=[("import identifier", "bar"),  ("import identifier", "./foo.js")]) 

    def test_import_star_as1(self):
        code = "import * as bar from './foo.js';"
        self._test_extract_codechunks(code, expected_code_chunks=[("import identifier", "bar"),  ("import identifier", "./foo.js")]) 

    def test_import_as_curly(self):
        code = "import {bar1 as bar1as, bar2 as bar2as} from './foo.js';"
        self._test_extract_codechunks(code, 
            expected_code_chunks=[
                ("import identifier","bar1"),
                ("import identifier","bar1as"), 
                ("import identifier","bar2"),
                ("import identifier","bar2as"),
                ("import identifier", "./foo.js"), 
                ]
            )

    def test_import_at(self):
        code = "import { IMainMenu } from '@jupyterlab/mainmenu';"
        self._test_extract_codechunks(code, expected_code_chunks=[
            ("import identifier","IMainMenu"), 
            ("import identifier","@jupyterlab/mainmenu")
            ])

class TestExtractCodeChunksBash(TestExtractCodeChunks):

    def setUp(self):
        super().setUp()
        self.language = "bash"
        
    def test_call_expr(self):
        code = "grep -v ShipIt | head -1"
        self._test_extract_codechunks(code, 
            expected_code_chunks=[("call identifier", "grep"), ("call identifier", "head")]
            )

    def test_variable(self):
        code = 'echo "$ATOM_APP"'
        self._test_extract_codechunks(code, 
            expected_code_chunks=[("call identifier", "echo"), ("generic identifier", "ATOM_APP")]
            )

    def test_comment(self):
        code = '# this is a comment\necho "$ATOM_APP"'
        self._test_extract_codechunks(code, 
            expected_code_chunks=[("comment", "# this is a comment"), ("call identifier", "echo"), ("generic identifier", "ATOM_APP")]
            )

    def test_variable_assignment(self):
        code = 'a=1\nexport B=2'
        self._test_extract_codechunks(code, 
            expected_code_chunks=[("generic identifier", "a"), ("generic identifier", "B")]
            )

    def test_calls_2(self):
        code = """
        echo "The script you are running has basename `basename "$0"`, dirname `dirname "$0"`"
        """
        self._test_extract_codechunks(code, 
            expected_code_chunks=[("call identifier", "echo"),("call identifier", "basename"), ("call identifier", "dirname")]
            )


