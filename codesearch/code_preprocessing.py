# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================


from collections import defaultdict
from functools import partial
import re
import enum
import logging 
from pathlib import Path

from codesearch.text_preprocessing import preprocess_text
from codesearch.install_parsers import get_parser

log = logging.getLogger(__name__)


class CompoundNameStrategy(enum.Enum):
    split = 1
    keep = 2
    split_and_keep = 3
    only_last = 4
    
def code_identifiers(code, language):
    annotated_codechunks = extract_codechunks(code, language)
    annotated_codechunks = filter_annotations(annotated_codechunks, lambda t,c: t.endswith("identifier"))
    return [identifier for _, identifier in annotated_codechunks]

def code_tokenization(code, language, **kwargs):
    """
    Splits snake and camel cased identifiers.
    Does not keep all code tokens. 
    Only comments, a few binary operators, and for and while are kept.
    """
    annotated_codechunks = extract_codechunks(code, language)
    return preprocess_codechunks(annotated_codechunks, **kwargs)
    

def calls_and_imports(code, language):
    annotated_codechunks = extract_codechunks(code, language)
    def _expand(calls, expand_root=False):
        result = []
        for _, call in calls:
            call_pieces = call.split(".")
            for i in range(len(call_pieces) - 1):
                result.append(".".join(call_pieces[len(call_pieces) - i - 1:]))
            result.append(call)
            if len(call_pieces) > 1 and expand_root:
                result.append(call_pieces[0])
        return result
    
    call_chunks = filter_annotations(annotated_codechunks, lambda t1, t2: t1 == "call identifier")
    calls = set(call for _, call in call_chunks)
    import_chunks = filter_annotations(annotated_codechunks, lambda t1, t2: t1 == "import identifier")
    imports = set()
    for _, import_identifier in import_chunks:
        imports.update(expand_import_identifier(import_identifier, language))

    return calls, imports

def expand_import_identifier(import_identifier, language):
    if language == "javascript":
        import_path = Path(import_identifier)
        import_parts = list(map(str,import_path.parts))
        import_parts[-1] = import_path.stem
        import_parts.append(import_identifier)
        return list(set(import_parts))
    elif language == "java" or language == "python":
        return list(set(import_identifier.split(".")))
    else:
        log.warning("Language {language} not supported")
        return []


def preprocess_codechunks(annotated_codechunks, identifier_types=["call", "import", "generic", "attribute", "argument", "keyword_argument"], lemmatize=True, remove_stop=True, clean_howto=False, keep_comments=True, keep_loops=True, keep_bin_ops=True, keep_unk=True, rstrip_numbers=True, case_split=True, stopwords=[]):
    bin_ops = ["<", ">", "+", "-", "/", "//", "*", "**", "@", "%"]
    loops = ["for", "while"]
    def keep(type_, _chunk):
        if stopwords and _chunk in stopwords:
            return False
        if keep_comments and type_  == "comment":
            return True
        if keep_loops and type_ in loops:
            return True
        if keep_bin_ops and type_ in bin_ops:
            return True
        if re.match("({}) identifier".format("|".join(identifier_types)), type_):
            return True
        if keep_unk and type_ == "?":
            return True
        return False

    annotated_codechunks = filter_annotations(annotated_codechunks, keep)
    annotation_map = defaultdict(lambda: "code")
    annotation_map["comment"] = "comment"

    annotated_codechunks = reduce_annotations(annotated_codechunks, annotation_map)

    comment_preprocessor = partial(preprocess_text, lemmatize=lemmatize, remove_stop=remove_stop, clean_howto=clean_howto)
    identifier_preprocessor = partial(preprocess_identifiers, rstrip_numbers=rstrip_numbers, case_split=case_split)

    annotated_codetokens = transform_codechunks(annotated_codechunks, "comment", comment_preprocessor)
    annotated_codetokens = transform_codechunks(annotated_codetokens, "code", identifier_preprocessor)
    
    try:
        types, code_tokens = zip(*annotated_codetokens)
    except ValueError:
        types, code_tokens = [], []
    return code_tokens, types


def reduce_annotations(annotated_codechunks, annotation_map):
    return [(annotation_map[t], codechunk) for t, codechunk in annotated_codechunks]


def filter_annotations(annotated_codechunks, to_keep_fn):
    return [(type_, codechunk) for type_, codechunk in annotated_codechunks 
            if to_keep_fn(type_, codechunk)
           ]

def transform_codechunks(annotated_codechunks, type_, transform_fn):
    result = []
    for t, c in annotated_codechunks:
        if t == type_:
            c_transformed = transform_fn(c)
            if not isinstance(c_transformed, list):
                c_transformed = [c_transformed]
            for c_ in c_transformed:
                result.append((t, c_))
        else:
            result.append((t, c))
    return result


def extract_codechunks(code, language):
    try:
        code_bytes = bytes(code, "utf8")
        tree = get_parser(language).parse(code_bytes)
        acc = CodeTokenAccumulator(code_bytes)
        acc.visit(tree.root_node)

        return [(type_, t.decode("utf8").strip()) for type_, t in acc.tokens]
    except Exception as e:
        log.warning(f"Exception when parsing code in language {language} using generic tokenization instead\n{code}\n{e}")
        return [('?', token.strip()) for token in code.split(" ")]


def select_span(start, end, lines):
    span = []
    if start[0] > end[0]:
        raise ValueError("Invalid span")
    while start[0] != end[0]:
        span.append(lines[start[0]][start[1]:])
        start = (start[0]+1, 0)
    span.append(lines[start[0]][start[1]:end[1]])
    return b"\n".join(span)


def preprocess_identifiers(identifier, rstrip_numbers=True, case_split=True):
    if identifier.startswith("_"):
        identifier = identifier[1:]
    if rstrip_numbers:   
        identifier = re.sub("(.*)[0-9]+$", r"\1", identifier)
        
    if case_split and "_" in identifier:
        identifiers = snake_case_split(identifier)
    elif case_split:
        identifiers = camel_case_split(identifier)
    else:
        identifiers = [identifier]
    identifiers = [t.lower() for t in identifiers if not t.isspace() and t]
    return identifiers


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches ]

def snake_case_split(identifier):
    return identifier.split("_")


class CodeToken(object):
    def __init__(self, start, end, type, value):
        self.start_point = start
        self.end_point = end
        self.type = type
        self.value = value
        
    def __repr__(self):
        return f"<{self.__class__.__name__} {self.value} type:{self.type} start:{self.start_point}, end:{self.end_point}"

    
class NodeVisitor(object):
    # Inspired by the node visitor of https://github.com/python/cpython/blob/3.8/Lib/ast.py
    def __init__(self, code):
        self.lines = code.split(b"\n")
        self._type_to_method = {"<": "st", ">": "gt", "+": "add", 
                                "-": "sub", "*": "mul", "/": "div",
                                "//": "floordiv", "%": "mod", "**": "pow",
                                "@": "matmul"
                               }
        

    def visit(self, node):
        """Visit a node."""
        method_suffix = self._type_to_method.get(node.type, node.type)
        method = f'visit_{method_suffix}'
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        for c in node.children:
            self.visit(c)
            
    def extract_token(self, node, type_=None):
        start, end = node.start_point, node.end_point, 
        token =  select_span(start, end, self.lines)
        if not type_:
            type_ = node.type
        return CodeToken(start, end, type_, token)


def print_nodes(code, language):
    code_bytes = bytes(code, "utf8")
    lines = code_bytes.split(b"\n")
    tree = get_parser(language).parse(code_bytes)
    _print_nodes(tree.root_node, lines)
    
def _print_nodes(node, lines, level=0, max_tokens=200):
    indent = '  ' * level
    print(f"{indent}{node.type}:")
    print(indent + select_span(node.start_point, node.end_point, lines).decode("utf-8")[:200])
    for child in node.children:
        _print_nodes(child, lines, level + 1)
    
RESERVED_KEYWORDS_SHELL = {b"break", b"case" b"do", b"done", b"elif", b"else", b"esac", b"fi", b"false", b"for", b"function", b"if", b"in", b"select", b"then", b"true", b"until", b"while", b"{", b"}", b"||", b"!", b"[", b"]", b":"}

class CodeTokenAccumulator(NodeVisitor):
    
    def __init__(self, code):
        super().__init__(code)
        self._tokens = []
        self.scope = "generic" # used to determine the type of identifier (call identifier, import identifier, attribute identifier)

    ################## Calls ##################

    def visit_call(self, node): # python
        # We extract only the function/method name (not the module/class/... where it was called upon)
        if node.children[0].type == "identifier":
            self._visit_with_scope(node.children[0], "call") 
            for c in node.children[1:]:
                self.visit(c)
        elif node.children[0].type == "attribute":
            attr = node.children[0]
            identifier = attr.children[-1]
            if identifier.type == "identifier":
                for c in attr.children[:-1]:
                    self.visit(c)
                self._visit_with_scope(identifier, "call")
                
            else: # should not happen
                for c in attr.children:
                    self.visit(c)
            for c in node.children[1:]: # visit argument list etc.
                self.visit(c)
        else:
            self.generic_visit(node) # shouldn't happen


    def visit_method_invocation(self, node): # java
        for i, c in enumerate(node.children):
            if c.type == "argument_list":
                break
        for j, c in enumerate(node.children):
            if j == i - 1: 
                if c.type == "identifier": 
                    self._visit_with_scope(c, "call") # method name
                else: # this shouldn't happen
                    self.visit(c)
            else:
                self.visit(c)

    def visit_call_expression(self, node): #javascript
        if node.children[0].type == "identifier":
            self._visit_with_scope(node.children[0], "call") 
            for c in node.children[1:]:
                self.visit(c)
        elif node.children[0].type == "member_expression": 
            attr = node.children[0]
            identifier = attr.children[-1]
            if identifier.type == "property_identifier":
                for c in attr.children[:-1]:
                    self.visit(c)
                self._visit_with_scope(identifier, "call")
            else:
                for c in attr.children:
                    self.visit(c)
            for c in node.children[1:]: # visit arguments etc.
                self.visit(c)
        else:
            self.generic_visit(node) # shouldn't happen
        
    
    def visit_template_string(self, node):
        self._visit_children_with_scope(node, "template_string")

    def visit_argument_list(self, node): # python, java
        self._visit_children_with_scope(node, "argument")

    def visit_arguments(self, node): # javascript
        self.visit_argument_list(node)

    def visit_command_name(self, node): # bash
        
        command_word = node.children[0]
        if command_word.type == "word":
            token = self.extract_token(command_word, "call identifier")
            if token.value not in RESERVED_KEYWORDS_SHELL:
                self._tokens.append(token)
        else:
            self.generic_visit(node)
    
    ################## Imports ##################

    def visit_import_statement(self, node):
        self._visit_children_with_scope(node, "import")
        
    def visit_import_from_statement(self, node):
        self._visit_children_with_scope(node, "import")

    def visit_import_declaration(self, node):
        self._visit_children_with_scope(node, "import")

    def visit_wildcard_import(self, node): # python
        pass

    def visit_asterisk(self, node): # java
        pass
        
    #def visit_dotted_name(self, node):
    #    self._handle_compound_name(node)
         
    def visit_attribute(self, node): # python
        if node.children[0].type == "identifier":
            self._visit_with_scope(node.children[0], "generic") # this is the variable to which the attribute expression applies
        else:
            self._visit_with_scope(node.children[0], "attribute")
        self._visit_with_scope(node.children[-1], "attribute") # this is the attribute part

    def visit_field_access(self, node): # java
       self.visit_attribute(node)

    def visit_property_identifier(self, node):
        token = self.extract_token(node, "attribute identifier")
        self._tokens.append(token)

    def visit_member_expression(self, node): # for javascript
        self.visit_attribute(node)
        
    def visit_identifier(self, node):
        type_ = f"{self.scope} identifier"
        token = self.extract_token(node, type_)
        self._tokens.append(token)

    def visit_variable_name(self, node):
        type_ = f"generic identifier"
        token = self.extract_token(node, type_)
        self._tokens.append(token)

    def visit_string(self, node):
        if self.scope != "import":
            self.generic_visit(node)
            return
        type_ = f"{self.scope} identifier"
        token = self.extract_token(node, type_)
        token.value = token.value[1:-1] # strip string markers

        self._tokens.append(token)

    def visit_property_identifier(self, node): # for javascript
        type_ = f"{self.scope} identifier"
        token = self.extract_token(node, type_)
        self._tokens.append(token)

    def visit_type_identifier(self, node):
        type_ = "type identifier"
        token = self.extract_token(node, type_)
        self._tokens.append(token)

    def visit_st(self, node):
        token = self.extract_token(node, "<")
        self._tokens.append(token)
        
    def visit_gt(self, node):
        token = self.extract_token(node, ">")
        self._tokens.append(token)
        
    def visit_add(self, node):
        token = self.extract_token(node, "+")
        self._tokens.append(token)    
        
    def visit_sub(self, node):
        token = self.extract_token(node, "-")
        self._tokens.append(token)    
        
    def visit_mul(self, node):
        if self.scope == "import":
            return
        token = self.extract_token(node, "*")
        self._tokens.append(token) 
        
    def visit_div(self, node):
        token = self.extract_token(node, "/")
        self._tokens.append(token)
        
    def visit_floordiv(self, node):
        token = self.extract_token(node, "//")
        self._tokens.append(token)

    def visit_mod(self, node):
        token = self.extract_token(node, "%")
        self._tokens.append(token) 
        
    def visit_pow(self, node):
        token = self.extract_token(node, "**")
        self._tokens.append(token)  
        
    def visit_matmul(self, node):
        token = self.extract_token(node, "@")
        self._tokens.append(token) 
        
    def visit_while(self, node):
        token = self.extract_token(node, "while")
        self._tokens.append(token)
    
    def visit_for(self, node):
        token = self.extract_token(node, "for")
        self._tokens.append(token)
        
    def visit_comment(self, node):
        token = self.extract_token(node, "comment")
        self._tokens.append(token)

    def visit_keyword_argument(self, node):
        self._visit_children_with_scope(node, "keyword_argument")
        
    def _visit_children_with_scope(self, node, scope):
        outer_scope = self.scope
        self.scope = scope
        self.generic_visit(node)
        self.scope = outer_scope

    def _visit_with_scope(self, node, scope):
        outer_scope = self.scope
        self.scope = scope
        self.visit(node)
        self.scope = outer_scope
    
    @property
    def tokens(self):
        self._tokens.sort(key=lambda c: (c.start_point[0], c.start_point[1], -c.end_point[0], -c.end_point[1]))
        return [(t.type, t.value) for t in self._tokens]