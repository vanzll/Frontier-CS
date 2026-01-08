import os
import re
import ast
import textwrap
from typing import Dict, List, Any


class Solution:
    def solve(self, resources_path: str) -> dict:
        grammar_text = ""
        try:
            gp = os.path.join(resources_path, "sql_grammar.txt")
            if os.path.isfile(gp):
                with open(gp, "r", encoding="utf-8", errors="ignore") as f:
                    grammar_text = f.read()
        except Exception:
            grammar_text = ""

        extracted_keywords = set()

        def extract_string_iter(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                yield node.value
            elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                for e in node.elts:
                    yield from extract_string_iter(e)

        try:
            tp = os.path.join(resources_path, "sql_engine", "tokenizer.py")
            if os.path.isfile(tp):
                with open(tp, "r", encoding="utf-8", errors="ignore") as f:
                    tsrc = f.read()
                mod = ast.parse(tsrc)
                for n in mod.body:
                    if isinstance(n, ast.Assign):
                        for tgt in n.targets:
                            if isinstance(tgt, ast.Name) and tgt.id.upper() in {
                                "KEYWORDS",
                                "RESERVED",
                                "RESERVED_WORDS",
                                "RESERVED_KEYWORDS",
                                "SQL_KEYWORDS",
                            }:
                                for s in extract_string_iter(n.value):
                                    if s:
                                        extracted_keywords.add(s)
        except Exception:
            pass

        try:
            if grammar_text:
                for a, b in re.findall(r"'([^']*)'|\"([^\"]*)\"", grammar_text):
                    s = a or b
                    if s:
                        extracted_keywords.add(s)
                for w in re.findall(r"\b[A-Z][A-Z0-9_]{1,}\b", grammar_text):
                    extracted_keywords.add(w)
        except Exception:
            pass

        fallback_keywords = [
            "SELECT", "FROM", "WHERE", "GROUP", "BY", "HAVING", "ORDER", "LIMIT", "OFFSET",
            "DISTINCT", "ALL", "AS", "AND", "OR", "NOT", "NULL", "IS", "IN", "EXISTS",
            "BETWEEN", "LIKE", "ESCAPE", "CASE", "WHEN", "THEN", "ELSE", "END",
            "UNION", "INTERSECT", "EXCEPT", "WITH", "RECURSIVE", "VALUES",
            "INSERT", "INTO", "DEFAULT", "UPDATE", "SET", "DELETE",
            "CREATE", "TABLE", "TEMP", "TEMPORARY", "IF", "PRIMARY", "KEY",
            "UNIQUE", "CHECK", "FOREIGN", "REFERENCES", "ON", "CONFLICT",
            "COLLATE", "INDEX", "VIEW", "TRIGGER", "DROP", "ALTER", "ADD", "COLUMN", "RENAME",
            "JOIN", "LEFT", "RIGHT", "FULL", "INNER", "OUTER", "CROSS", "NATURAL", "USING",
            "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE",
            "ASC", "DESC",
            "CAST", "COLLATION",
            "TRUE", "FALSE",
        ]
        for k in fallback_keywords:
            extracted_keywords.add(k)

        keywords_list = sorted({k for k in extracted_keywords if isinstance(k, str) and k and len(k) <= 40})

        code_template = r'''
import re
import random
import time

GRAMMAR_TEXT = __GRAMMAR_TEXT__
KEYWORDS = __KEYWORDS_LIST__

_PUNCT = {",", ")", ";", ".", "]"}
_NOPRE_SPACE = {",", ")", ";", ".", "]", "}", ":", "?"}
_NOPOST_SPACE = {"(", "[", "{", ".", "$", "@", ":"}

_SQL_TYPES = [
    "INTEGER", "INT", "BIGINT", "SMALLINT",
    "TEXT", "VARCHAR(255)", "CHAR(10)", "CLOB",
    "REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL(10,2)",
    "BLOB", "BOOLEAN",
    "DATE", "TIME", "DATETIME", "TIMESTAMP",
]

_FUNCS = [
    "COUNT", "SUM", "AVG", "MIN", "MAX",
    "ABS", "ROUND", "LENGTH", "LOWER", "UPPER",
    "SUBSTR", "COALESCE", "NULLIF",
    "CAST",
]

_BIN_OPS = [
    "+", "-", "*", "/", "%", "||",
    "=", "==", "!=", "<>", "<", "<=", ">", ">=",
    "AND", "OR",
    "&", "|", "<<", ">>",
]

_UNARY_OPS = ["NOT", "-", "+", "~"]

_LITERALS = ["NULL", "TRUE", "FALSE"]

_ID_BASE = [
    "t", "u", "v", "w", "x", "y", "z",
    "users", "orders", "products", "items", "log",
    "tmp", "_tmp", "T1", "T2", "main", "schema",
    "select", "from", "where", "group", "order", "limit",
]

_STR_BASE = [
    "", "a", "A", "test", "x", "y", "O'Reilly", "null", "NULL",
    "line\nbreak", "tab\tchar", "quote\"d", "unicødê", "semi;colon",
    "/*comment*/", "--comment",
]

_NUM_BASE = [
    "0", "1", "-1", "2", "10", "99", "100", "-0", "0.0", "-0.0",
    "1.0", ".5", "1.", "1e0", "1e-3", "-1e+9", "9223372036854775807",
    "9223372036854775808", "-9223372036854775809", "0x10", "0xFFFFFFFF",
]

_COMMENT_SNIPS = [
    "/* */",
    "/*comment*/",
    "/* nested /* comment */ end */",
    "--comment\n",
    "--\n",
]

_WS_SNIPS = [" ", "  ", "\t", "\n", "\r\n", " \n ", "\t \t", "\n\n"]

def _detokenize(tokens):
    out = []
    prev = ""
    for tok in tokens:
        if tok is None:
            continue
        if isinstance(tok, str) and tok == "":
            continue
        if isinstance(tok, str) and (" " in tok) and (tok.strip() == tok):
            parts = tok.split()
        else:
            parts = [tok]
        for t in parts:
            if not out:
                out.append(t)
                prev = t
                continue
            if (t in _NOPRE_SPACE) or (prev in _NOPOST_SPACE):
                out.append(t)
            else:
                out.append(" " + t)
            prev = t
    return "".join(out)

class _EBNFNode:
    __slots__ = ("k", "a", "b")
    def __init__(self, k, a=None, b=None):
        self.k = k
        self.a = a
        self.b = b

def _tokenize_rhs(s):
    s = s.strip()
    if not s:
        return []
    tokens = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in "()[]{}|?*+;":
            tokens.append(c)
            i += 1
            continue
        if c == "<":
            j = s.find(">", i + 1)
            if j != -1:
                tokens.append(s[i:j+1])
                i = j + 1
                continue
        if c == "'" or c == '"':
            q = c
            j = i + 1
            esc = False
            while j < n:
                ch = s[j]
                if esc:
                    esc = False
                    j += 1
                    continue
                if ch == "\\":
                    esc = True
                    j += 1
                    continue
                if ch == q:
                    break
                j += 1
            if j < n and s[j] == q:
                tokens.append(s[i:j+1])
                i = j + 1
            else:
                tokens.append(s[i:])
                break
            continue
        if s.startswith("::=", i):
            tokens.append("::=")
            i += 3
            continue
        if s.startswith("->", i):
            tokens.append("->")
            i += 2
            continue
        for op in ("||", "<<", ">>", "<=", ">=", "<>", "!=", "=="):
            if s.startswith(op, i):
                tokens.append(op)
                i += len(op)
                break
        else:
            if c in ",.=:+-*/%<>!&~^":
                tokens.append(c)
                i += 1
                continue
            m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", s[i:])
            if m:
                tokens.append(m.group(0))
                i += len(m.group(0))
                continue
            m = re.match(r"[0-9]+", s[i:])
            if m:
                tokens.append(m.group(0))
                i += len(m.group(0))
                continue
            tokens.append(c)
            i += 1
    return tokens

class _EBNFParser:
    __slots__ = ("toks", "i", "rule_names")
    def __init__(self, toks, rule_names):
        self.toks = toks
        self.i = 0
        self.rule_names = rule_names

    def _peek(self):
        if self.i >= len(self.toks):
            return None
        return self.toks[self.i]

    def _eat(self, x=None):
        if self.i >= len(self.toks):
            return None
        t = self.toks[self.i]
        if x is not None and t != x:
            return None
        self.i += 1
        return t

    def parse_expr(self):
        return self.parse_choice()

    def parse_choice(self):
        left = self.parse_seq()
        if left is None:
            left = _EBNFNode("seq", [])
        alts = [left]
        while self._peek() == "|":
            self._eat("|")
            right = self.parse_seq()
            if right is None:
                right = _EBNFNode("seq", [])
            alts.append(right)
        if len(alts) == 1:
            return alts[0]
        return _EBNFNode("choice", alts)

    def parse_seq(self):
        items = []
        while True:
            t = self._peek()
            if t is None or t in ("|", ")", "]", "}", ";"):
                break
            atom = self.parse_atom()
            if atom is None:
                break
            items.append(atom)
        if not items:
            return None
        if len(items) == 1:
            return items[0]
        return _EBNFNode("seq", items)

    def parse_atom(self):
        base = self.parse_base()
        if base is None:
            return None
        t = self._peek()
        if t in ("?", "*", "+"):
            op = self._eat()
            if op == "?":
                return _EBNFNode("opt", base)
            if op == "*":
                return _EBNFNode("rep", base, (0, 2))
            if op == "+":
                return _EBNFNode("rep", base, (1, 2))
        return base

    def parse_base(self):
        t = self._peek()
        if t is None:
            return None
        if t == "(":
            self._eat("(")
            e = self.parse_expr()
            self._eat(")")
            return e if e is not None else _EBNFNode("seq", [])
        if t == "[":
            self._eat("[")
            e = self.parse_expr()
            self._eat("]")
            return _EBNFNode("opt", e if e is not None else _EBNFNode("seq", []))
        if t == "{":
            self._eat("{")
            e = self.parse_expr()
            self._eat("}")
            return _EBNFNode("rep", e if e is not None else _EBNFNode("seq", []), (0, 2))
        tok = self._eat()
        if tok is None:
            return None
        if tok.startswith("<") and tok.endswith(">"):
            return _EBNFNode("nt", tok[1:-1])
        if (tok in self.rule_names) and (not tok.isupper()):
            return _EBNFNode("nt", tok)
        if tok and (tok[0] == tok[-1] == "'") and len(tok) >= 2:
            return _EBNFNode("t", tok[1:-1])
        if tok and (tok[0] == tok[-1] == '"') and len(tok) >= 2:
            return _EBNFNode("t", tok[1:-1])
        if tok in self.rule_names and tok.islower():
            return _EBNFNode("nt", tok)
        return _EBNFNode("t", tok)

class _Grammar:
    __slots__ = ("rules", "start", "rule_names", "alts")
    def __init__(self, rules, start):
        self.rules = rules
        self.start = start
        self.rule_names = set(rules.keys())
        self.alts = rules

def _parse_grammar(text):
    if not text:
        return None
    rules = {}
    order = []
    cur_lhs = None
    cur_rhs = []
    sep_re = re.compile(r"(::=|->|:)")
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("//"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = sep_re.search(line)
        if m:
            if cur_lhs is not None:
                rhs_text = " ".join(cur_rhs).strip()
                if rhs_text:
                    rules[cur_lhs] = rules.get(cur_lhs, []) + [rhs_text]
                cur_lhs = None
                cur_rhs = []
            parts = sep_re.split(line, maxsplit=1)
            if len(parts) >= 3:
                lhs = parts[0].strip()
                rhs = parts[2].strip()
            else:
                continue
            lhs = lhs.strip()
            if lhs.startswith("<") and lhs.endswith(">"):
                lhs = lhs[1:-1]
            if lhs:
                cur_lhs = lhs
                order.append(lhs)
                if rhs:
                    cur_rhs.append(rhs)
        else:
            if cur_lhs is not None:
                cur_rhs.append(line)
    if cur_lhs is not None:
        rhs_text = " ".join(cur_rhs).strip()
        if rhs_text:
            rules[cur_lhs] = rules.get(cur_lhs, []) + [rhs_text]

    if not rules:
        return None

    rule_names = set(rules.keys())
    compiled = {}
    for lhs, rhs_list in rules.items():
        alts = []
        for rhs in rhs_list:
            rhs = rhs.strip()
            if rhs.endswith(";"):
                rhs = rhs[:-1]
            toks = _tokenize_rhs(rhs)
            p = _EBNFParser(toks, rule_names)
            node = p.parse_expr()
            if node is None:
                node = _EBNFNode("seq", [])
            alts.append(node)
        compiled[lhs] = alts

    start = order[0] if order else next(iter(compiled.keys()))
    return _Grammar(compiled, start)

def _safe_sql_string(s):
    if s is None:
        return "NULL"
    s = str(s)
    s = s.replace("\x00", "")
    s = s.replace("'", "''")
    return "'" + s + "'"

def _rand_ident(rng):
    b = rng.choice(_ID_BASE)
    if rng.random() < 0.25:
        b += str(rng.randrange(0, 100))
    if rng.random() < 0.12:
        b = "_" + b
    if rng.random() < 0.10:
        q = rng.choice(['"', "`", "["])
        if q == "[":
            return "[" + b.replace("]", "]]") + "]"
        return q + b.replace(q, q + q) + q
    return b

def _rand_number(rng):
    if rng.random() < 0.75:
        return rng.choice(_NUM_BASE)
    if rng.random() < 0.5:
        return str(rng.randrange(-1000000, 1000000))
    return str(rng.random() * rng.randrange(0, 1000000))

def _rand_literal(rng):
    r = rng.random()
    if r < 0.25:
        return rng.choice(_LITERALS)
    if r < 0.55:
        return _rand_number(rng)
    return _safe_sql_string(rng.choice(_STR_BASE) if rng.random() < 0.7 else "".join(rng.choice("abcXYZ0123 _-") for _ in range(rng.randrange(0, 25))))

def _rand_func_call(rng, depth):
    fn = rng.choice(_FUNCS)
    if fn == "CAST":
        return ["CAST", "(", *_rand_expr(rng, max(0, depth - 1)), "AS", rng.choice(_SQL_TYPES), ")"]
    if fn == "COUNT" and rng.random() < 0.25:
        return ["COUNT", "(", "*", ")"]
    args = []
    argc = 1 if rng.random() < 0.75 else 2
    for i in range(argc):
        if i:
            args.append(",")
        args.extend(_rand_expr(rng, max(0, depth - 1)))
    return [fn, "(", *args, ")"]

def _rand_expr(rng, depth):
    if depth <= 0:
        r = rng.random()
        if r < 0.33:
            return [_rand_ident(rng)]
        if r < 0.66:
            return [_rand_literal(rng)]
        return ["(", _rand_literal(rng), ")"]
    r = rng.random()
    if r < 0.20:
        return [_rand_literal(rng)]
    if r < 0.38:
        return [_rand_ident(rng)]
    if r < 0.55:
        op = rng.choice(_UNARY_OPS)
        return [op, *_rand_expr(rng, depth - 1)]
    if r < 0.78:
        left = _rand_expr(rng, depth - 1)
        op = rng.choice(_BIN_OPS)
        right = _rand_expr(rng, depth - 1)
        return ["(", *left, op, *right, ")"]
    if r < 0.90:
        return _rand_func_call(rng, depth - 1)
    if r < 0.95:
        # IN (...) list
        items = []
        n = 1 + rng.randrange(1, 5)
        for i in range(n):
            if i:
                items.append(",")
            items.append(_rand_literal(rng))
        return ["(", *_rand_expr(rng, depth - 1), "IN", "(", *items, ")", ")"]
    # CASE
    return ["CASE", "WHEN", *_rand_expr(rng, depth - 1), "THEN", _rand_literal(rng), "ELSE", _rand_literal(rng), "END"]

def _make_select(rng, depth=2):
    cols = []
    ncols = 1 + rng.randrange(1, 5)
    for i in range(ncols):
        if i:
            cols.append(",")
        if rng.random() < 0.15:
            cols.append("*")
        else:
            cols.extend(_rand_expr(rng, 2 + depth))
            if rng.random() < 0.35:
                cols.extend(["AS", _rand_ident(rng)])
    tbl = _rand_ident(rng)
    toks = ["SELECT"]
    if rng.random() < 0.2:
        toks.append("DISTINCT")
    toks.extend(cols)
    toks.extend(["FROM", tbl])
    if rng.random() < 0.35:
        jt = rng.choice(["JOIN", "LEFT JOIN", "LEFT OUTER JOIN", "INNER JOIN", "CROSS JOIN"])
        toks.extend([jt, _rand_ident(rng)])
        if "CROSS" not in jt and rng.random() < 0.8:
            if rng.random() < 0.5:
                toks.extend(["ON", *_rand_expr(rng, 2 + depth)])
            else:
                toks.extend(["USING", "(", _rand_ident(rng), ")"])
    if rng.random() < 0.60:
        toks.extend(["WHERE", *_rand_expr(rng, 3 + depth)])
        if rng.random() < 0.25:
            toks.extend(["AND", _rand_ident(rng), "IS", "NULL"])
        if rng.random() < 0.25:
            toks.extend(["OR", _rand_ident(rng), "BETWEEN", _rand_number(rng), "AND", _rand_number(rng)])
    if rng.random() < 0.30:
        toks.extend(["GROUP", "BY", _rand_ident(rng)])
        if rng.random() < 0.35:
            toks.extend([",", _rand_ident(rng)])
        if rng.random() < 0.35:
            toks.extend(["HAVING", *_rand_expr(rng, 2 + depth)])
    if rng.random() < 0.50:
        toks.extend(["ORDER", "BY", _rand_ident(rng), rng.choice(["ASC", "DESC"])])
        if rng.random() < 0.25:
            toks.extend([",", _rand_ident(rng), rng.choice(["ASC", "DESC"])])
    if rng.random() < 0.40:
        toks.extend(["LIMIT", _rand_number(rng)])
        if rng.random() < 0.35:
            toks.extend(["OFFSET", _rand_number(rng)])
    if rng.random() < 0.18:
        toks = ["WITH", _rand_ident(rng), "AS", "(", "SELECT", "1", "AS", "x", ")", *toks]
    if rng.random() < 0.18:
        toks = ["(", *toks, ")", rng.choice(["UNION", "INTERSECT", "EXCEPT"]), *(_make_select(rng, max(0, depth - 1)).split())]
        # fallback: if split breaks tokenization, keep it as string, but this creates tokens with spaces; detokenize handles it
    return _detokenize([t for t in toks if t is not None])

def _make_insert(rng):
    tbl = _rand_ident(rng)
    if rng.random() < 0.35:
        cols = [_rand_ident(rng)]
        if rng.random() < 0.5:
            cols.extend([",", _rand_ident(rng)])
        if rng.random() < 0.25:
            cols.extend([",", _rand_ident(rng)])
        if rng.random() < 0.6:
            vals = []
            n = 1 + rng.randrange(1, 4)
            for i in range(n):
                if i:
                    vals.append(",")
                vals.append(_rand_literal(rng))
            return _detokenize(["INSERT", "INTO", tbl, "(", *cols, ")", "VALUES", "(", *vals, ")"])
        return _detokenize(["INSERT", "INTO", tbl, "(", *cols, ")", _make_select(rng, 1)])
    if rng.random() < 0.6:
        row = [_rand_literal(rng)]
        if rng.random() < 0.55:
            row.extend([",", _rand_literal(rng)])
        if rng.random() < 0.25:
            row.extend([",", _rand_literal(rng)])
        if rng.random() < 0.25:
            return _detokenize(["INSERT", "OR", rng.choice(["REPLACE", "IGNORE", "ABORT", "FAIL", "ROLLBACK"]), "INTO", tbl, "VALUES", "(", *row, ")"])
        if rng.random() < 0.25:
            row2 = [_rand_literal(rng)]
            if rng.random() < 0.55:
                row2.extend([",", _rand_literal(rng)])
            return _detokenize(["INSERT", "INTO", tbl, "VALUES", "(", *row, ")", ",", "(", *row2, ")"])
        return _detokenize(["INSERT", "INTO", tbl, "VALUES", "(", *row, ")"])
    return _detokenize(["INSERT", "INTO", tbl, _make_select(rng, 2)])

def _make_update(rng):
    tbl = _rand_ident(rng)
    sets = [_rand_ident(rng), "=", *_rand_expr(rng, 3)]
    if rng.random() < 0.5:
        sets.extend([",", _rand_ident(rng), "=", *_rand_expr(rng, 2)])
    toks = ["UPDATE", tbl, "SET", *sets]
    if rng.random() < 0.75:
        toks.extend(["WHERE", *_rand_expr(rng, 3)])
    return _detokenize(toks)

def _make_delete(rng):
    tbl = _rand_ident(rng)
    toks = ["DELETE", "FROM", tbl]
    if rng.random() < 0.75:
        toks.extend(["WHERE", *_rand_expr(rng, 3)])
    return _detokenize(toks)

def _make_create_table(rng):
    tbl = _rand_ident(rng)
    cols = []
    ncols = 1 + rng.randrange(1, 6)
    for i in range(ncols):
        if i:
            cols.append(",")
        col = [_rand_ident(rng), rng.choice(_SQL_TYPES)]
        if rng.random() < 0.20:
            col.extend(["PRIMARY", "KEY"])
            if rng.random() < 0.35:
                col.append("AUTOINCREMENT")
        if rng.random() < 0.20:
            col.append("NOT")
            col.append("NULL")
        if rng.random() < 0.15:
            col.append("UNIQUE")
        if rng.random() < 0.15:
            col.extend(["DEFAULT", _rand_literal(rng)])
        if rng.random() < 0.12:
            col.extend(["CHECK", "(", *_rand_expr(rng, 2), ")"])
        cols.extend(col)
    if rng.random() < 0.18:
        cols.extend([",", "FOREIGN", "KEY", "(", _rand_ident(rng), ")", "REFERENCES", _rand_ident(rng), "(", _rand_ident(rng), ")"])
        if rng.random() < 0.5:
            cols.extend(["ON", "DELETE", rng.choice(["CASCADE", "SET NULL", "SET DEFAULT", "RESTRICT", "NO ACTION"])])
    toks = ["CREATE"]
    if rng.random() < 0.15:
        toks.extend(["TEMP", "TABLE"])
    else:
        toks.append("TABLE")
    if rng.random() < 0.25:
        toks.extend(["IF", "NOT", "EXISTS"])
    toks.extend([tbl, "(", *cols, ")"])
    return _detokenize(toks)

def _make_ddl_misc(rng):
    tbl = _rand_ident(rng)
    r = rng.random()
    if r < 0.25:
        return _detokenize(["CREATE", "INDEX", _rand_ident(rng), "ON", tbl, "(", _rand_ident(rng), ")"])
    if r < 0.45:
        return _detokenize(["CREATE", "UNIQUE", "INDEX", _rand_ident(rng), "ON", tbl, "(", _rand_ident(rng), ")"])
    if r < 0.60:
        return _detokenize(["DROP", "TABLE", "IF", "EXISTS", tbl])
    if r < 0.75:
        return _detokenize(["ALTER", "TABLE", tbl, "ADD", "COLUMN", _rand_ident(rng), rng.choice(_SQL_TYPES)])
    if r < 0.88:
        return _detokenize(["CREATE", "VIEW", _rand_ident(rng), "AS", _make_select(rng, 2)])
    return _detokenize(["DROP", "INDEX", "IF", "EXISTS", _rand_ident(rng)])

def _make_txn(rng):
    return rng.choice([
        "BEGIN",
        "BEGIN TRANSACTION",
        "COMMIT",
        "ROLLBACK",
        "SAVEPOINT " + _rand_ident(rng),
        "RELEASE " + _rand_ident(rng),
        "ROLLBACK TO " + _rand_ident(rng),
    ])

def _add_semicolon(rng, s):
    if s is None:
        return ""
    if rng.random() < 0.65 and not s.rstrip().endswith(";"):
        return s.rstrip() + ";"
    return s

def _inject_noise(rng, s):
    if not s:
        return s
    if rng.random() < 0.20:
        sn = rng.choice(_COMMENT_SNIPS)
        pos = rng.randrange(0, len(s) + 1)
        s = s[:pos] + sn + s[pos:]
    if rng.random() < 0.30:
        ws = rng.choice(_WS_SNIPS)
        pos = rng.randrange(0, len(s) + 1)
        s = s[:pos] + ws + s[pos:]
    if rng.random() < 0.08:
        s = "\ufeff" + s
    if rng.random() < 0.04:
        s = s + "\x00"
    if rng.random() < 0.10:
        # case flipping
        s = "".join((ch.lower() if ch.isupper() and rng.random() < 0.5 else ch.upper() if ch.islower() and rng.random() < 0.5 else ch) for ch in s)
    return s

def _mutate(rng, s):
    if not s:
        return s
    ops = rng.randrange(2, 6)
    for _ in range(ops):
        r = rng.random()
        if r < 0.20:
            # delete slice
            if len(s) > 3:
                a = rng.randrange(0, len(s))
                b = min(len(s), a + rng.randrange(1, 1 + min(12, len(s) - a)))
                s = s[:a] + s[b:]
        elif r < 0.40:
            # duplicate slice
            if len(s) > 3:
                a = rng.randrange(0, len(s))
                b = min(len(s), a + rng.randrange(1, 1 + min(16, len(s) - a)))
                frag = s[a:b]
                pos = rng.randrange(0, len(s) + 1)
                s = s[:pos] + frag + s[pos:]
        elif r < 0.60:
            # insert keyword
            kw = rng.choice(KEYWORDS) if KEYWORDS else rng.choice(["SELECT", "FROM", "WHERE"])
            pos = rng.randrange(0, len(s) + 1)
            s = s[:pos] + " " + kw + " " + s[pos:]
        elif r < 0.78:
            # replace number
            s = re.sub(r"\b\d+\b", lambda m: str(rng.randrange(0, 100000)), s, count=1)
        else:
            # toggle quotes around a token
            m = re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\b", s)
            if m:
                tok = m.group(0)
                q = rng.choice(['"', "`", "["])
                if q == "[":
                    rep = "[" + tok + "]"
                else:
                    rep = q + tok + q
                s = s[:m.start()] + rep + s[m.end():]
        if len(s) > 4000:
            s = s[:4000]
    return s

class _GrammarGen:
    __slots__ = ("g", "rng")
    def __init__(self, g, rng):
        self.g = g
        self.rng = rng

    def gen(self, start=None, depth=10):
        if self.g is None:
            return None
        sym = start or self.g.start
        toks = self._gen_node(_EBNFNode("nt", sym), depth, 0, set())
        if toks is None:
            return None
        s = _detokenize(toks)
        return s

    def _gen_node(self, node, depth, steps, stack):
        if steps > 2000:
            return []
        k = node.k
        if k == "t":
            v = node.a
            if v is None:
                return []
            if v == "ε" or v == "EPS" or v == "epsilon":
                return []
            return [v]
        if k == "seq":
            out = []
            for ch in node.a:
                out.extend(self._gen_node(ch, depth, steps + 1, stack))
            return out
        if k == "choice":
            alts = node.a
            if not alts:
                return []
            # bias towards shorter expansions if depth is low
            if depth <= 2 and len(alts) > 1 and self.rng.random() < 0.75:
                pick = alts[0]
                best = None
                for a in alts:
                    c = self._estimate_nt(a)
                    if best is None or c < best:
                        best = c
                        pick = a
                return self._gen_node(pick, depth, steps + 1, stack)
            return self._gen_node(self.rng.choice(alts), depth, steps + 1, stack)
        if k == "opt":
            if depth <= 1:
                if self.rng.random() < 0.75:
                    return []
            else:
                if self.rng.random() < 0.5:
                    return []
            return self._gen_node(node.a, depth, steps + 1, stack)
        if k == "rep":
            lo, hi = node.b or (0, 2)
            if depth <= 1:
                hi = min(hi, 1)
            n = lo + self.rng.randrange(0, max(1, hi - lo + 1))
            out = []
            for _ in range(n):
                out.extend(self._gen_node(node.a, depth, steps + 1, stack))
            return out
        if k == "nt":
            name = node.a
            if self.g is None:
                return []
            alts = self.g.alts.get(name)
            if not alts:
                # fallback: treat as identifier-ish
                return [name]
            if depth <= 0:
                # try shortest alt
                pick = alts[0]
                best = None
                for a in alts:
                    c = self._estimate_nt(a)
                    if best is None or c < best:
                        best = c
                        pick = a
                return self._gen_node(pick, depth, steps + 1, stack)
            if name in stack and depth <= 2:
                # break recursion
                pick = alts[0]
                best = None
                for a in alts:
                    c = self._estimate_nt(a)
                    if best is None or c < best:
                        best = c
                        pick = a
                return self._gen_node(pick, depth - 1, steps + 1, stack)
            stack.add(name)
            pick = self.rng.choice(alts)
            out = self._gen_node(pick, depth - 1, steps + 1, stack)
            stack.remove(name)
            return out
        return []

    def _estimate_nt(self, node):
        # rough count of nonterminals
        k = node.k
        if k == "nt":
            return 1
        if k == "t":
            return 0
        if k == "seq":
            return sum(self._estimate_nt(x) for x in node.a)
        if k == "choice":
            return min((self._estimate_nt(x) for x in node.a), default=0)
        if k in ("opt",):
            return self._estimate_nt(node.a)
        if k == "rep":
            return self._estimate_nt(node.a)
        return 0

class _State:
    __slots__ = (
        "rng", "start", "calls", "batch_size", "pool", "templates", "seeds",
        "gg", "g", "start_time", "last_dur", "phase"
    )
    def __init__(self):
        self.rng = random.Random(0xBADC0DE)
        self.start = time.perf_counter()
        self.calls = 0
        self.batch_size = 350
        self.pool = []
        self.templates = []
        self.seeds = []
        self.g = _parse_grammar(GRAMMAR_TEXT)
        self.gg = _GrammarGen(self.g, self.rng) if self.g is not None else None
        self.start_time = time.perf_counter()
        self.last_dur = 0.0
        self.phase = 0
        self._init_corpus()

    def _init_corpus(self):
        seeds = [
            "SELECT 1",
            "SELECT * FROM t",
            "SELECT a, b FROM t WHERE a = 1",
            "SELECT DISTINCT a FROM t GROUP BY a HAVING COUNT(*) > 1 ORDER BY a DESC LIMIT 10 OFFSET 5",
            "SELECT a FROM t JOIN u ON t.id = u.id",
            "SELECT a FROM t LEFT OUTER JOIN u USING(id)",
            "SELECT CASE WHEN a > 0 THEN 'pos' ELSE 'neg' END FROM t",
            "SELECT a IN (1,2,3) FROM t",
            "SELECT EXISTS (SELECT 1)",
            "WITH cte AS (SELECT 1 AS x) SELECT x FROM cte",
            "VALUES (1), (2), (3)",
            "INSERT INTO t VALUES (1, 'x')",
            "INSERT INTO t(a,b) VALUES (1,'x'), (2,'y')",
            "INSERT INTO t(a) SELECT b FROM u",
            "UPDATE t SET a = a + 1 WHERE id BETWEEN 1 AND 10",
            "DELETE FROM t WHERE id IN (1,2,3)",
            "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL, price REAL DEFAULT 0.0, created_at TIMESTAMP)",
            "CREATE INDEX idx_t_name ON t(name)",
            "DROP TABLE IF EXISTS t",
            "ALTER TABLE t ADD COLUMN c TEXT",
            "BEGIN",
            "COMMIT",
            "ROLLBACK",
            "SELECT \"weird\".\"name\" FROM \"t\"",
            "SELECT [select], `from` FROM [t] WHERE `where` IS NOT NULL",
            "SELECT X'414243' AS blobval",
            "SELECT 1 /*comment*/ + 2 --tail\n",
            "SELECT 'unclosed",
            "SELECT \"unclosed",
            "SELECT ((",
            "SELECT )",
            "INSERT INTO t VALUES (NULL, TRUE, FALSE, 1e-3, 0x10)",
            "SELECT a LIKE '%x%' ESCAPE '\\\\' FROM t",
        ]
        self.seeds = [s for s in seeds if s]
        self.templates = []
        # Pre-generate template-based statements
        for _ in range(900):
            r = self.rng.random()
            if r < 0.44:
                s = _make_select(self.rng, depth=2)
            elif r < 0.60:
                s = _make_insert(self.rng)
            elif r < 0.75:
                s = _make_update(self.rng)
            elif r < 0.86:
                s = _make_delete(self.rng)
            elif r < 0.95:
                s = _make_create_table(self.rng)
            else:
                s = _make_ddl_misc(self.rng)
            s = _add_semicolon(self.rng, _inject_noise(self.rng, s))
            if s and len(s) <= 4000:
                self.templates.append(s)

        # Pre-generate grammar-based statements
        if self.gg is not None:
            for d in (6, 8, 10, 12, 14):
                for _ in range(220):
                    s = self.gg.gen(depth=d)
                    if not s:
                        continue
                    s = _add_semicolon(self.rng, _inject_noise(self.rng, s))
                    if 1 <= len(s) <= 4000:
                        self.pool.append(s)

        # Mix in seeds and extra edgecases
        extra = [
            "SELECT ?",
            "SELECT :name, @x, $y",
            "SELECT 1e309, -1e309, 0.0000000000000000001",
            "SELECT 1/0",
            "SELECT ~1, 1<<2, 8>>1, 1&3, 1|2",
            "SELECT NULL IS NULL, 1 IS NOT NULL",
            "SELECT 'a''b', \"a\"\"b\"",
            "SELECT /*a*/1/*b*/",
            "SELECT --x\n 1",
            "CREATE VIEW v AS SELECT * FROM t",
            "DROP INDEX IF EXISTS idx_t_name",
            "ALTER TABLE t RENAME TO t2",
        ]
        for s in extra:
            self.seeds.append(s)

        for s in self.seeds:
            self.pool.append(_add_semicolon(self.rng, _inject_noise(self.rng, s)))

        # A few intentionally gnarly inputs
        for _ in range(60):
            s = self.rng.choice(self.pool)
            s = _mutate(self.rng, _inject_noise(self.rng, s))
            s = _add_semicolon(self.rng, s)
            if s and len(s) <= 4000:
                self.pool.append(s)

    def _gen_one(self):
        r = self.rng.random()
        if self.calls < 3:
            # front-load diversity
            if r < 0.45 and self.templates:
                return self.rng.choice(self.templates)
            if r < 0.80 and self.pool:
                return self.rng.choice(self.pool)
            if self.gg is not None:
                s = self.gg.gen(depth=10)
                if s:
                    return _add_semicolon(self.rng, _inject_noise(self.rng, s))
            return _add_semicolon(self.rng, _inject_noise(self.rng, _make_select(self.rng, 2)))

        if r < 0.36 and self.templates:
            return self.rng.choice(self.templates)
        if r < 0.62 and self.pool:
            return self.rng.choice(self.pool)
        if r < 0.78 and self.gg is not None:
            s = self.gg.gen(depth=8 + self.rng.randrange(0, 8))
            if s:
                return _add_semicolon(self.rng, _inject_noise(self.rng, s))
        # mutate from pool
        base = self.rng.choice(self.pool) if self.pool else _make_select(self.rng, 2)
        s = _mutate(self.rng, base)
        s = _inject_noise(self.rng, s)
        s = _add_semicolon(self.rng, s)
        return s

    def _make_batch(self):
        bs = self.batch_size
        batch = []
        seen = set()
        tries = 0
        while len(batch) < bs and tries < bs * 6:
            tries += 1
            s = self._gen_one()
            if not s:
                continue
            if len(s) > 4000:
                s = s[:4000]
            # Deduplicate within batch to reduce wasted parses
            if s in seen:
                continue
            seen.add(s)
            batch.append(s)

        # Keep evolving pool with a subset of new mutations
        if batch and (self.calls % 2 == 1):
            addn = min(30, len(batch))
            for i in range(addn):
                self.pool.append(batch[self.rng.randrange(0, len(batch))])
            if len(self.pool) > 7000:
                # trim to keep memory stable
                del self.pool[:1500]
        return batch

    def fuzz(self, parse_sql):
        now = time.perf_counter()
        if now - self.start_time > 55.0:
            return False

        batch = self._make_batch()
        t0 = time.perf_counter()
        parse_sql(batch)
        t1 = time.perf_counter()
        self.last_dur = t1 - t0

        # adapt batch size to keep parse_sql calls low but avoid timeouts
        if self.last_dur < 0.15:
            self.batch_size = min(1400, int(self.batch_size * 1.25) + 10)
        elif self.last_dur < 0.30:
            self.batch_size = min(1400, int(self.batch_size * 1.12) + 5)
        elif self.last_dur > 1.10:
            self.batch_size = max(60, int(self.batch_size * 0.65))
        elif self.last_dur > 0.70:
            self.batch_size = max(80, int(self.batch_size * 0.82))

        self.calls += 1
        return True

_STATE = None

def fuzz(parse_sql):
    global _STATE
    if _STATE is None:
        _STATE = _State()
    return _STATE.fuzz(parse_sql)
'''
        code = code_template.replace("__GRAMMAR_TEXT__", repr(grammar_text)).replace("__KEYWORDS_LIST__", repr(keywords_list))
        code = textwrap.dedent(code)
        return {"code": code}