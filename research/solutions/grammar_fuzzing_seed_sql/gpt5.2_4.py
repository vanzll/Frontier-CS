import os
import sys
import re
import random
import itertools
import inspect
from dataclasses import is_dataclass, fields
from enum import Enum


class _GTerminal:
    __slots__ = ("value",)

    def __init__(self, value: str):
        self.value = value


class _GNonTerm:
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name


class _GSeq:
    __slots__ = ("items",)

    def __init__(self, items):
        self.items = items


class _GAlt:
    __slots__ = ("alts",)

    def __init__(self, alts):
        self.alts = alts


class _GOpt:
    __slots__ = ("node",)

    def __init__(self, node):
        self.node = node


class _GRep:
    __slots__ = ("node", "min1")

    def __init__(self, node, min1: bool):
        self.node = node
        self.min1 = min1


class _GEps:
    __slots__ = ()

    def __init__(self):
        pass


class _GrammarParser:
    def __init__(self, text: str):
        self.tokens = self._lex(text)
        self.i = 0

    @staticmethod
    def _lex(text: str):
        tokens = []
        i = 0
        n = len(text)
        specials = set("|()[]{}*+?,;.")
        while i < n:
            c = text[i]
            if c.isspace():
                i += 1
                continue
            if c in specials:
                tokens.append(c)
                i += 1
                continue
            if c in ("'", '"'):
                q = c
                i += 1
                buf = []
                while i < n:
                    ch = text[i]
                    if ch == q:
                        if i + 1 < n and text[i + 1] == q:
                            buf.append(q)
                            i += 2
                            continue
                        i += 1
                        break
                    if ch == "\\" and i + 1 < n:
                        buf.append(text[i + 1])
                        i += 2
                        continue
                    buf.append(ch)
                    i += 1
                tokens.append(("LIT", "".join(buf)))
                continue
            if c == "<":
                j = text.find(">", i + 1)
                if j == -1:
                    j = n - 1
                name = text[i + 1 : j].strip()
                tokens.append(("NT", name))
                i = j + 1
                continue
            j = i
            while j < n and (not text[j].isspace()) and (text[j] not in specials) and (text[j] not in ("'", '"')):
                if text[j] == "<":
                    break
                j += 1
            word = text[i:j]
            if word:
                tokens.append(("WORD", word))
                i = j
            else:
                i += 1
        return tokens

    def _peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def _eat(self, expected=None):
        tok = self._peek()
        if expected is not None and tok != expected:
            raise ValueError(f"Expected {expected}, got {tok}")
        self.i += 1
        return tok

    def parse_expr(self):
        alts = [self.parse_seq()]
        while self._peek() == "|":
            self._eat("|")
            alts.append(self.parse_seq())
        if len(alts) == 1:
            return alts[0]
        return _GAlt(alts)

    def parse_seq(self):
        items = []
        while True:
            tok = self._peek()
            if tok is None or tok in (")", "]", "}", "|"):
                break
            items.append(self.parse_term())
        if not items:
            return _GEps()
        if len(items) == 1:
            return items[0]
        return _GSeq(items)

    def parse_term(self):
        node = self.parse_factor()
        while True:
            tok = self._peek()
            if tok in ("?", "*", "+"):
                op = self._eat()
                if op == "?":
                    node = _GOpt(node)
                elif op == "*":
                    node = _GRep(node, min1=False)
                else:
                    node = _GRep(node, min1=True)
            else:
                break
        return node

    def parse_factor(self):
        tok = self._peek()
        if tok is None:
            return _GEps()
        if tok == "(":
            self._eat("(")
            node = self.parse_expr()
            if self._peek() == ")":
                self._eat(")")
            return node
        if tok == "[":
            self._eat("[")
            node = self.parse_expr()
            if self._peek() == "]":
                self._eat("]")
            return _GOpt(node)
        if tok == "{":
            self._eat("{")
            node = self.parse_expr()
            if self._peek() == "}":
                self._eat("}")
            return _GRep(node, min1=False)
        if isinstance(tok, tuple) and tok[0] == "LIT":
            self._eat()
            return _GTerminal(tok[1])
        if isinstance(tok, tuple) and tok[0] == "NT":
            self._eat()
            return _GNonTerm(tok[1])
        if isinstance(tok, tuple) and tok[0] == "WORD":
            self._eat()
            return _GTerminal(tok[1])
        if isinstance(tok, str):
            self._eat()
            return _GTerminal(tok)
        self._eat()
        return _GEps()


def _strip_grammar_comments(line: str) -> str:
    s = line
    for sep in ("#", "//"):
        if sep in s:
            s = s.split(sep, 1)[0]
    return s.rstrip("\n")


def _read_grammar_rules(grammar_path: str):
    if not os.path.exists(grammar_path):
        return [], {}
    lines = []
    with open(grammar_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = _strip_grammar_comments(raw).strip()
            if not s:
                continue
            lines.append(s)

    rules = []
    curr = None
    for s in lines:
        if "::=" in s:
            if curr:
                rules.append(curr)
            curr = s
        else:
            if curr:
                curr = curr + " " + s
            else:
                curr = s
    if curr:
        rules.append(curr)

    prod = {}
    order = []
    for rule in rules:
        if "::=" not in rule:
            continue
        lhs, rhs = rule.split("::=", 1)
        lhs = lhs.strip()
        m = re.match(r"<\s*([^>]+)\s*>$", lhs)
        if m:
            lhs_name = m.group(1).strip()
        else:
            lhs_name = lhs.strip()
            if lhs_name.startswith("<") and lhs_name.endswith(">"):
                lhs_name = lhs_name[1:-1].strip()
        rhs = rhs.strip()
        try:
            gp = _GrammarParser(rhs)
            node = gp.parse_expr()
            prod[lhs_name] = node
            order.append(lhs_name)
        except Exception:
            continue
    return order, prod


def _normalize_sql_from_tokens(tokens):
    if not tokens:
        return ""
    s = " ".join(t for t in tokens if t is not None and t != "")
    s = re.sub(r"\s+", " ", s).strip()

    s = re.sub(r"\s+([,;\)\]])", r"\1", s)
    s = re.sub(r"([\(\[])\s+", r"\1", s)
    s = re.sub(r"\s+\.", ".", s)
    s = re.sub(r"\.\s+", ".", s)
    s = re.sub(r"\s+::\s+", "::", s)
    s = re.sub(r"\s*=\s*", " = ", s)
    s = re.sub(r"\s*<>\s*", " <> ", s)
    s = re.sub(r"\s*!=\s*", " != ", s)
    s = re.sub(r"\s*<=\s*", " <= ", s)
    s = re.sub(r"\s*>=\s*", " >= ", s)
    s = re.sub(r"\s*<\s*", " < ", s)
    s = re.sub(r"\s*>\s*", " > ", s)
    s = re.sub(r"\s*\+\s*", " + ", s)
    s = re.sub(r"\s*-\s*", " - ", s)
    s = re.sub(r"\s*/\s*", " / ", s)
    s = re.sub(r"\s*\*\s*", " * ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_PLACEHOLDER_TERMINALS = {
    "IDENTIFIER": "t",
    "IDENT": "t",
    "ID": "t",
    "NAME": "t",
    "TABLE_NAME": "t",
    "COLUMN_NAME": "c",
    "STRING_LITERAL": "'x'",
    "STRING": "'x'",
    "TEXT_LITERAL": "'x'",
    "NUMBER": "1",
    "NUMERIC_LITERAL": "1",
    "INTEGER_LITERAL": "1",
    "INT_LITERAL": "1",
    "FLOAT_LITERAL": "1.5",
    "DECIMAL_LITERAL": "1.5",
    "BOOL_LITERAL": "TRUE",
    "BOOLEAN_LITERAL": "TRUE",
    "NULL_LITERAL": "NULL",
    "PARAM": "?",
    "PLACEHOLDER": "?",
}


def _map_terminal(term: str) -> str:
    if term is None:
        return ""
    t = term.strip()
    if not t:
        return ""
    if t.lower() in ("epsilon", "empty", "/*empty*/", "/* empty */"):
        return ""
    key = t.upper()
    if key in _PLACEHOLDER_TERMINALS:
        return _PLACEHOLDER_TERMINALS[key]
    if key in ("<IDENTIFIER>", "<NAME>"):
        return "t"
    if t == "''":
        return "''"
    return t


class _GrammarGenerator:
    def __init__(self, order, prod):
        self.order = order
        self.prod = prod
        self.memo = {}  # (name, depth) -> list[list[str]]
        self.max_out_per_symbol = 30
        self.max_out_per_node = 60
        self.max_prod_combos = 60

    def expand_nonterm(self, name: str, depth: int):
        key = (name, depth)
        if key in self.memo:
            return self.memo[key]
        if depth <= 0:
            out = [[self._fallback_for_nonterm(name)]]
            self.memo[key] = out
            return out
        node = self.prod.get(name)
        if node is None:
            out = [[self._fallback_for_nonterm(name)]]
            self.memo[key] = out
            return out
        res = self._gen_node(node, depth)
        res2 = []
        for toks in res:
            toks2 = [t for t in toks if t]
            res2.append(toks2)
        seen = set()
        dedup = []
        for toks in res2:
            sig = tuple(toks)
            if sig in seen:
                continue
            seen.add(sig)
            dedup.append(toks)
            if len(dedup) >= self.max_out_per_symbol:
                break
        self.memo[key] = dedup
        return dedup

    def _fallback_for_nonterm(self, name: str) -> str:
        n = name.strip().lower()
        if any(k in n for k in ("ident", "name", "table", "schema", "db")):
            return "t"
        if "column" in n or n.endswith("col") or n.endswith("field"):
            return "c"
        if "string" in n or "text" in n:
            return "'x'"
        if "number" in n or "int" in n or "digit" in n or "numeric" in n:
            return "1"
        if "float" in n or "real" in n or "decimal" in n:
            return "1.5"
        if "bool" in n:
            return "TRUE"
        if "null" in n:
            return "NULL"
        if "operator" in n:
            return "+"
        return "t"

    def _gen_node(self, node, depth: int):
        if depth <= 0:
            return [[]]
        if isinstance(node, _GEps):
            return [[]]
        if isinstance(node, _GTerminal):
            t = _map_terminal(node.value)
            return [[t]] if t != "" else [[]]
        if isinstance(node, _GNonTerm):
            return self.expand_nonterm(node.name, depth - 1)
        if isinstance(node, _GOpt):
            a = [[]]
            b = self._gen_node(node.node, depth - 1)
            out = a + b
            return out[: self.max_out_per_node]
        if isinstance(node, _GRep):
            child = self._gen_node(node.node, depth - 1)
            out = []
            min_rep = 1 if node.min1 else 0
            max_rep = 2
            for k in range(min_rep, max_rep + 1):
                if k == 0:
                    out.append([])
                    continue
                combos = [[]]
                for _ in range(k):
                    new_combos = []
                    for prefix in combos:
                        for part in child[:10]:
                            new = prefix + part
                            new_combos.append(new)
                            if len(new_combos) >= self.max_prod_combos:
                                break
                        if len(new_combos) >= self.max_prod_combos:
                            break
                    combos = new_combos
                out.extend(combos)
                if len(out) >= self.max_out_per_node:
                    break
            return out[: self.max_out_per_node]
        if isinstance(node, _GAlt):
            out = []
            for alt in node.alts:
                out.extend(self._gen_node(alt, depth - 1))
                if len(out) >= self.max_out_per_node:
                    break
            return out[: self.max_out_per_node]
        if isinstance(node, _GSeq):
            parts = [self._gen_node(it, depth - 1) for it in node.items]
            combos = [[]]
            for plist in parts:
                new_combos = []
                plist2 = plist[:15] if len(plist) > 15 else plist
                for prefix in combos:
                    for suffix in plist2:
                        new_combos.append(prefix + suffix)
                        if len(new_combos) >= self.max_prod_combos:
                            break
                    if len(new_combos) >= self.max_prod_combos:
                        break
                combos = new_combos
                if not combos:
                    break
            return combos[: self.max_out_per_node]
        return [[]]


def _safe_parse(parse_sql, sql: str):
    try:
        return True, parse_sql(sql)
    except Exception:
        return False, None


def _try_variants(parse_sql, sql: str):
    variants = [sql]
    s = sql.strip()
    if s.endswith(";"):
        variants.append(s[:-1].rstrip())
    else:
        variants.append(s + ";")
    if "\n" in s:
        variants.append(s.replace("\n", " "))
    if "\t" in s:
        variants.append(s.replace("\t", " "))
    for v in variants:
        ok, ast = _safe_parse(parse_sql, v)
        if ok:
            return v, ast
    return None, None


def _collect_ast_features(obj, features: set, ast_mod_name: str, seen_ids: set):
    if obj is None:
        return
    oid = id(obj)
    if oid in seen_ids:
        return
    seen_ids.add(oid)

    if isinstance(obj, (str, int, float, bool, bytes)):
        return
    if isinstance(obj, Enum):
        features.add(f"ENUM:{obj.__class__.__name__}:{obj.name}")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _collect_ast_features(k, features, ast_mod_name, seen_ids)
            _collect_ast_features(v, features, ast_mod_name, seen_ids)
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for it in obj:
            _collect_ast_features(it, features, ast_mod_name, seen_ids)
        return

    cls = obj.__class__
    if cls.__module__.endswith(ast_mod_name) or cls.__module__.split(".")[-1] == ast_mod_name:
        features.add(f"AST:{cls.__name__}")
    else:
        features.add(f"OBJMOD:{cls.__module__.split('.')[-1]}:{cls.__name__}")

    if is_dataclass(obj):
        try:
            for f in fields(obj):
                try:
                    val = getattr(obj, f.name)
                except Exception:
                    continue
                _collect_ast_features(val, features, ast_mod_name, seen_ids)
            return
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        try:
            for _, v in vars(obj).items():
                _collect_ast_features(v, features, ast_mod_name, seen_ids)
            return
        except Exception:
            pass

    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        if attr in ("parent",):
            continue
        try:
            val = getattr(obj, attr)
        except Exception:
            continue
        if callable(val):
            continue
        _collect_ast_features(val, features, ast_mod_name, seen_ids)


def _regex_token_features(sql: str):
    feats = set()
    up_words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sql)
    for w in up_words:
        if len(w) <= 30:
            feats.add("W:" + w.upper())
    for ch in [",", ";", "(", ")", ".", "*", "+", "-", "/", "%", "=", "<", ">", "!", "|", "&"]:
        if ch in sql:
            feats.add("CH:" + ch)
    for op in ["<>", "!=", "<=", ">=", "||"]:
        if op in sql:
            feats.add("OP:" + op)
    if "--" in sql:
        feats.add("COMMENT:--")
    if "/*" in sql:
        feats.add("COMMENT:/*")
    if "'" in sql:
        feats.add("QUOTE:'")
    if '"' in sql:
        feats.add('QUOTE:"')
    if "`" in sql:
        feats.add("QUOTE:`")
    if "[" in sql and "]" in sql:
        feats.add("QUOTE:[]")
    if "?" in sql:
        feats.add("PARAM:?")

    if re.search(r"\b\d+\.\d+([eE][+-]?\d+)?\b", sql):
        feats.add("NUM:FLOAT")
    if re.search(r"\b\d+[eE][+-]?\d+\b", sql):
        feats.add("NUM:EXP")
    if re.search(r"\b\d+\b", sql):
        feats.add("NUM:INT")
    return feats


def _safe_tokenize_features(tokenizer_module, sql: str):
    if tokenizer_module is None:
        return _regex_token_features(sql)

    # Try tokenize(sql) -> list[Token]
    toks = None
    try:
        if hasattr(tokenizer_module, "tokenize") and callable(getattr(tokenizer_module, "tokenize")):
            toks = tokenizer_module.tokenize(sql)
    except Exception:
        toks = None

    # Try Tokenizer class
    if toks is None:
        try:
            Tok = getattr(tokenizer_module, "Tokenizer", None)
            if Tok is not None:
                t = Tok(sql)
                if hasattr(t, "tokenize") and callable(getattr(t, "tokenize")):
                    toks = t.tokenize()
                elif hasattr(t, "__iter__"):
                    toks = list(iter(t))
                else:
                    toks = None
        except Exception:
            toks = None

    if toks is None:
        return _regex_token_features(sql)

    feats = set()
    for tok in toks:
        kind = None
        val = None
        for a in ("kind", "type", "token_type", "ttype", "name"):
            if hasattr(tok, a):
                try:
                    kind = getattr(tok, a)
                    break
                except Exception:
                    pass
        for a in ("value", "text", "lexeme", "literal", "raw"):
            if hasattr(tok, a):
                try:
                    val = getattr(tok, a)
                    break
                except Exception:
                    pass

        if isinstance(kind, Enum):
            feats.add(f"TK:{kind.__class__.__name__}.{kind.name}")
        elif kind is not None:
            ks = str(kind)
            if len(ks) <= 60:
                feats.add("TK:" + ks)

        if isinstance(val, Enum):
            feats.add(f"TVE:{val.__class__.__name__}.{val.name}")
        elif val is not None:
            vs = str(val)
            if len(vs) <= 30:
                feats.add("TV:" + vs.upper())

    feats |= _regex_token_features(sql)
    return feats


def _extract_keywords_from_tokenizer(tokenizer_module):
    kws = set()
    if tokenizer_module is None:
        return kws
    for attr in ("KEYWORDS", "RESERVED_KEYWORDS", "RESERVED_WORDS", "KEYWORD_MAP", "keyword_map", "keywords", "reserved_words"):
        if hasattr(tokenizer_module, attr):
            try:
                obj = getattr(tokenizer_module, attr)
            except Exception:
                continue
            try:
                if isinstance(obj, dict):
                    for k in obj.keys():
                        if isinstance(k, str) and k.isalpha():
                            kws.add(k.upper())
                elif isinstance(obj, (set, list, tuple)):
                    for k in obj:
                        if isinstance(k, str) and k.isalpha():
                            kws.add(k.upper())
            except Exception:
                pass
    return kws


def _extract_keywords_from_sources(resources_path: str):
    kws = set()
    for fn in ("parser.py", "tokenizer.py"):
        p = os.path.join(resources_path, "sql_engine", fn)
        if not os.path.exists(p):
            continue
        try:
            src = open(p, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for m in re.finditer(r"['\"]([A-Z][A-Z0-9_]{1,24})['\"]", src):
            s = m.group(1)
            if s.isalpha():
                kws.add(s.upper())
        for m in re.finditer(r"\b([A-Z][A-Z0-9_]{2,24})\b", src):
            s = m.group(1)
            if s.isalpha():
                kws.add(s.upper())
    noisy = {"TRUE", "FALSE", "NULL"}
    kws |= noisy
    return kws


def _manual_candidates(kws: set):
    cands = set()

    def add(s: str):
        s2 = s.strip()
        if s2:
            cands.add(s2)

    # Baseline queries
    add("SELECT 1")
    add("select 1")
    add("SELECT 1 + 2 AS sum")
    add("SELECT -1 AS neg, +2 AS pos")
    add("SELECT 1.23e-4 AS small, 1e6 AS big")
    add("SELECT 'a' AS s, 'a''b' AS esc")
    add('SELECT 1 AS "col", 2 AS "col2"')
    add("SELECT NULL, TRUE, FALSE")

    # Comments for tokenizer paths
    add("-- leading line comment\nSELECT 1")
    add("/* leading block comment */ SELECT 1")
    add("SELECT 1 /* trailing block comment */")
    add("SELECT 1 -- trailing line comment")

    # FROM variants
    add("SELECT * FROM t")
    add("SELECT t.c FROM t AS t")
    add("SELECT c FROM t WHERE c = 1")
    add("SELECT c FROM t WHERE c <> 1")
    add("SELECT c FROM t WHERE c != 1")
    add("SELECT c FROM t WHERE c < 1 OR c <= 1 OR c > 1 OR c >= 1")
    add("SELECT c FROM t WHERE NOT c = 1")
    add("SELECT c FROM t WHERE NOT (c = 1 OR c = 2)")
    add("SELECT c FROM t WHERE c IN (1, 2, 3)")
    add("SELECT c FROM t WHERE c NOT IN (1, 2, 3)")
    add("SELECT c FROM t WHERE c BETWEEN 1 AND 10")
    add("SELECT c FROM t WHERE c NOT BETWEEN 1 AND 10")
    add("SELECT c FROM t WHERE c IS NULL")
    add("SELECT c FROM t WHERE c IS NOT NULL")
    add("SELECT c FROM t WHERE c LIKE 'a%'")
    add("SELECT c FROM t WHERE c NOT LIKE '%z%'")
    add("SELECT c FROM t WHERE c = 'x' OR c = 'y' AND c <> 'z'")
    add("SELECT c FROM t WHERE (c = 'x' OR c = 'y') AND c <> 'z'")
    add("SELECT c FROM t WHERE c IN (SELECT c FROM t2)")
    add("SELECT c FROM t WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.c = t.c)")
    add("SELECT c FROM (SELECT 1 AS c) AS sub")
    add("SELECT sub.c FROM (SELECT 1 AS c) sub")
    add("SELECT * FROM t1, t2 WHERE t1.id = t2.id")

    # Joins
    add("SELECT t1.c, t2.d FROM t1 JOIN t2 ON t1.id = t2.id")
    add("SELECT t1.c, t2.d FROM t1 INNER JOIN t2 ON t1.id = t2.id")
    add("SELECT t1.c, t2.d FROM t1 LEFT JOIN t2 ON t1.id = t2.id")
    add("SELECT t1.c, t2.d FROM t1 RIGHT JOIN t2 ON t1.id = t2.id")
    add("SELECT t1.c, t2.d FROM t1 FULL JOIN t2 ON t1.id = t2.id")
    add("SELECT t1.c, t2.d FROM t1 CROSS JOIN t2")

    # Aggregation
    add("SELECT c, COUNT(*) FROM t GROUP BY c")
    add("SELECT c, COUNT(*) AS n FROM t GROUP BY c HAVING COUNT(*) > 1")
    add("SELECT DISTINCT c FROM t")
    add("SELECT ALL c FROM t")

    # Ordering and limiting
    add("SELECT c FROM t ORDER BY c")
    add("SELECT c FROM t ORDER BY c ASC")
    add("SELECT c FROM t ORDER BY c DESC")
    add("SELECT c FROM t ORDER BY c DESC, c ASC")
    add("SELECT c FROM t LIMIT 10")
    add("SELECT c FROM t LIMIT 10 OFFSET 5")
    add("SELECT c FROM t OFFSET 5 LIMIT 10")

    # Expressions and functions
    add("SELECT (1 + 2) * 3 AS expr")
    add("SELECT 1 + 2 * 3 - 4 / 2 AS prec")
    add("SELECT COALESCE(NULL, 1) AS co")
    add("SELECT NULLIF(1, 1) AS nu")
    add("SELECT CAST(1 AS TEXT) AS ct")
    add("SELECT CASE WHEN 1 = 1 THEN 'y' ELSE 'n' END AS c")
    add("SELECT CASE 1 WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END AS c")

    # Set operations
    add("SELECT 1 UNION SELECT 2")
    add("SELECT 1 UNION ALL SELECT 2")
    add("SELECT 1 INTERSECT SELECT 1")
    add("SELECT 1 EXCEPT SELECT 2")

    # VALUES
    add("VALUES (1)")
    add("VALUES (1, 'a'), (2, 'b')")
    add("SELECT * FROM (VALUES (1), (2)) AS v(x)")

    # Parameters (if supported)
    add("SELECT ?")
    add("SELECT :p")
    add("SELECT $1")
    add("SELECT @p")
    add("SELECT * FROM t WHERE c = ?")

    # CTE
    add("WITH x AS (SELECT 1 AS a) SELECT a FROM x")
    add("WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM x WHERE n < 3) SELECT n FROM x")

    # DML/DDL (only kept if parser supports)
    add("INSERT INTO t VALUES (1)")
    add("INSERT INTO t (c1, c2) VALUES (1, 'a')")
    add("INSERT INTO t (c1, c2) VALUES (1, 'a'), (2, 'b')")
    add("INSERT INTO t SELECT c FROM t2")
    add("UPDATE t SET c = 1")
    add("UPDATE t SET c = c + 1 WHERE c < 10")
    add("UPDATE t SET c1 = 1, c2 = 'x' WHERE c1 <> 0")
    add("DELETE FROM t")
    add("DELETE FROM t WHERE c = 1")

    add("CREATE TABLE t (id INT)")
    add("CREATE TABLE t (id INTEGER PRIMARY KEY, c TEXT NOT NULL, d REAL DEFAULT 1.0)")
    add("CREATE TABLE t (id INT, c INT, UNIQUE (c))")
    add("CREATE TABLE t (id INT, c INT, CHECK (c > 0))")
    add("CREATE TABLE t2 (id INT, c INT, FOREIGN KEY (id) REFERENCES t(id))")
    add("ALTER TABLE t ADD COLUMN c2 INT")
    add("DROP TABLE t")
    add("CREATE INDEX idx_t_c ON t(c)")
    add("DROP INDEX idx_t_c")
    add("CREATE VIEW v AS SELECT * FROM t")
    add("DROP VIEW v")

    add("BEGIN")
    add("BEGIN TRANSACTION")
    add("COMMIT")
    add("ROLLBACK")

    # Prune if keyword set suggests minimal subset (but still keep broad; validation will filter)
    if kws:
        # If SELECT isn't supported, nothing will parse anyway; keep all.
        pass

    return list(cands)


def _grammar_candidates(grammar_path: str, parse_sql, max_total: int = 220):
    order, prod = _read_grammar_rules(grammar_path)
    if not prod:
        return []

    gen = _GrammarGenerator(order, prod)

    names = list(prod.keys())
    start_candidates = []
    if order:
        start_candidates.append(order[0])

    def want(name: str):
        n = name.lower()
        return any(k in n for k in ("stmt", "statement", "query", "select", "sql", "command"))

    for nm in names:
        if want(nm):
            start_candidates.append(nm)

    # Dedup while preserving order
    seen = set()
    starts = []
    for s in start_candidates:
        if s in seen:
            continue
        seen.add(s)
        starts.append(s)
        if len(starts) >= 10:
            break

    cands = set()
    for st in starts:
        toks_lists = gen.expand_nonterm(st, depth=6)
        for toks in toks_lists:
            sql = _normalize_sql_from_tokens(toks)
            if not sql:
                continue
            v, _ = _try_variants(parse_sql, sql)
            if v:
                cands.add(v.strip())
            if len(cands) >= max_total:
                break
        if len(cands) >= max_total:
            break

    return list(cands)


def _supports_multi_statement(parse_sql):
    ok1, _ = _safe_parse(parse_sql, "SELECT 1; SELECT 2")
    if not ok1:
        ok1, _ = _safe_parse(parse_sql, "SELECT 1\nSELECT 2")
        if not ok1:
            return False
    ok2, _ = _safe_parse(parse_sql, "SELECT 1; SELCT 2")
    # multi supported if invalid second statement triggers failure
    return ok1 and (not ok2)


def _pack_statements(parse_sql, statements, max_batch_chars=7000, prefer_semicolon=True):
    if not statements:
        return []
    if not _supports_multi_statement(parse_sql):
        return statements[:]

    batches = []
    curr = []

    def try_join(stmts):
        if not stmts:
            return None
        if prefer_semicolon:
            s1 = ";\n".join(stmts)
            if _safe_parse(parse_sql, s1)[0]:
                return s1
            s2 = s1 + ";"
            if _safe_parse(parse_sql, s2)[0]:
                return s2
        s3 = "\n".join(stmts)
        if _safe_parse(parse_sql, s3)[0]:
            return s3
        s4 = s3 + "\n"
        if _safe_parse(parse_sql, s4)[0]:
            return s4
        return None

    for stmt in statements:
        if not curr:
            curr = [stmt]
            continue
        # tentative
        tentative = curr + [stmt]
        if sum(len(x) for x in tentative) + 3 * (len(tentative) - 1) > max_batch_chars:
            joined = try_join(curr)
            if joined is None:
                batches.extend(curr)
            else:
                batches.append(joined)
            curr = [stmt]
            continue
        joined = try_join(tentative)
        if joined is not None:
            curr = tentative
        else:
            joined_curr = try_join(curr)
            if joined_curr is None:
                batches.extend(curr)
            else:
                batches.append(joined_curr)
            curr = [stmt]

    if curr:
        joined = try_join(curr)
        if joined is None:
            batches.extend(curr)
        else:
            batches.append(joined)

    # Final sanity: ensure each batch parses
    out = []
    for b in batches:
        if _safe_parse(parse_sql, b)[0]:
            out.append(b)
        else:
            # fallback: split by semicolon or newline
            parts = [p.strip() for p in re.split(r";\s*\n|;\s*|[\n\r]+", b) if p.strip()]
            for p in parts:
                v, _ = _try_variants(parse_sql, p)
                if v:
                    out.append(v)
    return out


class Solution:
    def solve(self, resources_path: str) -> list[str]:
        resources_path = os.path.abspath(resources_path)
        if resources_path not in sys.path:
            sys.path.insert(0, resources_path)

        parse_sql = None
        tokmod = None
        astmod = None

        try:
            import sql_engine  # type: ignore

            if hasattr(sql_engine, "parse_sql") and callable(getattr(sql_engine, "parse_sql")):
                parse_sql = sql_engine.parse_sql
            else:
                from sql_engine.parser import parse_sql as _ps  # type: ignore

                parse_sql = _ps
            try:
                from sql_engine import tokenizer as _tok  # type: ignore

                tokmod = _tok
            except Exception:
                try:
                    from sql_engine.tokenizer import tokenize as _tok2  # type: ignore

                    tokmod = sys.modules.get("sql_engine.tokenizer", None)
                except Exception:
                    tokmod = None
            try:
                from sql_engine import ast_nodes as _ast  # type: ignore

                astmod = _ast
            except Exception:
                astmod = None
        except Exception:
            return ["SELECT 1"]

        kws = set()
        kws |= _extract_keywords_from_tokenizer(tokmod)
        kws |= _extract_keywords_from_sources(resources_path)

        manual = _manual_candidates(kws)

        grammar_path = os.path.join(resources_path, "sql_grammar.txt")
        grammar_cands = []
        try:
            grammar_cands = _grammar_candidates(grammar_path, parse_sql, max_total=220)
        except Exception:
            grammar_cands = []

        # Combine and validate
        candidates = []
        seen_sql = set()

        # Ensure determinism
        random.seed(0)

        pool = manual + grammar_cands
        # Shuffle to avoid bias and to diversify early
        pool = list(dict.fromkeys(pool))
        random.shuffle(pool)

        for s in pool:
            v, ast = _try_variants(parse_sql, s)
            if not v:
                continue
            vv = v.strip()
            if vv in seen_sql:
                continue
            seen_sql.add(vv)
            candidates.append((vv, ast))

        if not candidates:
            return ["SELECT 1"]

        ast_mod_name = "ast_nodes"
        if astmod is not None:
            try:
                ast_mod_name = astmod.__name__.split(".")[-1]
            except Exception:
                ast_mod_name = "ast_nodes"

        # Build feature sets
        stmt_feats = []
        for sql, ast in candidates:
            feats = set()
            feats |= _safe_tokenize_features(tokmod, sql)
            try:
                if ast is not None:
                    _collect_ast_features(ast, feats, ast_mod_name, set())
            except Exception:
                pass
            # Encourage top-level keyword coverage
            m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", sql)
            if m:
                feats.add("TOP:" + m.group(1).upper())
            stmt_feats.append((sql, feats))

        # Greedy selection for coverage by features
        covered = set()
        selected = []
        remaining = stmt_feats[:]

        max_select = 60
        while remaining and len(selected) < max_select:
            best_idx = -1
            best_gain = 0
            best_len = None
            for i, (sql, feats) in enumerate(remaining):
                gain = len(feats - covered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_len = len(sql)
                elif gain == best_gain and gain > 0:
                    l = len(sql)
                    if best_len is None or l < best_len:
                        best_idx = i
                        best_len = l
            if best_idx == -1 or best_gain <= 0:
                break
            sql, feats = remaining.pop(best_idx)
            selected.append(sql)
            covered |= feats

        # Ensure at least a few core statements exist even if feature heuristic stalls
        if len(selected) < 8:
            core = [
                "SELECT 1",
                "SELECT * FROM t",
                "SELECT c FROM t WHERE c = 1",
                "SELECT c, COUNT(*) FROM t GROUP BY c",
                "SELECT 1 UNION SELECT 2",
                "INSERT INTO t VALUES (1)",
                "UPDATE t SET c = 1",
                "DELETE FROM t WHERE c = 1",
            ]
            for s in core:
                v, _ = _try_variants(parse_sql, s)
                if v and v not in selected:
                    selected.append(v)
                if len(selected) >= 10:
                    break

        # Prefer a small number of packed test cases if supported
        packed = _pack_statements(parse_sql, selected, max_batch_chars=8000, prefer_semicolon=True)

        # Final clean: ensure everything parses and unique
        out = []
        out_seen = set()
        for s in packed:
            v, _ = _try_variants(parse_sql, s)
            if not v:
                continue
            vv = v.strip()
            if vv in out_seen:
                continue
            out_seen.add(vv)
            out.append(vv)

        if not out:
            return ["SELECT 1"]
        return out[:60]