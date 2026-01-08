import os
import re
import ast
import time
import random
import string
from typing import Dict, List, Tuple, Optional


def _strip_comments(line: str) -> str:
    # Remove trailing comments that are likely grammar comments (not SQL comments inside quotes)
    s = line
    for sep in ("#", "//"):
        idx = s.find(sep)
        if idx != -1:
            s = s[:idx]
    return s.strip()


def _tokenize_rhs(s: str) -> List[str]:
    tokens = []
    i = 0
    n = len(s)

    specials_1 = set("[]{}()|*+?;,")
    # Include comparison/arith symbols; handle 2-char operators separately
    # '.' is included as its own token for common SQL dotted identifiers
    specials_1.update(set(".=<>!+-/%"))

    two_char_ops = {"<=", ">=", "<>", "!=", "==", ":=", "||", "&&", "::"}

    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue

        if c in ("'", '"', "`"):
            q = c
            j = i + 1
            buf = [q]
            # SQL escaping: doubled quotes for ' and "
            while j < n:
                ch = s[j]
                buf.append(ch)
                if ch == q:
                    if q in ("'", '"') and j + 1 < n and s[j + 1] == q:
                        buf.append(s[j + 1])
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            tokens.append("".join(buf))
            i = j
            continue

        if c == "<":
            j = s.find(">", i + 1)
            if j != -1:
                tokens.append(s[i:j + 1])
                i = j + 1
                continue

        if i + 1 < n and s[i:i + 2] in two_char_ops:
            tokens.append(s[i:i + 2])
            i += 2
            continue

        if c in specials_1:
            tokens.append(c)
            i += 1
            continue

        # /regex/ style
        if c == "/" and i + 1 < n:
            j = i + 1
            while j < n and s[j] != "/":
                j += 1
            if j < n and s[j] == "/":
                tokens.append(s[i:j + 1])
                i = j + 1
                continue

        # word
        j = i + 1
        while j < n:
            ch = s[j]
            if ch.isspace():
                break
            if j + 1 < n and s[j:j + 2] in two_char_ops:
                break
            if ch in specials_1:
                break
            if ch in ("'", '"', "`", "<"):
                break
            j += 1
        tokens.append(s[i:j])
        i = j

    return [t for t in tokens if t != ""]


def _split_alternatives(tokens: List[str]) -> List[List[str]]:
    alts = []
    cur = []
    depth = 0
    # track nesting for (), [], {}
    for t in tokens:
        if t in ("(", "[", "{"):
            depth += 1
            cur.append(t)
            continue
        if t in (")", "]", "}"):
            cur.append(t)
            if depth > 0:
                depth -= 1
            continue
        if t == "|" and depth == 0:
            alts.append(cur)
            cur = []
            continue
        cur.append(t)
    alts.append(cur)
    # allow empty alternative
    return [a for a in alts]


def _normalize_lhs(lhs: str) -> str:
    lhs = lhs.strip()
    if lhs.startswith("<") and lhs.endswith(">"):
        lhs = lhs[1:-1].strip()
    lhs = lhs.strip()
    return lhs


def _parse_grammar(grammar_text: str) -> Tuple[Dict[str, List[List[str]]], List[str]]:
    rules: Dict[str, List[List[str]]] = {}
    order: List[str] = []

    cur_lhs = None
    rhs_parts: List[str] = []

    for raw in grammar_text.splitlines():
        line = _strip_comments(raw)
        if not line:
            continue

        # continuation lines
        if cur_lhs is not None and (line.startswith("|") or line.lstrip().startswith("|")):
            rhs_parts.append(line.strip())
            continue

        # new rule
        m = None
        for sep in ("::=", ":=", "->"):
            if sep in line:
                parts = line.split(sep, 1)
                m = (parts[0], parts[1])
                break
        if m:
            # flush previous
            if cur_lhs is not None:
                rhs_str = " ".join(rhs_parts).strip()
                toks = _tokenize_rhs(rhs_str)
                alts = _split_alternatives(toks)
                rules.setdefault(cur_lhs, []).extend(alts)
                rhs_parts = []
            cur_lhs = _normalize_lhs(m[0])
            if cur_lhs not in rules:
                order.append(cur_lhs)
            rhs_parts = [m[1].strip()]
            continue

        # continuation without explicit '|'
        if cur_lhs is not None:
            rhs_parts.append(line.strip())

    if cur_lhs is not None:
        rhs_str = " ".join(rhs_parts).strip()
        toks = _tokenize_rhs(rhs_str)
        alts = _split_alternatives(toks)
        rules.setdefault(cur_lhs, []).extend(alts)

    # Clean rules: remove pure epsilon markers
    eps_markers = {"ε", "EPSILON", "epsilon", "EMPTY", "empty", "/* empty */"}
    cleaned: Dict[str, List[List[str]]] = {}
    for lhs, alts in rules.items():
        out_alts = []
        for alt in alts:
            alt2 = [t for t in alt if t not in eps_markers]
            # if alt had only eps, keep empty
            if not alt2 and alt:
                out_alts.append([])
            else:
                out_alts.append(alt2)
        cleaned[lhs] = out_alts

    return cleaned, order


def _extract_keywords_from_python(py_text: str) -> List[str]:
    # Heuristic: collect uppercase identifier-like strings appearing in string literals
    kws = set()
    try:
        tree = ast.parse(py_text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                s = node.value
                if 1 < len(s) <= 32 and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s):
                    if s.upper() == s and any(ch.isalpha() for ch in s):
                        kws.add(s)
    except Exception:
        pass

    # fallback regex extraction
    for m in re.finditer(r"['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", py_text):
        s = m.group(1)
        if s.upper() == s and 1 < len(s) <= 32 and any(ch.isalpha() for ch in s):
            kws.add(s)
    return sorted(kws)


def _pick_start_symbol(rules: Dict[str, List[List[str]]], order: List[str]) -> str:
    candidates = [
        "statement",
        "stmt",
        "sql",
        "program",
        "query",
        "statement_list",
        "statements",
        "input",
        "start",
    ]
    lower_map = {k.lower(): k for k in rules.keys()}
    for c in candidates:
        if c in rules:
            return c
        if c in lower_map:
            return lower_map[c]
    if order:
        return order[0]
    return next(iter(rules.keys()))


def _collect_terminals(rules: Dict[str, List[List[str]]]) -> Tuple[List[str], List[str]]:
    kws = set()
    punct = set()
    for lhs, alts in rules.items():
        for alt in alts:
            for t in alt:
                if not t:
                    continue
                if t.startswith("<") and t.endswith(">"):
                    continue
                if t in ("[", "]", "{", "}", "(", ")", "|", "*", "+", "?"):
                    continue
                # quoted literal -> terminal literal without quotes
                if (len(t) >= 2 and ((t[0] == t[-1] == "'") or (t[0] == t[-1] == '"') or (t[0] == t[-1] == "`"))):
                    lit = t[1:-1]
                    if lit and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lit) and lit.upper() == lit:
                        kws.add(lit)
                    else:
                        # treat punctuation-like in quotes as punct
                        if any(ch in ",;()[]{}.=<>!+-/*%" for ch in lit):
                            punct.add(lit)
                    continue

                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t):
                    if t.upper() == t and any(ch.isalpha() for ch in t):
                        kws.add(t)
                else:
                    if any(ch in ",;()[]{}.=<>!+-/*%." for ch in t):
                        punct.add(t)
    return sorted(kws), sorted(punct)


def _build_fuzzer_code(rules: Dict[str, List[List[str]]], start_symbol: str, keywords: List[str], punct: List[str]) -> str:
    # Reduce size: keep rules, but avoid very large literals; still fine.
    rules_repr = repr(rules)
    start_repr = repr(start_symbol)
    keywords_repr = repr(keywords)
    punct_repr = repr(punct)

    code = f"""
import re
import time
import random
import string

_RULES = {rules_repr}
_START = {start_repr}
_KEYWORDS = {keywords_repr}
_PUNCT = {punct_repr}

_state = {{
    "t0": None,
    "calls": 0,
    "rng": None,
    "corpus": [],
    "corpus_set": set(),
    "alt_meta": None,
    "start_alts": None,
}}

_re_tok = re.compile(r\"\"\"(--[^\\n]*\\n|/\\*.*?\\*/|'(?:''|[^'])*'|"(?:\"\"|[^"])*"|`[^`]*`|\\b\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?\\b|0x[0-9A-Fa-f]+|\\b[A-Za-z_][A-Za-z0-9_]*\\b|<=|>=|<>|!=|==|:=|\\|\\||&&|\\?\\d+|[(),;.*+\\-/%=<>\u005b\u005d]|\u005c\u005cs+|.)\"\"\", re.DOTALL)

def _sql_join(tokens):
    out = []
    prev = ""
    for t in tokens:
        if t is None or t == "":
            continue
        # collapse explicit whitespace tokens
        if t.isspace():
            if out and out[-1] != " ":
                out.append(" ")
            prev = " "
            continue

        # no spaces before these
        if t in (",", ";", ")", "]"):
            if out and out[-1] == " ":
                out.pop()
            out.append(t)
            out.append(" ")
            prev = t
            continue

        # no space after these
        if t in ("(", "["):
            if out and out[-1] == " ":
                out.pop()
            out.append(t)
            prev = t
            continue

        # dot joins tightly
        if t == ".":
            if out and out[-1] == " ":
                out.pop()
            out.append(".")
            prev = "."
            continue

        # default: ensure single space separation
        if out and out[-1] not in (" ", "(", "[", ".", "") and prev not in ("(", "[", ".", " "):
            out.append(" ")
        out.append(t)
        prev = t

    s = "".join(out).strip()
    # remove trailing space
    if s.endswith(" "):
        s = s[:-1]
    return s

def _rand_ident(rng):
    # Mix plain, quoted, bracketed identifiers
    base = rng.choice(["t", "tab", "tbl", "x", "y", "z", "col", "c", "v", "idx", "schema"])
    suffix = str(rng.randrange(0, 1000))
    ident = base + suffix
    r = rng.random()
    if r < 0.10:
        return f'"{ident}"'
    if r < 0.16:
        return f'`{ident}`'
    if r < 0.22:
        return f'[{ident}]'
    return ident

def _rand_string(rng):
    # include edge cases: empty, quotes, unicode, longish
    choices = [
        "",
        "a",
        "test",
        "O''Reilly",
        "line\\nfeed",
        "tab\\tsep",
        "%%",
        "_",
        "漢字",
        "emoji_🙂",
        "null\\u0000byte",
        "''''",
    ]
    s = rng.choice(choices)
    if rng.random() < 0.30:
        # random ascii payload
        n = rng.randrange(0, 24)
        s = "".join(rng.choice(string.ascii_letters + string.digits + " _-") for _ in range(n))
        if rng.random() < 0.15:
            s += "'"
        if rng.random() < 0.15:
            s = s.replace("'", "''")
    # single-quoted SQL string, escape by doubling
    s2 = s.replace("'", "''")
    return "'" + s2 + "'"

def _rand_number(rng):
    r = rng.random()
    if r < 0.25:
        return str(rng.randrange(-5, 6))
    if r < 0.45:
        return str(rng.randrange(-100000, 100000))
    if r < 0.65:
        # float/scientific
        a = rng.randrange(-1000, 1000) / (10 ** rng.randrange(0, 4))
        e = rng.randrange(-20, 20)
        return f"{a}e{e}"
    if r < 0.80:
        return "0x" + "".join(rng.choice("0123456789ABCDEF") for _ in range(rng.randrange(1, 9)))
    if r < 0.90:
        return "1." + "0" * rng.randrange(0, 12)
    return str(2 ** rng.randrange(0, 31))

def _rand_value(rng):
    r = rng.random()
    if r < 0.20:
        return "NULL"
    if r < 0.35:
        return rng.choice(["TRUE", "FALSE"])
    if r < 0.65:
        return _rand_number(rng)
    return _rand_string(rng)

def _maybe_ws(rng):
    # whitespace variants to hit tokenizer paths
    return rng.choice([" ", "  ", "\\t", "\\n", "\\r\\n", " \\t ", ""])

def _maybe_comment(rng):
    r = rng.random()
    if r < 0.08:
        return "/*" + rng.choice(["", "x", "comment", " multi\\nline ", "*/ /*"]) + "*/"
    if r < 0.16:
        return "--" + rng.choice(["", " c", " comment", "x"]) + "\\n"
    return ""

def _rand_kw(rng):
    if _KEYWORDS:
        return rng.choice(_KEYWORDS)
    return rng.choice(["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "WITH"])

def _is_nonterm(tok):
    return len(tok) >= 3 and tok[0] == "<" and tok[-1] == ">"

def _nonterm_name(tok):
    return tok[1:-1].strip()

def _find_matching(tokens, i, open_t, close_t):
    depth = 0
    j = i
    while j < len(tokens):
        t = tokens[j]
        if t == open_t:
            depth += 1
        elif t == close_t:
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return -1

def _expand_terminal(tok, rng):
    up = tok.upper()
    # Treat common token-type placeholders as generators
    if up in ("IDENT", "IDENTIFIER", "ID", "NAME", "TABLE", "TABLE_NAME", "COLUMN", "COLUMN_NAME", "COL", "SCHEMA", "DATABASE"):
        return _rand_ident(rng)
    if up in ("STRING", "STR", "TEXT", "CHAR", "VARCHAR", "NVARCHAR"):
        return _rand_string(rng)
    if up in ("NUMBER", "NUM", "INT", "INTEGER", "FLOAT", "REAL", "DECIMAL", "NUMERIC"):
        return _rand_number(rng)
    if up in ("VALUE", "LITERAL", "CONST", "CONSTANT"):
        return _rand_value(rng)

    # quoted literal in grammar
    if len(tok) >= 2 and ((tok[0] == tok[-1] == "'") or (tok[0] == tok[-1] == '"') or (tok[0] == tok[-1] == "`")):
        return tok[1:-1]

    # /regex/ in grammar -> generate identifier-ish
    if len(tok) >= 2 and tok[0] == "/" and tok[-1] == "/":
        return _rand_ident(rng)

    return tok

def _choose_alt(rule_name, alts, depth, rng, meta):
    # meta: list of (nonterm_count, len)
    if not alts:
        return []
    if depth <= 1 and meta:
        # prefer fewer nonterm occurrences when depth is low
        best = []
        best_score = None
        for idx, alt in enumerate(alts):
            ntc, ln = meta[idx]
            score = (ntc * 10 + ln)
            if best_score is None or score < best_score:
                best_score = score
                best = [alt]
            elif score == best_score:
                best.append(alt)
        return rng.choice(best)
    # slightly bias towards shorter alternatives to keep statements manageable
    if meta and rng.random() < 0.65:
        weights = []
        for (ntc, ln) in meta:
            w = 1.0 / (1.0 + 0.35 * ln + 1.1 * ntc)
            weights.append(w)
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for alt, w in zip(alts, weights):
            acc += w
            if acc >= r:
                return alt
        return alts[-1]
    return rng.choice(alts)

def _gen_from_rule(rule, depth, rng, stack):
    # Special-case common rule names
    rname = rule.lower()
    if "ident" in rname or rname.endswith("name") or "table" in rname or "column" in rname:
        if rng.random() < 0.75:
            return [_rand_ident(rng)]
    if "string" in rname or rname.endswith("str"):
        return [_rand_string(rng)]
    if "number" in rname or "integer" in rname or rname.endswith("int") or "float" in rname:
        return [_rand_number(rng)]
    if "literal" in rname or "value" in rname or "const" in rname:
        return [_rand_value(rng)]

    alts = _RULES.get(rule)
    if not alts:
        return [_rand_ident(rng)]

    meta = None
    if _state["alt_meta"] is not None:
        meta = _state["alt_meta"].get(rule)

    if depth <= 0:
        # stop recursion
        return [_rand_ident(rng)]

    # recursion guard
    if rule in stack and depth <= 2:
        # choose a terminal-ish alternative if possible
        if meta:
            best = None
            for idx, (ntc, ln) in enumerate(meta):
                if ntc == 0:
                    cand = alts[idx]
                    if best is None or len(cand) < len(best):
                        best = cand
            if best is not None:
                return _expand_tokens(best, depth - 1, rng, stack)
        return [_rand_ident(rng)]

    alt = _choose_alt(rule, alts, depth, rng, meta)
    return _expand_tokens(alt, depth - 1, rng, stack + (rule,))

def _expand_atom(tokens, i, depth, rng, stack):
    t = tokens[i]
    if t == "(":
        j = _find_matching(tokens, i, "(", ")")
        if j == -1:
            return ([], i + 1)
        inner = tokens[i + 1:j]
        expanded = _expand_tokens(inner, depth, rng, stack)
        return (expanded, j + 1)

    if t == "[":
        j = _find_matching(tokens, i, "[", "]")
        if j == -1:
            return ([], i + 1)
        inner = tokens[i + 1:j]
        if rng.random() < 0.55:
            expanded = _expand_tokens(inner, depth, rng, stack)
        else:
            expanded = []
        return (expanded, j + 1)

    if t == "{":
        j = _find_matching(tokens, i, "{", "}")
        if j == -1:
            return ([], i + 1)
        inner = tokens[i + 1:j]
        # repeat 0..2
        reps = 0
        r = rng.random()
        if r < 0.40:
            reps = 0
        elif r < 0.80:
            reps = 1
        else:
            reps = 2
        out = []
        for _ in range(reps):
            out.extend(_expand_tokens(inner, depth, rng, stack))
        return (out, j + 1)

    if _is_nonterm(t):
        nm = _nonterm_name(t)
        out = _gen_from_rule(nm, depth, rng, stack)
        return (out, i + 1)

    # terminal
    return ([_expand_terminal(t, rng)], i + 1)

def _expand_tokens(tokens, depth, rng, stack):
    out = []
    i = 0
    L = len(tokens)
    while i < L:
        t = tokens[i]
        if t == "|":
            # should not occur at this stage
            i += 1
            continue

        atom, j = _expand_atom(tokens, i, depth, rng, stack)
        i = j

        # postfix quantifiers
        if i < L and tokens[i] in ("?", "*", "+"):
            q = tokens[i]
            i += 1
            if q == "?":
                if rng.random() < 0.5:
                    atom = []
            elif q == "*":
                reps = rng.randrange(0, 3)
                atom = atom * reps
            else:  # +
                reps = rng.randrange(1, 4)
                atom = atom * reps

        out.extend(atom)
        if len(out) > 220:
            break
    return out

def _spice_sql(s, rng):
    # Optional leading BOM/whitespace/comments, trailing semicolon/garbage to hit tokenizer & error paths
    lead = ""
    if rng.random() < 0.02:
        lead += "\\ufeff"
    if rng.random() < 0.25:
        lead += _maybe_ws(rng)
    if rng.random() < 0.20:
        lead += _maybe_comment(rng)
    if rng.random() < 0.20:
        lead += _maybe_ws(rng)

    tail = ""
    if rng.random() < 0.60:
        tail += rng.choice(["", ";", " ;", "; ", ";;"])
    if rng.random() < 0.06:
        tail += rng.choice([" /*", " --", " )", " ,", " .", " ??", " INVALID", " '", " 0x", " /*unterminated"])
    if rng.random() < 0.25:
        tail += _maybe_ws(rng)

    # Case mangling
    if rng.random() < 0.20:
        def _case_word(m):
            w = m.group(0)
            if w.upper() in set(_KEYWORDS):
                return rng.choice([w.upper(), w.lower(), w.title()])
            return w
        s = re.sub(r"\\b[A-Za-z_][A-Za-z0-9_]*\\b", _case_word, s)

    return (lead + s + tail).strip()

def _templates(rng):
    t = _rand_ident(rng)
    c1 = _rand_ident(rng)
    c2 = _rand_ident(rng)
    v1 = _rand_value(rng)
    v2 = _rand_value(rng)
    n1 = _rand_number(rng)
    s1 = _rand_string(rng)
    typ = rng.choice(["INT", "INTEGER", "TEXT", "REAL", "BLOB", "NUMERIC", "VARCHAR(255)"])
    op = rng.choice(["=", "<", ">", "<=", ">=", "<>", "!=", "LIKE", "IN", "IS", "IS NOT"])
    agg = rng.choice(["COUNT", "SUM", "MIN", "MAX", "AVG"])
    fn = rng.choice(["COALESCE", "NULLIF", "ABS", "ROUND", "SUBSTR", "LENGTH"])
    join = rng.choice(["JOIN", "LEFT JOIN", "INNER JOIN", "CROSS JOIN"])
    order = rng.choice(["ASC", "DESC"])
    lim = str(rng.randrange(0, 50))
    off = str(rng.randrange(0, 50))
    tbl2 = _rand_ident(rng)

    return [
        f"SELECT {c1} FROM {t}",
        f"SELECT * FROM {t}",
        f"SELECT DISTINCT {c1}, {c2} FROM {t} WHERE {c1} {op} {v1}",
        f"SELECT {agg}({c1}) FROM {t} GROUP BY {c2} HAVING {agg}({c1}) > {n1}",
        f"SELECT {fn}({c1}, {v1}) FROM {t} ORDER BY {c2} {order} LIMIT {lim} OFFSET {off}",
        f"WITH cte AS (SELECT {c1} FROM {t} WHERE {c2} {op} {v2}) SELECT * FROM cte",
        f"INSERT INTO {t}({c1},{c2}) VALUES ({v1},{v2})",
        f"INSERT INTO {t} VALUES ({v1}, {v2}, {n1})",
        f"UPDATE {t} SET {c1} = {v1}, {c2} = {v2} WHERE {c1} {op} {v2}",
        f"DELETE FROM {t} WHERE {c1} {op} {v1}",
        f"CREATE TABLE {t} ({c1} {typ}, {c2} {typ} NOT NULL, PRIMARY KEY ({c1}))",
        f"CREATE INDEX idx_{t} ON {t}({c1},{c2})",
        f"DROP TABLE {t}",
        f"ALTER TABLE {t} ADD COLUMN {c2} {typ}",
        f"BEGIN",
        f"COMMIT",
        f"ROLLBACK",
        f"EXPLAIN SELECT {c1} FROM {t}",
        f"SELECT {t}.{c1} FROM {t} {join} {tbl2} ON {t}.{c1} = {tbl2}.{c1}",
        f"SELECT CASE WHEN {c1} {op} {v1} THEN {v2} ELSE {s1} END FROM {t}",
        f"SELECT {c1} FROM {t} WHERE {c1} BETWEEN {n1} AND {n1}",
        f"SELECT {c1} FROM {t} WHERE {c1} IN ({v1},{v2},{n1})",
    ]

def _mutate(s, rng):
    toks = [m.group(0) for m in _re_tok.finditer(s)]
    # drop whitespace tokens for structure mutations, keep later with join
    core = [t for t in toks if not t.isspace()]
    if not core:
        core = ["SELECT", "*", "FROM", _rand_ident(rng)]

    ops = rng.randrange(2, 7)
    for _ in range(ops):
        if not core:
            break
        r = rng.random()
        if r < 0.18:
            # delete
            del core[rng.randrange(0, len(core))]
        elif r < 0.38:
            # insert keyword / punct / value
            ins = rng.choice([
                _rand_kw(rng),
                _rand_ident(rng),
                _rand_value(rng),
                rng.choice(["(", ")", ",", ";", ".", "=", "<>", "||", "+", "-", "*", "/"]),
                _maybe_comment(rng),
            ])
            core.insert(rng.randrange(0, len(core) + 1), ins)
        elif r < 0.58:
            # replace token
            i = rng.randrange(0, len(core))
            tok = core[i]
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
                core[i] = rng.choice([_rand_kw(rng), _rand_ident(rng), tok.upper(), tok.lower()])
            elif tok.startswith("'") or tok.startswith('"') or tok.startswith("`"):
                core[i] = _rand_string(rng)
            elif re.fullmatch(r"\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?", tok) or tok.lower().startswith("0x"):
                core[i] = _rand_number(rng)
            else:
                core[i] = rng.choice([",", ")", "(", ";", ".", "=", "<", ">", "<=", ">=", "<>", "!=", "||"])
        elif r < 0.72:
            # duplicate
            i = rng.randrange(0, len(core))
            core.insert(i, core[i])
        elif r < 0.86:
            # wrap a random span in parentheses
            if len(core) >= 2:
                a = rng.randrange(0, len(core))
                b = rng.randrange(a, min(len(core), a + rng.randrange(1, 8)))
                core.insert(a, "(")
                core.insert(b + 2, ")")
        else:
            # shuffle small window
            if len(core) >= 3:
                a = rng.randrange(0, len(core) - 2)
                b = min(len(core), a + rng.randrange(2, 7))
                window = core[a:b]
                rng.shuffle(window)
                core[a:b] = window

    # Add random whitespace/comment sprinkling
    out = []
    for t in core:
        if t and rng.random() < 0.08:
            out.append(_maybe_comment(rng))
        out.append(t)
        if rng.random() < 0.25:
            out.append(_maybe_ws(rng))
    s2 = _sql_join(out)
    if len(s2) > 600:
        s2 = s2[:600]
    return _spice_sql(s2, rng)

def _gen_stmt(rng):
    # Choose from grammar, templates, mutation, random garbage
    r = rng.random()
    if r < 0.60 and _START in _RULES:
        # sometimes generate multi-statement
        if rng.random() < 0.10:
            parts = []
            k = rng.randrange(2, 5)
            for _ in range(k):
                toks = _gen_from_rule(_START, depth=rng.randrange(4, 10), rng=rng, stack=())
                s = _sql_join(toks)
                s = _spice_sql(s, rng)
                if s:
                    parts.append(s.rstrip(";"))
            if parts:
                return _spice_sql("; ".join(parts) + ";", rng)
        toks = _gen_from_rule(_START, depth=rng.randrange(5, 11), rng=rng, stack=())
        s = _sql_join(toks)
        if not s or len(s) < 2:
            s = rng.choice(_templates(rng))
        return _spice_sql(s, rng)

    if r < 0.85:
        return _spice_sql(rng.choice(_templates(rng)), rng)

    if r < 0.97 and _state["corpus"]:
        return _mutate(rng.choice(_state["corpus"]), rng)

    # random garbage to hit tokenizer errors
    garbage = []
    for _ in range(rng.randrange(6, 40)):
        kind = rng.random()
        if kind < 0.35:
            garbage.append(_rand_kw(rng))
        elif kind < 0.60:
            garbage.append(_rand_ident(rng))
        elif kind < 0.78:
            garbage.append(_rand_value(rng))
        else:
            garbage.append(rng.choice(["(", ")", ",", ";", ".", "=", "<>", "!=", "||", "+", "-", "*", "/", "%", "[", "]"]))
        if rng.random() < 0.30:
            garbage.append(_maybe_ws(rng))
        if rng.random() < 0.08:
            garbage.append(_maybe_comment(rng))
    s = _sql_join(garbage)
    return _spice_sql(s, rng)

def _init_once():
    if _state["t0"] is not None:
        return
    _state["t0"] = time.time()
    seed = (time.time_ns() ^ (id(_state) << 1)) & 0xFFFFFFFFFFFFFFFF
    _state["rng"] = random.Random(seed)

    # alt meta for heuristics
    alt_meta = {{}}
    for rule, alts in _RULES.items():
        metas = []
        for alt in alts:
            ntc = 0
            for t in alt:
                if t and (t.startswith("<") and t.endswith(">")):
                    ntc += 1
            metas.append((ntc, len(alt)))
        alt_meta[rule] = metas
    _state["alt_meta"] = alt_meta

    # seed corpus
    rng = _state["rng"]
    seeds = []
    for _ in range(120):
        seeds.append(_spice_sql(rng.choice(_templates(rng)), rng))
    if _START in _RULES:
        for _ in range(220):
            toks = _gen_from_rule(_START, depth=rng.randrange(5, 11), rng=rng, stack=())
            s = _sql_join(toks)
            s = _spice_sql(s, rng)
            if s:
                seeds.append(s)

    for s in seeds:
        if not s:
            continue
        if len(_state["corpus"]) >= 2500:
            break
        if s not in _state["corpus_set"]:
            _state["corpus"].append(s)
            _state["corpus_set"].add(s)

def fuzz(parse_sql):
    _init_once()
    rng = _state["rng"]
    _state["calls"] += 1

    elapsed = time.time() - _state["t0"]
    # Keep parse_sql calls low while still exploring
    if elapsed > 58.8:
        return False
    if _state["calls"] > 140 and elapsed > 25:
        return False

    # Batch sizing: larger early, smaller later
    if elapsed < 8:
        batch_size = 1800
    elif elapsed < 20:
        batch_size = 1200
    elif elapsed < 40:
        batch_size = 850
    else:
        batch_size = 650

    stmts = []
    seen_local = set()
    # Ensure some targeted statement types every call
    forced = [
        "SELECT 1",
        "SELECT 'x''y'",
        "SELECT NULL",
        "SELECT 1+2*3",
        "SELECT * FROM " + _rand_ident(rng),
        "INSERT INTO " + _rand_ident(rng) + " VALUES (" + _rand_value(rng) + ")",
        "UPDATE " + _rand_ident(rng) + " SET " + _rand_ident(rng) + " = " + _rand_value(rng),
        "DELETE FROM " + _rand_ident(rng),
    ]
    for s in forced:
        s2 = _spice_sql(s, rng)
        if s2 and s2 not in seen_local:
            seen_local.add(s2)
            stmts.append(s2)

    while len(stmts) < batch_size:
        s = _gen_stmt(rng)
        if not s:
            continue
        # keep moderate statement lengths
        if len(s) > 900:
            s = s[:900]
        if s in seen_local:
            continue
        seen_local.add(s)
        stmts.append(s)

    parse_sql(stmts)

    # Update corpus with a small random subset
    for _ in range(40):
        s = rng.choice(stmts)
        if s not in _state["corpus_set"]:
            _state["corpus"].append(s)
            _state["corpus_set"].add(s)
            if len(_state["corpus"]) > 3000:
                # trim
                drop = _state["corpus"][:400]
                _state["corpus"] = _state["corpus"][400:]
                for d in drop:
                    _state["corpus_set"].discard(d)

    return True
"""
    return code.strip() + "\n"


class Solution:
    def solve(self, resources_path: str) -> dict:
        grammar_path = os.path.join(resources_path, "sql_grammar.txt")
        with open(grammar_path, "r", encoding="utf-8", errors="ignore") as f:
            grammar_text = f.read()

        rules, order = _parse_grammar(grammar_text)
        start_symbol = _pick_start_symbol(rules, order)
        kws_g, punct = _collect_terminals(rules)

        kws_py = []
        for p in (
            os.path.join(resources_path, "sql_engine", "tokenizer.py"),
            os.path.join(resources_path, "sql_engine", "parser.py"),
        ):
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    kws_py.extend(_extract_keywords_from_python(f.read()))
            except Exception:
                pass

        # Merge keywords, add a broad SQL set for robustness
        broad = [
            "SELECT", "INSERT", "UPDATE", "DELETE",
            "CREATE", "DROP", "ALTER",
            "TABLE", "INDEX", "VIEW", "TRIGGER",
            "INTO", "VALUES", "SET", "FROM", "WHERE",
            "GROUP", "BY", "HAVING", "ORDER", "LIMIT", "OFFSET",
            "DISTINCT", "AS", "AND", "OR", "NOT", "NULL",
            "TRUE", "FALSE",
            "JOIN", "LEFT", "RIGHT", "FULL", "INNER", "OUTER", "CROSS", "ON",
            "UNION", "ALL", "INTERSECT", "EXCEPT",
            "WITH", "RECURSIVE",
            "CASE", "WHEN", "THEN", "ELSE", "END",
            "EXISTS", "IN", "IS", "LIKE", "BETWEEN",
            "PRIMARY", "KEY", "FOREIGN", "REFERENCES", "UNIQUE",
            "CHECK", "DEFAULT", "CONSTRAINT",
            "BEGIN", "COMMIT", "ROLLBACK",
            "EXPLAIN", "ANALYZE", "PRAGMA",
            "ASC", "DESC",
        ]
        kwset = set(kws_g)
        kwset.update(kws_py)
        kwset.update(broad)
        # Remove overly long or non-identifier-ish
        keywords = sorted([k for k in kwset if 1 < len(k) <= 32 and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)])

        code = _build_fuzzer_code(rules, start_symbol, keywords, punct)
        return {"code": code}