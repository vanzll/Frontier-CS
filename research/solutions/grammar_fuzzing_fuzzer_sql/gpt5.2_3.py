import os
import re
import json
from typing import Dict


def _read_text(path: str, limit: int = 2_000_000) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read(limit)
        try:
            return data.decode("utf-8")
        except Exception:
            return data.decode("latin-1", errors="replace")
    except Exception:
        return ""


def _extract_keywords(*texts: str):
    kws = set()
    for t in texts:
        for m in re.finditer(r"(?<![A-Za-z0-9_])([A-Z][A-Z0-9_]{1,24})(?![A-Za-z0-9_])", t):
            s = m.group(1)
            if s in ("TRUE", "FALSE", "NULL"):
                kws.add(s)
            elif len(s) <= 24 and not s.endswith("_"):
                kws.add(s)
        for m in re.finditer(r"""['"]([A-Z][A-Z0-9_]{1,24})['"]""", t):
            kws.add(m.group(1))
    base = {
        "SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "UPDATE", "SET", "DELETE",
        "CREATE", "TABLE", "DROP", "ALTER", "INDEX", "VIEW", "TRIGGER", "PRIMARY", "KEY",
        "FOREIGN", "REFERENCES", "NOT", "NULL", "UNIQUE", "CHECK", "DEFAULT",
        "AND", "OR", "IN", "IS", "LIKE", "BETWEEN", "EXISTS",
        "JOIN", "LEFT", "RIGHT", "FULL", "OUTER", "INNER", "CROSS", "ON", "USING",
        "GROUP", "BY", "HAVING", "ORDER", "LIMIT", "OFFSET", "DISTINCT",
        "UNION", "ALL", "AS", "CASE", "WHEN", "THEN", "ELSE", "END",
        "CAST", "ASC", "DESC", "WITH",
        "BEGIN", "COMMIT", "ROLLBACK", "TRANSACTION",
    }
    kws |= base
    out = sorted(kws)
    return out


class Solution:
    def solve(self, resources_path: str) -> dict:
        grammar_path = os.path.join(resources_path, "sql_grammar.txt")
        parser_path = os.path.join(resources_path, "sql_engine", "parser.py")
        tokenizer_path = os.path.join(resources_path, "sql_engine", "tokenizer.py")
        ast_path = os.path.join(resources_path, "sql_engine", "ast_nodes.py")

        grammar_text = _read_text(grammar_path, limit=4_000_000)
        parser_text = _read_text(parser_path)
        tokenizer_text = _read_text(tokenizer_path)
        ast_text = _read_text(ast_path)

        keywords = _extract_keywords(grammar_text, parser_text, tokenizer_text, ast_text)

        grammar_lit = json.dumps(grammar_text)
        keywords_lit = json.dumps(keywords)

        fuzzer_code = f"""
import random
import time
import re
from collections import defaultdict, deque

GRAMMAR_TEXT = {grammar_lit}
KEYWORDS = {keywords_lit}

_WS_CHOICES = (" ", "\\t", "\\n", "\\r\\n", "  ", "\\n\\n")
_PUNCT_NO_PRE_SPACE = set([",", ")", "]", "}}", ";", ".", ":", "::"])
_PUNCT_NO_POST_SPACE = set(["(", "[", "{{", "."])
_SIMPLE_PUNCT = set(list("(),;.*+-/%"))

_re_rule = re.compile(r"^\\s*([^#;]+?)\\s*(?:::=|::=|:=|=)\\s*(.+?)\\s*$")
_re_comment = re.compile(r"(?://.*$)|(?:#.*$)")
_re_ws = re.compile(r"\\s+")
_re_fix1 = re.compile(r"\\s+([,;\\)\\]])")
_re_fix2 = re.compile(r"([\\(\\[])\\s+")
_re_fix3 = re.compile(r"\\s+([\\.])\\s+")
_re_fix4 = re.compile(r"\\s{3,}")
_re_bare = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

class _G:
    __slots__ = ("rules", "order", "cost", "start", "_rnd")
    def __init__(self, text: str, seed: int = 1):
        self.rules = {{}}
        self.order = []
        self._rnd = random.Random(seed)
        self._parse(text)
        self._resolve_syms()
        self.cost = self._compute_costs()
        self.start = self._detect_start()

    def _tokenize(self, s: str):
        out = []
        i = 0
        n = len(s)
        while i < n:
            c = s[i]
            if c.isspace():
                i += 1
                continue
            if c in "[]{{}}()|*+?":
                out.append(c)
                i += 1
                continue
            if c in ("'", '"'):
                q = c
                i += 1
                buf = []
                while i < n:
                    ch = s[i]
                    if ch == q:
                        if i + 1 < n and s[i + 1] == q:
                            buf.append(q)
                            i += 2
                            continue
                        i += 1
                        break
                    if ch == "\\\\":  # preserve escapes
                        if i + 1 < n:
                            buf.append(s[i])
                            buf.append(s[i+1])
                            i += 2
                            continue
                    buf.append(ch)
                    i += 1
                out.append(("TQ", "".join(buf)))
                continue
            if c == "<":
                j = s.find(">", i + 1)
                if j != -1:
                    out.append(("NT", s[i+1:j].strip()))
                    i = j + 1
                    continue
            if c in _SIMPLE_PUNCT:
                # coalesce multi-char ops
                j = i + 1
                while j < n and s[j] in "=<>|&!:+-*/%.":
                    j += 1
                out.append(("TP", s[i:j]))
                i = j
                continue
            # bare word / token
            j = i + 1
            while j < n and not s[j].isspace() and s[j] not in "[]{{}}()|*+?":
                if s[j] in ("'", '"', "<"):
                    break
                j += 1
            tok = s[i:j]
            out.append(("SYM", tok))
            i = j
        return out

    def _parse(self, text: str):
        for raw in text.splitlines():
            line = _re_comment.sub("", raw).strip()
            if not line:
                continue
            m = _re_rule.match(line)
            if not m:
                continue
            lhs = m.group(1).strip()
            rhs = m.group(2).strip()
            lhs = lhs.strip()
            if lhs.startswith("<") and lhs.endswith(">"):
                lhs = lhs[1:-1].strip()
            self.order.append(lhs)
            toks = self._tokenize(rhs)
            expr, idx = self._parse_expr(toks, 0, stop=set())
            if expr is None:
                continue
            self.rules[lhs] = expr

    def _parse_expr(self, toks, i, stop):
        alts = []
        seq = []
        while i < len(toks):
            t = toks[i]
            if isinstance(t, str) and t in stop:
                break
            if t == "|":
                alts.append(("SEQ", seq) if len(seq) != 1 else seq[0])
                seq = []
                i += 1
                continue
            item, i = self._parse_item(toks, i)
            if item is None:
                i += 1
                continue
            seq.append(item)
        if seq:
            alts.append(("SEQ", seq) if len(seq) != 1 else seq[0])
        if not alts:
            return None, i
        if len(alts) == 1:
            return alts[0], i
        return ("ALT", alts), i

    def _parse_item(self, toks, i):
        if i >= len(toks):
            return None, i
        t = toks[i]
        node = None
        if t == "(":
            node, i2 = self._parse_expr(toks, i + 1, stop=set([")"]))
            i = i2 + 1 if i2 < len(toks) and toks[i2] == ")" else i2
            if node is None:
                node = ("T", "(")
        elif t == "[":
            inner, i2 = self._parse_expr(toks, i + 1, stop=set(["]"]))
            i = i2 + 1 if i2 < len(toks) and toks[i2] == "]" else i2
            if inner is None:
                inner = ("T", "")
            node = ("OPT", inner)
        elif t == "{{":
            inner, i2 = self._parse_expr(toks, i + 1, stop=set(["}}"]))
            i = i2 + 1 if i2 < len(toks) and toks[i2] == "}}" else i2
            if inner is None:
                inner = ("T", "")
            node = ("REP", inner, 0, 3)
        elif t == "{{":
            # unreachable (kept)
            i += 1
            node = ("T", "{{")
        elif t == "{{":
            i += 1
            node = ("T", "{{")
        elif t == "{":
            inner, i2 = self._parse_expr(toks, i + 1, stop=set(["}"]))
            i = i2 + 1 if i2 < len(toks) and toks[i2] == "}" else i2
            if inner is None:
                inner = ("T", "")
            node = ("REP", inner, 0, 3)
        else:
            if isinstance(t, tuple):
                kind, val = t
                if kind == "NT":
                    node = ("NT", val)
                elif kind == "TQ":
                    node = ("T", val)
                elif kind == "TP":
                    node = ("T", val)
                elif kind == "SYM":
                    node = ("SYM", val)
                else:
                    node = ("T", str(val))
            else:
                node = ("T", str(t))
            i += 1
        # quantifier
        if i < len(toks):
            q = toks[i]
            if q in ("?", "*", "+"):
                i += 1
                if q == "?":
                    node = ("OPT", node)
                elif q == "*":
                    node = ("REP", node, 0, 3)
                else:
                    node = ("REP", node, 1, 3)
        return node, i

    def _resolve_syms(self):
        # Convert ("SYM", x) to NT if x is a rule, else terminal
        if not self.rules:
            return
        names = set(self.rules.keys())
        def conv(node):
            if node is None:
                return None
            t = node[0]
            if t == "SYM":
                x = node[1]
                if x.startswith("<") and x.endswith(">"):
                    x = x[1:-1].strip()
                    return ("NT", x) if x in names else ("T", x)
                return ("NT", x) if x in names else ("T", x)
            if t == "SEQ":
                return ("SEQ", [conv(x) for x in node[1] if x is not None])
            if t == "ALT":
                return ("ALT", [conv(x) for x in node[1] if x is not None])
            if t == "OPT":
                return ("OPT", conv(node[1]))
            if t == "REP":
                return ("REP", conv(node[1]), node[2], node[3])
            return node
        for k, v in list(self.rules.items()):
            self.rules[k] = conv(v)

    def _detect_start(self):
        if not self.rules:
            return None
        cand = [c for c in ("sql", "statement", "stmt", "program", "query", "input") if c in self.rules]
        if cand:
            return cand[0]
        # try case-insensitive match
        low = {{k.lower(): k for k in self.rules.keys()}}
        for c in ("sql", "statement", "stmt", "program", "query", "input"):
            if c in low:
                return low[c]
        return self.order[0] if self.order else next(iter(self.rules.keys()))

    def _compute_costs(self):
        cost = {{k: 10**9 for k in self.rules}}
        def node_cost(node):
            if node is None:
                return 0
            t = node[0]
            if t == "T":
                return 1 if node[1] else 0
            if t == "NT":
                return cost.get(node[1], 5)
            if t == "SEQ":
                s = 0
                for ch in node[1]:
                    s += node_cost(ch)
                    if s > 10**8:
                        return 10**9
                return s
            if t == "ALT":
                return min((node_cost(ch) for ch in node[1]), default=10**9)
            if t == "OPT":
                return 0
            if t == "REP":
                inner = node_cost(node[1])
                return node[2] * inner
            return 5
        # fixed point
        for _ in range(60):
            changed = False
            for k, v in self.rules.items():
                nc = node_cost(v)
                if nc < cost[k]:
                    cost[k] = nc
                    changed = True
            if not changed:
                break
        return cost

    def gen(self, sym: str = None, max_depth: int = 14, max_tokens: int = 120):
        if not self.rules:
            return ["SELECT", "1"]
        if sym is None:
            sym = self.start
        stack = set()
        out = []
        def emit(tok):
            if tok is None:
                return
            if tok == "":
                return
            out.append(tok)

        def gen_node(node, depth, remain):
            if remain <= 0:
                return 0
            if node is None:
                return 0
            t = node[0]
            if t == "T":
                s = node[1]
                if s:
                    emit(s)
                    return 1
                return 0
            if t == "NT":
                name = node[1]
                # custom nonterminals
                gen_special = _special_nt(name, self._rnd)
                if gen_special is not None:
                    for tok in gen_special:
                        if remain <= 0:
                            break
                        emit(tok)
                        remain -= 1
                    return 0
                if name not in self.rules:
                    emit(name)
                    return 1
                if depth <= 0:
                    # choose cheapest alternative
                    return gen_node(_pick_cheapest(self.rules[name], self.cost, self._rnd), depth - 1, remain)
                if name in stack:
                    # recursion break
                    return gen_node(_pick_cheapest(self.rules[name], self.cost, self._rnd), depth - 1, remain)
                stack.add(name)
                used = gen_node(self.rules[name], depth - 1, remain)
                stack.remove(name)
                return used
            if t == "SEQ":
                for ch in node[1]:
                    if remain <= 0:
                        break
                    used = gen_node(ch, depth, remain)
                    remain -= used
                return 0
            if t == "ALT":
                opts = node[1]
                if not opts:
                    return 0
                if depth <= 2:
                    pick = _pick_cheapest(node, self.cost, self._rnd)
                else:
                    pick = _pick_weighted_alt(opts, self.cost, depth, self._rnd)
                return gen_node(pick, depth, remain)
            if t == "OPT":
                p = 0.65 if depth > 6 else 0.35
                if self._rnd.random() < p:
                    return gen_node(node[1], depth - 1, remain)
                return 0
            if t == "REP":
                inner, mn, mx = node[1], node[2], node[3]
                if depth <= 3:
                    n = mn
                else:
                    # bias small but non-zero
                    if mn == 0:
                        n = 0 if self._rnd.random() < 0.45 else 1
                        if self._rnd.random() < 0.25:
                            n += 1
                    else:
                        n = mn
                        if self._rnd.random() < 0.35:
                            n = min(mx, mn + 1)
                        if self._rnd.random() < 0.15:
                            n = min(mx, n + 1)
                for _ in range(n):
                    if remain <= 0:
                        break
                    used = gen_node(inner, depth - 1, remain)
                    remain -= used
                return 0
            return 0

        gen_node(("NT", sym), max_depth, max_tokens)
        if not out:
            return ["SELECT", "1"]
        return out

def _pick_cheapest(node, cost, rnd):
    if node is None:
        return ("T", "")
    t = node[0]
    if t == "ALT":
        best = None
        bestc = 10**9
        for opt in node[1]:
            c = _node_cost_est(opt, cost)
            if c < bestc:
                bestc = c
                best = opt
        return best if best is not None else (node[1][0] if node[1] else ("T",""))
    if t == "SEQ":
        return node
    return node

def _node_cost_est(node, cost):
    if node is None:
        return 0
    t = node[0]
    if t == "T":
        return 1 if node[1] else 0
    if t == "NT":
        return cost.get(node[1], 5)
    if t == "SEQ":
        s = 0
        for ch in node[1]:
            s += _node_cost_est(ch, cost)
        return s
    if t == "ALT":
        return min((_node_cost_est(ch, cost) for ch in node[1]), default=10**9)
    if t == "OPT":
        return 0
    if t == "REP":
        return node[2] * _node_cost_est(node[1], cost)
    return 5

def _pick_weighted_alt(opts, cost, depth, rnd):
    # prefer diverse but not too large
    scored = []
    for o in opts:
        c = _node_cost_est(o, cost)
        # lower c preferred; at high depth stronger preference for lower
        w = 1.0 / (1.0 + c)
        if depth < 6:
            w = w ** 0.65
        else:
            w = w ** 1.25
        scored.append((w, o))
    total = sum(w for w, _ in scored)
    r = rnd.random() * total if total > 0 else 0.0
    s = 0.0
    for w, o in scored:
        s += w
        if s >= r:
            return o
    return scored[-1][1] if scored else opts[0]

def _rand_ident(rnd):
    base = ["t", "u", "v", "w", "x", "y", "z", "users", "orders", "items", "prod", "log", "a", "b", "c"]
    s = rnd.choice(base) + str(rnd.randrange(0, 50))
    if rnd.random() < 0.15:
        return "`%s`" % s
    if rnd.random() < 0.12:
        return '"%s"' % s
    if rnd.random() < 0.08:
        return "[%s]" % s
    return s

def _rand_string(rnd):
    parts = ["", "a", "b", "test", "O''Reilly", "x\\\\y", "line1\\nline2", "Ω", "漢字", "NULL", "''", "/* */", "--"]
    s = rnd.choice(parts)
    if rnd.random() < 0.6:
        s += rnd.choice(parts)
    s = s.replace("\\\\", "\\\\\\\\")
    s = s.replace("'", "''")
    return "'" + s + "'"

def _rand_number(rnd):
    kind = rnd.randrange(0, 8)
    if kind == 0:
        return str(rnd.randrange(-3, 4))
    if kind == 1:
        return str(rnd.randrange(-1000, 1000))
    if kind == 2:
        return str(rnd.random() * (10 ** rnd.randrange(0, 4)))
    if kind == 3:
        return str(rnd.randrange(0, 100)) + "." + str(rnd.randrange(0, 1000))
    if kind == 4:
        return str(rnd.randrange(1, 20)) + "e" + str(rnd.randrange(-10, 10))
    if kind == 5:
        return "0x" + format(rnd.randrange(0, 1 << 16), "x")
    if kind == 6:
        return "00" + str(rnd.randrange(0, 9))
    return "-0"

def _special_nt(name, rnd):
    ln = name.lower()
    if any(x in ln for x in ("ident", "identifier", "name", "table", "column", "col", "schema", "alias")):
        return [_rand_ident(rnd)]
    if any(x in ln for x in ("string", "str", "text", "char", "varchar", "quoted")):
        return [_rand_string(rnd)]
    if any(x in ln for x in ("number", "numeric", "int", "integer", "float", "real", "double", "decimal")):
        return [_rand_number(rnd)]
    if ln in ("true", "false", "null"):
        return [ln.upper()]
    if "ws" in ln or "space" in ln:
        return [rnd.choice(_WS_CHOICES)]
    return None

def _join_tokens(toks, rnd):
    if not toks:
        return ""
    out = []
    prev = ""
    for tok in toks:
        if tok is None:
            continue
        tok = str(tok)
        if tok == "":
            continue
        if not out:
            out.append(tok)
            prev = tok
            continue
        no_pre = tok in _PUNCT_NO_PRE_SPACE or (len(tok) == 1 and tok in ",);]")
        no_post_prev = prev in _PUNCT_NO_POST_SPACE or (len(prev) == 1 and prev in "([.")
        if no_pre or no_post_prev:
            out.append(tok)
        else:
            out.append(" " + tok)
        prev = tok
    s = "".join(out)
    # fix common spacing artifacts
    s = _re_fix1.sub(r"\\1", s)
    s = _re_fix2.sub(r"\\1", s)
    s = _re_fix3.sub(r"\\1", s)
    s = _re_fix4.sub("  ", s)
    return s

def _randomize_case(sql, rnd):
    # randomize keyword casing without heavy parsing
    def f(m):
        w = m.group(0)
        if len(w) <= 1:
            return w
        mode = rnd.randrange(0, 4)
        if mode == 0:
            return w.upper()
        if mode == 1:
            return w.lower()
        if mode == 2:
            return w.capitalize()
        # mixed
        return "".join((ch.upper() if rnd.random() < 0.5 else ch.lower()) for ch in w)
    return re.sub(r"\\b[A-Za-z_][A-Za-z0-9_]*\\b", f, sql)

def _inject_noise(sql, rnd):
    if not sql:
        return sql
    if rnd.random() < 0.25:
        sql = rnd.choice(["", " ", "\\n", "\\t", "\\ufeff"]) + sql
    if rnd.random() < 0.35:
        sql = sql + rnd.choice(["", " ", "\\n", ";", ";;", "\\n;\\n"])
    # insert comments and weird whitespace
    if rnd.random() < 0.50:
        pos = rnd.randrange(0, len(sql) + 1)
        ins = rnd.choice(["/*x*/", "/**/", "/*nested*/*/", "--c\\n", "--\\n", "/*\\n*/"])
        sql = sql[:pos] + ins + sql[pos:]
    if rnd.random() < 0.35:
        sql = _randomize_case(sql, rnd)
    return sql

def _mutate(sql, rnd, fragments):
    if not sql:
        return sql
    s = sql
    L = len(s)
    r = rnd.random()
    if r < 0.20 and L > 0:
        # delete slice
        a = rnd.randrange(0, L)
        b = min(L, a + rnd.randrange(1, 1 + min(12, L - a)))
        s = s[:a] + s[b:]
    elif r < 0.40:
        # insert fragment
        frag = rnd.choice(fragments)
        a = rnd.randrange(0, L + 1)
        s = s[:a] + frag + s[a:]
    elif r < 0.58 and L > 0:
        # flip char
        a = rnd.randrange(0, L)
        ch = rnd.choice(["'", '"', ")", "(", ",", ";", " ", "\\t", "\\n", "*", ".", "=", "<", ">", "!", "|", "&"])
        s = s[:a] + ch + s[a+1:]
    elif r < 0.72 and L > 0:
        # duplicate region
        a = rnd.randrange(0, L)
        b = min(L, a + rnd.randrange(1, 1 + min(20, L - a)))
        seg = s[a:b]
        c = rnd.randrange(0, len(s) + 1)
        s = s[:c] + seg + s[c:]
    else:
        # truncate or extend
        if rnd.random() < 0.5:
            s = s[: rnd.randrange(0, L + 1)]
        else:
            s = s + rnd.choice(fragments)
    if len(s) > 4000:
        s = s[:4000]
    return s

def _make_templates(rnd):
    ids = lambda: _rand_ident(rnd)
    num = lambda: _rand_number(rnd)
    st = lambda: _rand_string(rnd)
    tmpls = []
    tmpls.append(lambda: f"SELECT {num()} AS {ids()}")
    tmpls.append(lambda: f"SELECT * FROM {ids()}")
    tmpls.append(lambda: f"SELECT {ids()}.{ids()}, {ids()} FROM {ids()} AS {ids()} WHERE {ids()} = {num()}")
    tmpls.append(lambda: f"SELECT DISTINCT {ids()} FROM {ids()} ORDER BY {ids()} DESC LIMIT {rnd.randrange(0,10)}")
    tmpls.append(lambda: f"INSERT INTO {ids()}({ids()},{ids()}) VALUES ({num()},{st()})")
    tmpls.append(lambda: f"UPDATE {ids()} SET {ids()}={num()}, {ids()}={st()} WHERE {ids()} IN ({num()},{num()},{num()})")
    tmpls.append(lambda: f"DELETE FROM {ids()} WHERE {ids()} BETWEEN {num()} AND {num()}")
    tmpls.append(lambda: f"CREATE TABLE {ids()} ({ids()} INT PRIMARY KEY, {ids()} TEXT, {ids()} REAL)")
    tmpls.append(lambda: f"CREATE INDEX {ids()} ON {ids()}({ids()},{ids()})")
    tmpls.append(lambda: f"DROP TABLE {ids()}")
    tmpls.append(lambda: f"SELECT CASE WHEN {ids()} > {num()} THEN {st()} ELSE {st()} END FROM {ids()}")
    tmpls.append(lambda: f"SELECT {ids()} FROM {ids()} WHERE {ids()} LIKE {st()}")
    tmpls.append(lambda: f"SELECT {ids()} FROM {ids()} WHERE {ids()} IS NULL OR {ids()} IS NOT NULL")
    tmpls.append(lambda: f"BEGIN TRANSACTION")
    tmpls.append(lambda: f"COMMIT")
    tmpls.append(lambda: f"ROLLBACK")
    tmpls.append(lambda: f"SELECT {num()} UNION ALL SELECT {num()} UNION SELECT {num()}")
    tmpls.append(lambda: f"SELECT {ids()} FROM {ids()} GROUP BY {ids()} HAVING COUNT(*) > {rnd.randrange(0,5)}")
    tmpls.append(lambda: f"SELECT {ids()} FROM {ids()} WHERE EXISTS (SELECT 1 FROM {ids()} WHERE {ids()}={num()})")
    tmpls.append(lambda: f"SELECT {ids()} FROM {ids()} JOIN {ids()} ON {ids()}.{ids()} = {ids()}.{ids()}")
    tmpls.append(lambda: f"SELECT {ids()} FROM {ids()} LEFT JOIN {ids()} ON {ids()}={ids()} WHERE {ids()} AND NOT {ids()}")
    tmpls.append(lambda: f"SELECT CAST({num()} AS INT), CAST({st()} AS TEXT)")
    return tmpls

class _State:
    __slots__ = ("rnd", "g", "start_time", "calls", "batch", "corpus", "fragments", "templates", "last_dt", "hard_stop")
    def __init__(self):
        seed = (time.time_ns() ^ (id(self) << 1)) & 0xFFFFFFFFFFFFFFFF
        self.rnd = random.Random(seed)
        self.g = _G(GRAMMAR_TEXT, seed & 0xFFFFFFFF)
        self.start_time = None
        self.calls = 0
        self.batch = 450
        self.corpus = deque(maxlen=1200)
        self.fragments = self._init_fragments()
        self.templates = _make_templates(self.rnd)
        self.last_dt = 0.0
        self.hard_stop = 59.2

    def _init_fragments(self):
        fr = [
            "", " ", "\\n", "\\t", ";", ";;", ",", "(", ")", "*", ".", "=", "<", ">", "<=", ">=", "<>", "!=", "||",
            "--x\\n", "/*x*/", "/**/",
            "SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP",
            "NULL", "TRUE", "FALSE",
            "ORDER BY", "GROUP BY", "HAVING", "LIMIT", "OFFSET",
            "UNION", "UNION ALL", "DISTINCT",
            "JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN", "CROSS JOIN", "ON",
            "IN", "IS NULL", "IS NOT NULL", "LIKE", "BETWEEN", "EXISTS",
            "CASE WHEN", "THEN", "ELSE", "END",
        ]
        # include discovered keywords too
        if KEYWORDS:
            for _ in range(min(250, len(KEYWORDS))):
                fr.append(KEYWORDS[_])
        return fr

    def gen_one(self):
        r = self.rnd.random()
        if self.g.start and self.g.rules and r < 0.62:
            toks = self.g.gen(self.g.start, max_depth=14 if self.rnd.random() < 0.85 else 20, max_tokens=140 if self.rnd.random() < 0.2 else 90)
            sql = _join_tokens(toks, self.rnd)
        elif r < 0.86 and self.templates:
            sql = self.templates[self.rnd.randrange(0, len(self.templates))]()
        else:
            if self.corpus and self.rnd.random() < 0.75:
                base = self.corpus[self.rnd.randrange(0, len(self.corpus))]
            else:
                base = "SELECT 1"
            sql = _mutate(base, self.rnd, self.fragments)
        if self.rnd.random() < 0.70:
            sql = _inject_noise(sql, self.rnd)
        # occasional multi-statement in one input
        if self.rnd.random() < 0.08:
            sql2 = self.templates[self.rnd.randrange(0, len(self.templates))]() if self.templates else "SELECT 1"
            sql = (sql.rstrip("; \\t\\r\\n") + ";" + self.rnd.choice(_WS_CHOICES) + sql2)
        # keep size reasonable
        if len(sql) > 4500:
            sql = sql[:4500]
        return sql

STATE = None

def fuzz(parse_sql):
    global STATE
    if STATE is None:
        STATE = _State()
        STATE.start_time = time.time()
        # seed corpus with a few focused statements
        seeds = [
            "SELECT 1",
            "SELECT * FROM t",
            "SELECT a,b FROM t WHERE a=1",
            "INSERT INTO t(a,b) VALUES (1,'x')",
            "UPDATE t SET a=2 WHERE b IS NULL",
            "DELETE FROM t WHERE a IN (1,2,3)",
            "CREATE TABLE t(a INT PRIMARY KEY, b TEXT, c REAL)",
            "DROP TABLE t",
            "SELECT CASE WHEN 1=1 THEN 'a' ELSE 'b' END",
            "SELECT 1 UNION SELECT 2 UNION ALL SELECT 3",
            "BEGIN TRANSACTION",
            "COMMIT",
            "ROLLBACK",
            "--comment\\nSELECT 1",
            "/*block*/SELECT 1",
        ]
        for s in seeds:
            STATE.corpus.append(s)

    now = time.time()
    elapsed = now - STATE.start_time
    if elapsed >= STATE.hard_stop:
        return False

    # adjust batch based on last call duration; target ~0.9s per fuzz call
    if STATE.calls > 2 and STATE.last_dt > 0.0:
        if STATE.last_dt < 0.55:
            STATE.batch = min(1400, int(STATE.batch * 1.18) + 1)
        elif STATE.last_dt > 1.25:
            STATE.batch = max(120, int(STATE.batch * 0.78))
        # if near end, shrink to finish quickly
        remaining = STATE.hard_stop - elapsed
        if remaining < 4.0:
            STATE.batch = max(80, min(STATE.batch, 240))
        if remaining < 1.5:
            STATE.batch = max(40, min(STATE.batch, 120))

    batch = STATE.batch
    rnd = STATE.rnd
    # ensure some very small and some pathological cases per batch
    stmts = []
    stmts_append = stmts.append

    # a few edge cases for tokenizer/parser error branches
    edge = [
        "",
        " ",
        "\\n",
        ";",
        ";;",
        "'",
        "'''''",
        "/*",
        "*/",
        "/* unterminated",
        "-- unterminated",
        "SELECT",
        "SELECT FROM",
        "INSERT INTO",
        "CREATE TABLE",
        "DROP",
        "SELECT 1e309",
        "SELECT 0xG",
        "SELECT \\ufeff1",
        "SELECT 'a\\n\\t\\r' ",
        "SELECT \"unterminated",
        "SELECT [unterminated",
        "SELECT `unterminated",
        "SELECT 1 + + 2",
        "SELECT (1",
        "SELECT 1)",
    ]
    for _ in range(14):
        stmts_append(edge[rnd.randrange(0, len(edge))])

    for _ in range(batch - len(stmts)):
        sql = STATE.gen_one()
        stmts_append(sql)

    t0 = time.time()
    parse_sql(stmts)
    t1 = time.time()
    STATE.last_dt = (t1 - t0)

    # update corpus with some of the newly generated statements
    step = max(1, len(stmts) // 30)
    for i in range(0, len(stmts), step):
        s = stmts[i]
        if 0 < len(s) <= 1800:
            STATE.corpus.append(s)

    STATE.calls += 1
    return True
"""
        return {"code": fuzzer_code}