import os
import sys
import re
import random
import importlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any


_PUNCT_NO_SPACE_BEFORE = {",", ")", ";", ".", "]"}
_PUNCT_NO_SPACE_AFTER = {"(", ".", "["}
_MULTI_CHAR_OPS = {"<=", ">=", "<>", "!=", "==", "||", "&&", "::", "->", "->>"}


def _strip_comments_grammar(line: str) -> str:
    # Remove # comments and // comments (common in grammar docs)
    s = line
    for marker in ("#", "//"):
        idx = s.find(marker)
        if idx != -1:
            s = s[:idx]
    return s.rstrip("\n")


def _top_level_split(s: str, sep: str = "|") -> List[str]:
    out = []
    buf = []
    depth_par = depth_br = depth_cb = 0
    in_sq = False
    in_dq = False
    esc = False
    for ch in s:
        if esc:
            buf.append(ch)
            esc = False
            continue
        if ch == "\\":
            buf.append(ch)
            esc = True
            continue
        if ch == "'" and not in_dq:
            in_sq = not in_sq
            buf.append(ch)
            continue
        if ch == '"' and not in_sq:
            in_dq = not in_dq
            buf.append(ch)
            continue
        if in_sq or in_dq:
            buf.append(ch)
            continue
        if ch == "(":
            depth_par += 1
            buf.append(ch)
            continue
        if ch == ")":
            depth_par = max(0, depth_par - 1)
            buf.append(ch)
            continue
        if ch == "[":
            depth_br += 1
            buf.append(ch)
            continue
        if ch == "]":
            depth_br = max(0, depth_br - 1)
            buf.append(ch)
            continue
        if ch == "{":
            depth_cb += 1
            buf.append(ch)
            continue
        if ch == "}":
            depth_cb = max(0, depth_cb - 1)
            buf.append(ch)
            continue
        if ch == sep and depth_par == 0 and depth_br == 0 and depth_cb == 0:
            out.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def _tokenize_rhs(rhs: str) -> List[str]:
    tokens = []
    i = 0
    n = len(rhs)
    while i < n:
        c = rhs[i]
        if c.isspace():
            i += 1
            continue
        if c in "()[]{}|?*+":
            tokens.append(c)
            i += 1
            continue
        if c in ("'", '"'):
            q = c
            j = i + 1
            buf = [q]
            while j < n:
                ch = rhs[j]
                buf.append(ch)
                if ch == q:
                    # handle doubled quote in SQL-style literals inside grammar quotes
                    if j + 1 < n and rhs[j + 1] == q:
                        buf.append(q)
                        j += 2
                        continue
                    j += 1
                    break
                if ch == "\\" and j + 1 < n:
                    buf.append(rhs[j + 1])
                    j += 2
                    continue
                j += 1
            tokens.append("".join(buf))
            i = j
            continue
        if c == "<":
            j = rhs.find(">", i + 1)
            if j != -1:
                tokens.append(rhs[i:j + 1])
                i = j + 1
                continue
        if c == "/":
            # regex-ish token /.../
            j = i + 1
            buf = [c]
            esc = False
            while j < n:
                ch = rhs[j]
                buf.append(ch)
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == "/":
                    j += 1
                    break
                j += 1
            tokens.append("".join(buf))
            i = j
            continue
        # operators/punct
        # Try multi-char operators
        for op in sorted(_MULTI_CHAR_OPS, key=len, reverse=True):
            if rhs.startswith(op, i):
                tokens.append(op)
                i += len(op)
                break
        else:
            if c in ",;.=<>+-/%":
                # handle two-char comparisons not in _MULTI_CHAR_OPS
                if i + 1 < n and rhs[i:i + 2] in ("<=", ">=", "<>", "!=", "=="):
                    tokens.append(rhs[i:i + 2])
                    i += 2
                else:
                    tokens.append(c)
                    i += 1
                continue
            # word
            j = i
            while j < n and (rhs[j].isalnum() or rhs[j] in ("_", "$")):
                j += 1
            if j > i:
                tokens.append(rhs[i:j])
                i = j
            else:
                # unknown char, keep it
                tokens.append(c)
                i += 1
    return tokens


@dataclass(frozen=True)
class _Node:
    pass


@dataclass(frozen=True)
class _Tok(_Node):
    v: str


@dataclass(frozen=True)
class _NonTerm(_Node):
    name: str


@dataclass(frozen=True)
class _Seq(_Node):
    items: Tuple[_Node, ...]


@dataclass(frozen=True)
class _Alt(_Node):
    alts: Tuple[_Node, ...]


@dataclass(frozen=True)
class _Opt(_Node):
    item: _Node


@dataclass(frozen=True)
class _Rep(_Node):
    item: _Node


class _RHSParser:
    def __init__(self, tokens: List[str]):
        self.toks = tokens
        self.i = 0

    def _peek(self) -> Optional[str]:
        if self.i >= len(self.toks):
            return None
        return self.toks[self.i]

    def _eat(self, t: str) -> bool:
        if self._peek() == t:
            self.i += 1
            return True
        return False

    def parse(self) -> _Node:
        node = self._parse_alt()
        return node

    def _parse_alt(self) -> _Node:
        parts = [self._parse_seq()]
        while self._eat("|"):
            parts.append(self._parse_seq())
        if len(parts) == 1:
            return parts[0]
        return _Alt(tuple(parts))

    def _parse_seq(self) -> _Node:
        items = []
        while True:
            t = self._peek()
            if t is None or t in ("|", ")", "]", "}"):
                break
            items.append(self._parse_atom())
        if not items:
            return _Seq(tuple())
        if len(items) == 1:
            return items[0]
        return _Seq(tuple(items))

    def _parse_atom(self) -> _Node:
        t = self._peek()
        if t is None:
            return _Seq(tuple())
        if self._eat("("):
            inner = self._parse_alt()
            self._eat(")")
            node = inner
        elif self._eat("["):
            inner = self._parse_alt()
            self._eat("]")
            node = _Opt(inner)
        elif self._eat("{"):
            inner = self._parse_alt()
            self._eat("}")
            node = _Rep(inner)
        else:
            self.i += 1
            if t.startswith("<") and t.endswith(">"):
                node = _NonTerm(t[1:-1].strip())
            else:
                node = _Tok(t)

        nxt = self._peek()
        if nxt in ("?", "*", "+"):
            self.i += 1
            if nxt == "?":
                node = _Opt(node)
            elif nxt == "*":
                node = _Rep(node)
            else:  # +
                node = _Seq((node, _Rep(node)))
        return node


def _unquote_terminal(tok: str) -> str:
    if len(tok) >= 2 and ((tok[0] == tok[-1] == "'") or (tok[0] == tok[-1] == '"')):
        return tok[1:-1]
    return tok


def _materialize_token(tok: str, rng: random.Random) -> str:
    t = _unquote_terminal(tok)
    tu = t.upper()
    # map common lexer placeholders to concrete SQL
    if t.startswith("<") and t.endswith(">"):
        # should not happen here; handled as NonTerm
        t = t[1:-1]
        tu = t.upper()

    if t.startswith("/") and t.endswith("/") and len(t) > 2:
        # regex placeholder
        return "x"

    # Common lexical symbols / types
    if tu in ("IDENT", "IDENTIFIER", "ID", "NAME", "COLUMN", "COLUMN_NAME", "TABLE", "TABLE_NAME"):
        return rng.choice(["t", "t1", "t2", "col", "col1", "col2", "id"])
    if "IDENT" in tu or "NAME" in tu:
        return rng.choice(["t", "t1", "t2", "col", "col1", "col2", "id"])
    if "TABLE" in tu:
        return rng.choice(["t", "t1", "t2", "my_table"])
    if "COLUMN" in tu or "FIELD" in tu:
        return rng.choice(["a", "b", "c", "id", "col1"])
    if "NUMBER" in tu or "INT" in tu or "DIGIT" in tu:
        return rng.choice(["0", "1", "2", "10", "3.14", "-1", "1e3"])
    if "STRING" in tu or "TEXT" in tu or "CHAR" in tu:
        return rng.choice(["'x'", "'y'", "'a''b'", "''"])
    if "BOOL" in tu:
        return rng.choice(["TRUE", "FALSE"])
    if tu == "NULL":
        return "NULL"

    # keep keywords/operators/punct unchanged
    return t


def _sql_detokenize(tokens: List[str]) -> str:
    if not tokens:
        return ""
    # merge some adjacent operators if accidentally split
    merged = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            pair = tokens[i] + tokens[i + 1]
            if pair in _MULTI_CHAR_OPS:
                merged.append(pair)
                i += 2
                continue
        merged.append(tokens[i])
        i += 1

    out = []
    for j, tok in enumerate(merged):
        if not out:
            out.append(tok)
            continue
        prev = out[-1]
        if tok in _PUNCT_NO_SPACE_BEFORE:
            out.append(tok)
        elif prev in _PUNCT_NO_SPACE_AFTER:
            out.append(tok)
        else:
            # avoid spaces around dot chains: a . b -> a.b
            if prev == ".":
                out.append(tok)
            else:
                out.append(" " + tok)
    s = "".join(out).strip()
    s = re.sub(r"\s+", " ", s)
    # compact spaces around some punct
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r"\s+;", ";", s)
    return s


class _GrammarExpander:
    def __init__(self, rules: Dict[str, _Node], start: str, rng: random.Random):
        self.rules = rules
        self.start = start
        self.rng = rng
        self._alt_choice_counter: Dict[str, int] = {}

    def _choose_alt(self, name: str, node: _Node) -> _Node:
        if not isinstance(node, _Alt):
            return node
        k = self._alt_choice_counter.get(name, 0)
        self._alt_choice_counter[name] = k + 1
        # cycle through alternatives deterministically, with slight randomness
        idx = k % len(node.alts)
        if self.rng.random() < 0.15:
            idx = self.rng.randrange(len(node.alts))
        return node.alts[idx]

    def _fallback_nonterm(self, name: str) -> List[str]:
        nu = name.upper()
        if "IDENT" in nu or "NAME" in nu:
            return ["t"]
        if "TABLE" in nu:
            return ["t"]
        if "COLUMN" in nu or "FIELD" in nu:
            return ["a"]
        if "STRING" in nu or "TEXT" in nu or "CHAR" in nu:
            return ["'x'"]
        if "NUMBER" in nu or "INT" in nu or "DIGIT" in nu:
            return ["1"]
        if "BOOL" in nu:
            return ["TRUE"]
        if "NULL" in nu:
            return ["NULL"]
        if "OP" in nu:
            return ["="]
        return ["t"]

    def expand(self, node: _Node, depth: int, nt_stack: Tuple[str, ...] = ()) -> List[str]:
        if isinstance(node, _Tok):
            v = node.v
            if v in ("ε", "EPSILON", "epsilon", "EMPTY", "empty"):
                return []
            return [_materialize_token(v, self.rng)]
        if isinstance(node, _NonTerm):
            name = node.name
            # break cycles
            if depth <= 0 or name in nt_stack:
                return self._fallback_nonterm(name)
            rule = self.rules.get(name)
            if rule is None:
                return self._fallback_nonterm(name)
            chosen = self._choose_alt(name, rule)
            return self.expand(chosen, depth - 1, nt_stack + (name,))
        if isinstance(node, _Seq):
            out = []
            for it in node.items:
                out.extend(self.expand(it, depth, nt_stack))
            return out
        if isinstance(node, _Alt):
            chosen = self._choose_alt("", node)
            return self.expand(chosen, depth, nt_stack)
        if isinstance(node, _Opt):
            if self.rng.random() < 0.55:
                return self.expand(node.item, depth, nt_stack)
            return []
        if isinstance(node, _Rep):
            # repeat 0..k with k limited by depth
            kmax = 2 if depth > 2 else 1
            if self.rng.random() < 0.55:
                reps = self.rng.randint(0, kmax)
            else:
                reps = 0
            out = []
            for _ in range(reps):
                out.extend(self.expand(node.item, max(0, depth - 1), nt_stack))
            return out
        return []

    def generate(self, count: int = 200, max_depth: int = 10) -> List[str]:
        res = []
        for _ in range(count):
            toks = self.expand(_NonTerm(self.start), max_depth)
            s = _sql_detokenize([t for t in toks if t and t.upper() not in ("WHITESPACE", "WS")])
            s = s.strip()
            if not s:
                continue
            res.append(s)
        return res


def _load_grammar(grammar_path: str) -> Tuple[Dict[str, _Node], Optional[str]]:
    if not os.path.exists(grammar_path):
        return {}, None
    rules_raw: Dict[str, str] = {}
    with open(grammar_path, "r", encoding="utf-8", errors="ignore") as f:
        pending_lhs = None
        pending_rhs = []
        for line in f:
            line = _strip_comments_grammar(line).strip()
            if not line:
                continue
            if "::=" in line or ":=" in line:
                if pending_lhs is not None and pending_rhs:
                    rules_raw[pending_lhs] = " ".join(pending_rhs).strip()
                sep = "::=" if "::=" in line else ":="
                lhs, rhs = line.split(sep, 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
                # normalize lhs: remove <> if present
                if lhs.startswith("<") and lhs.endswith(">"):
                    lhs = lhs[1:-1].strip()
                pending_lhs = lhs
                pending_rhs = [rhs] if rhs else []
            else:
                if pending_lhs is not None:
                    pending_rhs.append(line)
        if pending_lhs is not None and pending_rhs:
            rules_raw[pending_lhs] = " ".join(pending_rhs).strip()

    rules: Dict[str, _Node] = {}
    for lhs, rhs in rules_raw.items():
        rhs = rhs.strip()
        if not rhs:
            rules[lhs] = _Seq(tuple())
            continue
        # Some grammars use ';' at end of rule; remove if so
        if rhs.endswith(";") and rhs.count("'") == 0 and rhs.count('"') == 0:
            rhs = rhs[:-1].strip()
        try:
            toks = _tokenize_rhs(rhs)
            parser = _RHSParser(toks)
            node = parser.parse()
            rules[lhs] = node
        except Exception:
            # fallback: treat rhs as a literal token sequence
            tks = [t for t in re.split(r"\s+", rhs) if t]
            rules[lhs] = _Seq(tuple(_Tok(t) for t in tks))

    start = None
    if rules:
        # heuristics to pick start symbol
        names = list(rules.keys())
        lower_map = {n.lower(): n for n in names}
        for cand in (
            "sql_stmt_list", "sqlstmtlist", "stmt_list", "statement_list", "statements",
            "sql", "start", "statement", "stmt", "query", "program", "input"
        ):
            if cand in lower_map:
                start = lower_map[cand]
                break
        if start is None:
            # try find one that references multiple statements
            for n in names:
                if "stmt" in n.lower() and "list" in n.lower():
                    start = n
                    break
        if start is None:
            start = names[0]
    return rules, start


def _normalize_stmt(s: str) -> str:
    s2 = s.strip()
    s2 = re.sub(r"\s+", " ", s2)
    # normalize trailing semicolon spacing
    s2 = re.sub(r"\s*;\s*$", ";", s2)
    return s2


def _features(stmt: str) -> set:
    # Extract rough token features to greedily select diverse statements
    # words, operators, punctuation
    parts = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|<=|>=|<>|!=|==|\|\||&&|::|->>|->|[(),;.*=<>+\-/%]", stmt)
    feats = set()
    for p in parts:
        if re.match(r"[A-Za-z_]", p):
            feats.add(p.upper())
        else:
            feats.add(p)
    return feats


class Solution:
    def solve(self, resources_path: str) -> List[str]:
        resources_path = os.path.abspath(resources_path)

        # Prepare import path for sql_engine
        if resources_path not in sys.path:
            sys.path.insert(0, resources_path)

        parse_sql = None
        try:
            sql_engine = importlib.import_module("sql_engine")
            parse_sql = getattr(sql_engine, "parse_sql", None)
        except Exception:
            parse_sql = None

        if parse_sql is None:
            try:
                parser_mod = importlib.import_module("sql_engine.parser")
                parse_sql = getattr(parser_mod, "parse_sql", None)
            except Exception:
                parse_sql = None

        if parse_sql is None:
            # As a last resort, return some common SQL; evaluation will filter invalid,
            # but we must return something.
            return [
                "SELECT 1",
                "SELECT * FROM t",
                "INSERT INTO t(a) VALUES (1)",
                "UPDATE t SET a=1",
                "DELETE FROM t",
            ]

        rng = random.Random(0)

        curated = [
            "SELECT 1",
            "SELECT 1 + 2 * 3",
            "SELECT -1, +2, 3.14, 1e3, 0",
            "SELECT NULL, TRUE, FALSE",
            "SELECT 'x', 'a''b', ''",
            "SELECT \"a\", \"weird\"\"name\"",
            "SELECT `a`, `t`",
            "SELECT [a], [t]",
            "SELECT * FROM t",
            "SELECT a, b AS bb, c FROM t",
            "SELECT DISTINCT a FROM t",
            "SELECT ALL a FROM t",
            "SELECT a FROM t WHERE a = 1",
            "SELECT a FROM t WHERE a <> 1",
            "SELECT a FROM t WHERE a != 1",
            "SELECT a FROM t WHERE a < 1 OR a > 2 AND b = 3",
            "SELECT a FROM t WHERE NOT (a = 1)",
            "SELECT a FROM t WHERE a IS NULL",
            "SELECT a FROM t WHERE a IS NOT NULL",
            "SELECT a FROM t WHERE a IN (1, 2, 3)",
            "SELECT a FROM t WHERE a NOT IN (SELECT a FROM t2)",
            "SELECT a FROM t WHERE a BETWEEN 1 AND 10",
            "SELECT a FROM t WHERE a NOT BETWEEN 1 AND 10",
            "SELECT a FROM t WHERE a LIKE '%x%'",
            "SELECT a FROM t WHERE a NOT LIKE '%x%'",
            r"SELECT a FROM t WHERE a LIKE '%\_%' ESCAPE '\'",
            "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.id = t.id)",
            "SELECT a FROM t WHERE a = (SELECT MAX(a) FROM t2)",
            "SELECT a FROM t WHERE a = ANY (SELECT a FROM t2)",
            "SELECT a FROM t WHERE a = ALL (SELECT a FROM t2)",
            "SELECT a FROM t ORDER BY a",
            "SELECT a FROM t ORDER BY a DESC, b ASC",
            "SELECT a FROM t ORDER BY 1 DESC",
            "SELECT a FROM t LIMIT 10",
            "SELECT a FROM t LIMIT 10 OFFSET 5",
            "SELECT a FROM t OFFSET 5",
            "SELECT a, COUNT(*) FROM t GROUP BY a",
            "SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(*) > 1",
            "SELECT SUM(a), AVG(a), MIN(a), MAX(a) FROM t",
            "SELECT COALESCE(a, 0), NULLIF(a, 0) FROM t",
            "SELECT CAST(a AS INTEGER) FROM t",
            "SELECT CASE WHEN a > 0 THEN 'pos' WHEN a = 0 THEN 'zero' ELSE 'neg' END FROM t",
            "SELECT a FROM t1 INNER JOIN t2 ON t1.id = t2.id",
            "SELECT a FROM t1 LEFT JOIN t2 ON t1.id = t2.id",
            "SELECT a FROM t1 RIGHT JOIN t2 ON t1.id = t2.id",
            "SELECT a FROM t1 FULL JOIN t2 ON t1.id = t2.id",
            "SELECT a FROM t1 CROSS JOIN t2",
            "SELECT a FROM t1 NATURAL JOIN t2",
            "SELECT a FROM t1 JOIN t2 USING (id)",
            "SELECT * FROM (SELECT 1 AS x, 2 AS y) sub",
            "SELECT sub.x FROM (SELECT 1 AS x) sub WHERE sub.x = 1",
            "SELECT a FROM t UNION SELECT a FROM t2",
            "SELECT a FROM t UNION ALL SELECT a FROM t2",
            "SELECT a FROM t INTERSECT SELECT a FROM t2",
            "SELECT a FROM t EXCEPT SELECT a FROM t2",
            "WITH cte AS (SELECT 1 AS a) SELECT a FROM cte",
            "WITH RECURSIVE cte(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cte WHERE x<3) SELECT x FROM cte",
            "VALUES (1, 2), (3, 4)",
            "INSERT INTO t(a, b) VALUES (1, 'x')",
            "INSERT INTO t(a, b) VALUES (1, 'x'), (2, 'y')",
            "INSERT INTO t DEFAULT VALUES",
            "INSERT INTO t(a) SELECT a FROM t2",
            "UPDATE t SET a = 1",
            "UPDATE t SET a = a + 1, b = 'x' WHERE id = 10",
            "DELETE FROM t",
            "DELETE FROM t WHERE id = 1",
            "CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT NOT NULL, b REAL DEFAULT 1.0)",
            "CREATE TABLE t (a INT, b TEXT, CHECK (a > 0), UNIQUE (b))",
            "CREATE TEMP TABLE t (a INT)",
            "DROP TABLE t",
            "ALTER TABLE t ADD COLUMN c TEXT",
            "CREATE INDEX idx_t_a ON t(a)",
            "DROP INDEX idx_t_a",
            "BEGIN",
            "COMMIT",
            "ROLLBACK",
            "PRAGMA cache_size = 1000",
            "EXPLAIN SELECT * FROM t",
            "SELECT /*block comment*/ 1",
            "SELECT 1 -- line comment\n",
            "SELECT 1; SELECT 2",
            "CREATE TABLE t(a INT); INSERT INTO t(a) VALUES(1); SELECT a FROM t",
        ]

        # Try grammar generation
        grammar_path = os.path.join(resources_path, "sql_grammar.txt")
        rules, start = _load_grammar(grammar_path)
        grammar_candidates = []
        if rules and start:
            exp = _GrammarExpander(rules, start, random.Random(0))
            grammar_candidates = exp.generate(count=350, max_depth=11)

        candidates = curated + grammar_candidates

        def _try_parse(stmt: str) -> bool:
            try:
                parse_sql(stmt)
                return True
            except Exception:
                return False

        valid: Dict[str, str] = {}
        # validate and normalize; also attempt semicolon variations
        for s in candidates:
            s0 = s.strip()
            if not s0:
                continue

            trials = [s0]
            if not s0.endswith(";"):
                trials.append(s0 + ";")
            else:
                trials.append(s0[:-1].rstrip())

            ok_stmt = None
            for tr in trials:
                if not tr.strip():
                    continue
                if _try_parse(tr):
                    ok_stmt = tr
                    break
            if ok_stmt is None:
                continue
            key = _normalize_stmt(ok_stmt)
            if key not in valid:
                valid[key] = ok_stmt.strip()

        valid_list = list(valid.values())
        if not valid_list:
            # ensure at least a minimal valid statement, try a couple
            for s in ("SELECT 1", "SELECT 1;", "VALUES (1)", "VALUES (1);"):
                if _try_parse(s):
                    return [s]
            return ["SELECT 1"]

        # Greedy selection for diversity with a small cap
        cap = 35
        items = []
        for s in valid_list:
            feats = _features(s)
            items.append((s, feats))

        selected = []
        covered = set()
        remaining = items[:]

        # seed with a multi-statement if available and valid (often boosts parser coverage)
        multi = [it for it in remaining if it[0].count(";") >= 2]
        if multi:
            multi.sort(key=lambda x: (len(x[1]), len(x[0])), reverse=True)
            s, f = multi[0]
            selected.append(s)
            covered |= f
            remaining = [it for it in remaining if it[0] != s]

        while remaining and len(selected) < cap:
            best_idx = -1
            best_gain = -1
            best_score = None
            for idx, (s, f) in enumerate(remaining):
                gain = len(f - covered)
                # Prefer higher gain, then more total features, then longer (often exercises more)
                score = (gain, len(f), min(len(s), 2000))
                if score > (best_gain, -1 if best_score is None else best_score[1], -1 if best_score is None else best_score[2]):
                    best_gain = gain
                    best_score = score
                    best_idx = idx
            if best_idx < 0:
                break
            s, f = remaining.pop(best_idx)
            selected.append(s)
            covered |= f
            # stop if additional gain is consistently small and we already have some
            if len(selected) >= 20 and best_gain <= 1 and len(covered) >= 60:
                break

        # Ensure deterministic ordering and remove any empty
        selected = [x.strip() for x in selected if x and x.strip()]
        if not selected:
            selected = [valid_list[0].strip()]
        return selected[:cap]