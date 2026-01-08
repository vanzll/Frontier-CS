import os
import sys
import re
import importlib
from collections import defaultdict, deque

_EPS_TOKENS = {"ε", "EPSILON", "epsilon", "EMPTY", "empty", "E", "e", "null", "NULL"}  # handle later carefully


def _safe_read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        try:
            with open(path, "r", encoding="latin-1") as f:
                return f.read()
        except Exception:
            return ""


def _strip_inline_comment(s: str) -> str:
    # remove BNF-ish comments; keep quoted parts intact
    out = []
    i = 0
    in_sq = False
    in_dq = False
    while i < len(s):
        ch = s[i]
        if ch == "'" and not in_dq:
            in_sq = not in_sq
            out.append(ch)
            i += 1
            continue
        if ch == '"' and not in_sq:
            in_dq = not in_dq
            out.append(ch)
            i += 1
            continue
        if not in_sq and not in_dq:
            if ch == "#":
                break
            if ch == "/" and i + 1 < len(s) and s[i + 1] == "/":
                break
            if ch == "-" and i + 1 < len(s) and s[i + 1] == "-":
                break
        out.append(ch)
        i += 1
    return "".join(out).strip()


def _tok_rhs(rhs: str) -> list[str]:
    tokens = []
    i = 0
    n = len(rhs)
    while i < n:
        ch = rhs[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "|[]{}":
            tokens.append(ch)
            i += 1
            continue
        if ch in ("'", '"'):
            q = ch
            j = i + 1
            while j < n:
                if rhs[j] == q:
                    if q == "'" and j + 1 < n and rhs[j + 1] == "'":
                        j += 2
                        continue
                    break
                if rhs[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                j += 1
            if j >= n:
                lit = rhs[i:]
                i = n
            else:
                lit = rhs[i : j + 1]
                i = j + 1
            # keep literal as-is (including quotes), but trim spaces
            tokens.append(lit.strip())
            continue
        if ch == "<":
            j = rhs.find(">", i + 1)
            if j == -1:
                # treat as symbol
                tokens.append("<")
                i += 1
            else:
                nt = rhs[i + 1 : j].strip()
                tokens.append(f"<{nt}>")
                i = j + 1
            continue
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (rhs[j].isalnum() or rhs[j] == "_"):
                j += 1
            tokens.append(rhs[i:j])
            i = j
            continue

        # multi-char operators/punct
        two = rhs[i : i + 2]
        three = rhs[i : i + 3]
        if three in ("!~~", "!~*", "~~*", "|||"):
            tokens.append(three)
            i += 3
            continue
        if two in ("<=", ">=", "<>", "!=", "||", "::", "==", "->", "=>"):
            tokens.append(two)
            i += 2
            continue

        # single char punctuation/op
        tokens.append(ch)
        i += 1
    return [t for t in tokens if t and not t.isspace()]


def _parse_rhs_tokens(tokens: list[str]) -> list[list[str]]:
    # EBNF-ish: [...] optional, {...} repetition (0/1/2 occurrences), | alternation
    def parse_expr(pos: int, stop: set[str]) -> tuple[list[list[str]], int]:
        alts, pos = parse_concat(pos, stop | {"|"})
        res = list(alts)
        while pos < len(tokens) and tokens[pos] == "|":
            pos += 1
            more, pos = parse_concat(pos, stop | {"|"})
            res.extend(more)
        return res, pos

    def parse_concat(pos: int, stop: set[str]) -> tuple[list[list[str]], int]:
        cur = [[]]
        while pos < len(tokens) and tokens[pos] not in stop:
            tok = tokens[pos]
            if tok == "[":
                sub, pos = parse_expr(pos + 1, {"]"})
                if pos < len(tokens) and tokens[pos] == "]":
                    pos += 1
                # optional: empty + each sub
                new_cur = []
                for c in cur:
                    new_cur.append(list(c))
                    for s in sub:
                        new_cur.append(c + s)
                cur = _dedup_seqs(new_cur, limit=256)
                continue
            if tok == "{":
                sub, pos = parse_expr(pos + 1, {"}"})
                if pos < len(tokens) and tokens[pos] == "}":
                    pos += 1
                # repetition: 0, 1, 2 occurrences
                new_cur = []
                for c in cur:
                    new_cur.append(list(c))
                    for s in sub:
                        new_cur.append(c + s)
                    for s in sub:
                        new_cur.append(c + s + s)
                cur = _dedup_seqs(new_cur, limit=256)
                continue
            # normal token
            pos += 1
            for c in cur:
                c.append(tok)
        return cur, pos

    alts, pos = parse_expr(0, set())
    # ignore trailing tokens if any
    return _dedup_seqs(alts, limit=512)


def _dedup_seqs(seqs: list[list[str]], limit: int = 512) -> list[list[str]]:
    seen = set()
    out = []
    for s in seqs:
        key = tuple(s)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def _strip_angle(sym: str) -> str:
    sym = sym.strip()
    if sym.startswith("<") and sym.endswith(">"):
        return sym[1:-1].strip()
    return sym


def _parse_grammar_file(grammar_text: str) -> dict[str, list[list[str]]]:
    prods: dict[str, list[list[str]]] = defaultdict(list)
    if not grammar_text:
        return {}

    for raw in grammar_text.splitlines():
        line = _strip_inline_comment(raw)
        if not line:
            continue
        if "::=" in line:
            lhs, rhs = line.split("::=", 1)
        elif ":=" in line:
            lhs, rhs = line.split(":=", 1)
        elif "->" in line:
            lhs, rhs = line.split("->", 1)
        else:
            continue
        lhs = lhs.strip()
        rhs = rhs.strip()
        if not lhs or not rhs:
            continue

        lhs_sym = _strip_angle(lhs)
        rhs_tokens = _tok_rhs(rhs)

        # Normalize angle nonterminals in RHS to bare names but mark as nonterminal tokens by checking later
        norm_tokens = []
        for t in rhs_tokens:
            if t.startswith("<") and t.endswith(">"):
                norm_tokens.append(_strip_angle(t))
            else:
                # Strip quotes around terminals if they are quoted keywords/symbols.
                if len(t) >= 2 and ((t[0] == "'" and t[-1] == "'") or (t[0] == '"' and t[-1] == '"')):
                    # keep SQL string literals quoted; but grammar terminals for keywords are often quoted.
                    inner = t[1:-1]
                    # heuristic: if inner looks like a keyword/symbol (no spaces) then use inner; else keep quoted.
                    if inner and not any(c.isspace() for c in inner) and not (t[0] == "'" and ("''" in inner or "\\" in inner)):
                        norm_tokens.append(inner)
                    else:
                        norm_tokens.append(t)
                else:
                    norm_tokens.append(t)

        alts = _parse_rhs_tokens(norm_tokens)
        # Filter epsilon-like tokens
        cleaned_alts = []
        for alt in alts:
            cleaned = []
            for tok in alt:
                if tok in ("ε", "EPSILON", "epsilon", "EMPTY", "empty"):
                    continue
                cleaned.append(tok)
            cleaned_alts.append(cleaned)
        for alt in cleaned_alts:
            prods[lhs_sym].append(alt)

    # Dedup and cap alternatives per production
    final = {}
    for k, alts in prods.items():
        final[k] = _dedup_seqs(alts, limit=512)
    return final


def _sql_join(tokens: list[str]) -> str:
    # Conservative joining: spaces between most tokens; no spaces around '.', and no space before , ) ;
    out = []
    prev = ""
    for tok in tokens:
        if tok in ("", None):
            continue
        if tok in ("ε", "EPSILON", "epsilon", "EMPTY", "empty"):
            continue
        if tok == ".":
            if out:
                out[-1] = out[-1].rstrip()
                out[-1] += "."
            else:
                out.append(".")
            prev = "."
            continue
        if tok in (",", ")", ";"):
            if out:
                out[-1] = out[-1].rstrip()
                out[-1] += tok
            else:
                out.append(tok)
            prev = tok
            continue
        if prev == ".":
            if out:
                out[-1] += tok
            else:
                out.append(tok)
            prev = tok
            continue
        # default: add with space separation
        if not out:
            out.append(tok)
        else:
            # no space after '('
            if prev == "(":
                out.append(tok)
            else:
                out.append(" " + tok)
        prev = tok
    s = "".join(out).strip()
    # Remove spaces before '(' for function-like tokens only if it helps (leave as-is; tokenizer can handle spaces).
    return s


def _is_identifier_like(tok: str) -> bool:
    if not tok:
        return False
    if tok[0] in ("'", '"', "`"):
        return True
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", tok))


def _placeholder_for_symbol(name: str) -> str:
    if not name:
        return "t1"
    n = name.lower()
    if any(x in n for x in ("table", "relation")):
        return "t1"
    if any(x in n for x in ("column", "field", "attr")):
        return "c1"
    if any(x in n for x in ("schema", "database", "db")):
        return "db1"
    if any(x in n for x in ("view",)):
        return "v1"
    if any(x in n for x in ("index",)):
        return "idx1"
    if any(x in n for x in ("alias",)):
        return "a"
    if any(x in n for x in ("string", "text", "char")):
        return "'x'"
    if any(x in n for x in ("float", "double", "real", "decimal")):
        return "1.5"
    if any(x in n for x in ("int", "number", "digit", "count", "size", "limit", "offset")):
        return "1"
    if any(x in n for x in ("bool", "boolean")):
        return "TRUE"
    if any(x in n for x in ("null",)):
        return "NULL"
    if any(x in n for x in ("type", "datatype")):
        return "INT"
    if any(x in n for x in ("operator", "op")):
        return "="
    if any(x in n for x in ("expr", "expression")):
        return "1"
    return "t1"


def _render_tokens(tokens: list[str], productions: dict[str, list[list[str]]]) -> str:
    out = []
    for tok in tokens:
        if tok is None:
            continue
        tok = tok.strip()
        if not tok:
            continue
        if tok in ("ε", "EPSILON", "epsilon", "EMPTY", "empty"):
            continue

        # Some grammars use token-type placeholders (IDENTIFIER, STRING, NUMBER)
        tl = tok.lower()
        if tok in productions:
            # should not happen in final render
            out.append(_placeholder_for_symbol(tok))
            continue
        if tl in productions:
            out.append(_placeholder_for_symbol(tok))
            continue

        if tok in ("IDENTIFIER", "ID", "NAME", "ident", "identifier", "id", "name"):
            out.append("t1")
            continue
        if tok in ("COLUMN", "COLUMN_NAME", "column", "column_name"):
            out.append("c1")
            continue
        if tok in ("TABLE", "TABLE_NAME", "table", "table_name"):
            out.append("t1")
            continue
        if tok in ("STRING", "STRING_LITERAL", "str", "string", "string_literal"):
            out.append("'x'")
            continue
        if tok in ("NUMBER", "NUM", "INT", "INTEGER", "integer", "int", "num"):
            out.append("1")
            continue
        if tok in ("FLOAT", "REAL", "DOUBLE", "DECIMAL"):
            out.append("1.5")
            continue
        if tok in ("TRUE", "FALSE", "NULL"):
            out.append(tok)
            continue

        # Convert grammar terminals like <...> already stripped; if still angle, treat as placeholder
        if tok.startswith("<") and tok.endswith(">"):
            out.append(_placeholder_for_symbol(_strip_angle(tok)))
            continue

        # keep literal as is
        out.append(tok)

    stmt = _sql_join(out).strip()
    # cleanup doubled spaces
    stmt = re.sub(r"[ \t]+", " ", stmt).strip()
    return stmt


def _compute_alt_order(productions: dict[str, list[list[str]]]) -> dict[str, list[int]]:
    nts = set(productions.keys())

    def count_nts(alt: list[str]) -> int:
        return sum(1 for t in alt if t in nts)

    orders: dict[str, list[int]] = {}
    for nt, alts in productions.items():
        idxs = list(range(len(alts)))
        idxs.sort(key=lambda i: (count_nts(alts[i]), len(alts[i])))
        # diversity: take a few shortest + a few longer
        diverse = []
        if idxs:
            diverse.extend(idxs[: min(6, len(idxs))])
            # add some longer ones
            for i in idxs[-min(6, len(idxs)) :]:
                if i not in diverse:
                    diverse.append(i)
        orders[nt] = diverse
    return orders


def _infer_start_symbols(productions: dict[str, list[list[str]]]) -> list[str]:
    if not productions:
        return []
    keys = list(productions.keys())
    preferred = []
    for cand in (
        "sql",
        "program",
        "input",
        "statement_list",
        "statements",
        "statement",
        "stmt_list",
        "stmt",
        "query",
        "query_stmt",
        "sql_stmt",
    ):
        if cand in productions:
            preferred.append(cand)
    # add other stmt-like nonterminals
    stmt_like = [k for k in keys if re.search(r"(stmt|statement|query)$", k, re.IGNORECASE)]
    # stable order by appearance in file is lost; use alphabetical for determinism
    stmt_like = sorted(set(stmt_like) - set(preferred))
    # fall back to first alphabetical
    rest = sorted(set(keys) - set(preferred) - set(stmt_like))
    out = preferred + stmt_like + rest[:1]
    # unique preserving order
    seen = set()
    res = []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


def _generate_from_grammar(productions: dict[str, list[list[str]]], start: str, max_results: int = 120) -> list[str]:
    if not productions or start not in productions:
        return []
    nts = set(productions.keys())
    alt_order = _compute_alt_order(productions)

    def nonterm_count(tokens: list[str]) -> int:
        return sum(1 for t in tokens if t in nts)

    def first_nonterm(tokens: list[str]) -> tuple[int, str] | tuple[int, None]:
        for i, t in enumerate(tokens):
            if t in nts:
                return i, t
        return -1, None

    # Beam search expansion
    max_tokens = 220
    beam_size = 400
    max_steps = 80
    results = []
    seen_rendered = set()

    State = tuple[list[str], dict[str, int], int]  # tokens, nt_counts, steps

    init: State = ([start], {}, 0)
    beam: list[State] = [init]

    for _ in range(max_steps):
        if len(results) >= max_results:
            break
        new_beam: list[State] = []
        for tokens, counts, steps in beam:
            if len(results) >= max_results:
                break
            idx, nt = first_nonterm(tokens)
            if nt is None:
                stmt = _render_tokens(tokens, productions)
                if stmt and stmt not in seen_rendered and len(stmt) <= 2000:
                    seen_rendered.add(stmt)
                    results.append(stmt)
                continue

            c = counts.get(nt, 0)
            alts = productions.get(nt, [])
            if not alts:
                # replace with placeholder terminal
                repl = _placeholder_for_symbol(nt)
                new_toks = tokens[:idx] + [repl] + tokens[idx + 1 :]
                if len(new_toks) <= max_tokens:
                    new_counts = dict(counts)
                    new_counts[nt] = c + 1
                    new_beam.append((new_toks, new_counts, steps + 1))
                continue

            order = alt_order.get(nt, list(range(len(alts))))
            # limit branching
            if c >= 2:
                use_idxs = order[:1]
            elif c == 1:
                use_idxs = order[:3]
            else:
                use_idxs = order[:6]

            for ai in use_idxs:
                alt = alts[ai]
                if not alt:
                    new_toks = tokens[:idx] + tokens[idx + 1 :]
                else:
                    new_toks = tokens[:idx] + alt + tokens[idx + 1 :]
                if not new_toks:
                    continue
                if len(new_toks) > max_tokens:
                    continue
                new_counts = dict(counts)
                new_counts[nt] = c + 1
                new_beam.append((new_toks, new_counts, steps + 1))

        if not new_beam:
            break

        # Select best states (fewer nonterminals, shorter)
        new_beam.sort(key=lambda st: (nonterm_count(st[0]), len(st[0]), st[2]))
        # Dedup by token tuple prefix
        dedup = []
        seen = set()
        for tks, cnts, stp in new_beam:
            key = tuple(tks[:60])
            if key in seen:
                continue
            seen.add(key)
            dedup.append((tks, cnts, stp))
            if len(dedup) >= beam_size:
                break
        beam = dedup

    return results


def _first_keyword(stmt: str) -> str:
    s = stmt.strip()
    # strip leading SQL comments
    while True:
        s2 = s.lstrip()
        if s2.startswith("--"):
            nl = s2.find("\n")
            if nl == -1:
                return ""
            s = s2[nl + 1 :]
            continue
        if s2.startswith("/*"):
            end = s2.find("*/")
            if end == -1:
                return ""
            s = s2[end + 2 :]
            continue
        s = s2
        break
    m = re.match(r"([A-Za-z_]+)", s)
    return (m.group(1).upper() if m else "")


def _import_parse_sql(resources_path: str):
    if resources_path not in sys.path:
        sys.path.insert(0, resources_path)
    try:
        mod = importlib.import_module("sql_engine")
    except Exception:
        # fallback: maybe package is in resources_path/sql_engine directly
        engine_dir = os.path.join(resources_path, "sql_engine")
        if engine_dir not in sys.path:
            sys.path.insert(0, engine_dir)
        mod = importlib.import_module("sql_engine")
    for attr in ("parse_sql", "parse", "parseSQL"):
        if hasattr(mod, attr):
            return getattr(mod, attr)
    try:
        pmod = importlib.import_module("sql_engine.parser")
        for attr in ("parse_sql", "parse", "parseSQL"):
            if hasattr(pmod, attr):
                return getattr(pmod, attr)
    except Exception:
        pass
    raise ImportError("Could not find parse_sql in sql_engine")


class Solution:
    def solve(self, resources_path: str) -> list[str]:
        parse_sql = _import_parse_sql(resources_path)

        def is_valid(sql: str) -> bool:
            try:
                parse_sql(sql)
                return True
            except Exception:
                return False

        def add_stmt(stmt: str, out: list[str], seen: set[str]) -> None:
            if stmt is None:
                return
            s = stmt.strip()
            if not s:
                return
            variants = []
            variants.append(s)
            if not s.endswith(";"):
                variants.append(s + ";")
            if s.endswith(";"):
                variants.append(s[:-1].rstrip())
            # also attempt normalize multiple spaces
            variants.append(re.sub(r"[ \t]+", " ", s).strip())
            if not variants[-1].endswith(";"):
                variants.append(variants[-1] + ";")

            for v in variants:
                v = v.strip()
                if not v:
                    continue
                if v in seen:
                    continue
                if is_valid(v):
                    seen.add(v)
                    out.append(v)
                    return

        manual_candidates = [
            "SELECT 1",
            "SELECT 1 + 2 * 3",
            "SELECT -1",
            "SELECT (1)",
            "SELECT NULL",
            "SELECT TRUE",
            "SELECT FALSE",
            "SELECT 'a''b'",
            "SELECT 'x'",
            "SELECT 1e2",
            "SELECT 1.5",
            "SELECT 1/2",
            "SELECT 1 % 2",
            "SELECT 1 = 1",
            "SELECT 1 < 2",
            "SELECT 1 <= 2",
            "SELECT 1 > 2",
            "SELECT 1 >= 2",
            "SELECT 1 <> 2",
            "SELECT 1 != 2",
            "SELECT 'a' || 'b'",
            "SELECT * FROM t1",
            "SELECT c1 FROM t1",
            "SELECT c1, c2 FROM t1",
            "SELECT t1.c1 FROM t1",
            "SELECT c1 AS a FROM t1",
            "SELECT DISTINCT c1 FROM t1",
            "SELECT c1 FROM t1 WHERE c2 = 1",
            "SELECT c1 FROM t1 WHERE c2 <> 1",
            "SELECT c1 FROM t1 WHERE c2 != 1",
            "SELECT c1 FROM t1 WHERE c2 < 1 OR c3 > 2 AND c4 <= 3",
            "SELECT c1 FROM t1 WHERE NOT c2 = 1",
            "SELECT c1 FROM t1 WHERE c2 BETWEEN 1 AND 10",
            "SELECT c1 FROM t1 WHERE c2 IN (1, 2, 3)",
            "SELECT c1 FROM t1 WHERE c2 IN (SELECT c2 FROM t2)",
            "SELECT c1 FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.c2 = t1.c2)",
            "SELECT c1 FROM t1 WHERE c2 LIKE 'a%'",
            "SELECT c1 FROM t1 WHERE c2 IS NULL",
            "SELECT c1 FROM t1 WHERE c2 IS NOT NULL",
            "SELECT c1 FROM t1 ORDER BY c1",
            "SELECT c1 FROM t1 ORDER BY c1 DESC",
            "SELECT c1 FROM t1 ORDER BY c1 DESC, c2 ASC",
            "SELECT c1 FROM t1 LIMIT 10",
            "SELECT c1 FROM t1 LIMIT 10 OFFSET 5",
            "SELECT c1 FROM t1 GROUP BY c1",
            "SELECT c1, COUNT(*) FROM t1 GROUP BY c1 HAVING COUNT(*) > 1",
            "SELECT COALESCE(c1, 0) FROM t1",
            "SELECT NULLIF(c1, 0) FROM t1",
            "SELECT CAST(c1 AS INT) FROM t1",
            "SELECT CASE WHEN c1 > 0 THEN 'pos' ELSE 'neg' END FROM t1",
            "SELECT a.c1, b.c2 FROM t1 a JOIN t2 b ON a.id = b.id",
            "SELECT a.c1, b.c2 FROM t1 a INNER JOIN t2 b ON a.id = b.id",
            "SELECT a.c1, b.c2 FROM t1 a LEFT JOIN t2 b ON a.id = b.id",
            "SELECT a.c1 FROM t1 a CROSS JOIN t2 b",
            "SELECT * FROM (SELECT 1 AS x) sub",
            "SELECT * FROM t1 UNION SELECT * FROM t2",
            "SELECT * FROM t1 UNION ALL SELECT * FROM t2",
            "INSERT INTO t1 VALUES (1, 'x')",
            "INSERT INTO t1 (c1, c2) VALUES (1, 'x'), (2, 'y')",
            "INSERT INTO t1 (c1) SELECT c1 FROM t2",
            "UPDATE t1 SET c1 = 1",
            "UPDATE t1 SET c1 = c1 + 1 WHERE c2 = 'x'",
            "DELETE FROM t1",
            "DELETE FROM t1 WHERE c1 = 1",
            "CREATE TABLE t1 (id INT, c1 TEXT)",
            "CREATE TABLE t2 (id INT PRIMARY KEY, c1 TEXT NOT NULL, c2 INT DEFAULT 0)",
            "DROP TABLE t1",
            "ALTER TABLE t1 ADD COLUMN c2 INT",
            "ALTER TABLE t1 DROP COLUMN c2",
            "CREATE INDEX idx_t1_c1 ON t1 (c1)",
            "CREATE VIEW v1 AS SELECT c1 FROM t1",
            "BEGIN",
            "COMMIT",
            "ROLLBACK",
            "-- comment\nSELECT 1",
            "/* block */ SELECT 1",
            "SELECT \"c1\" FROM \"t1\"",
            "SELECT `c1` FROM `t1`",
            "SELECT 1; SELECT 2",
        ]

        out: list[str] = []
        seen: set[str] = set()

        # Add manual statements first
        for s in manual_candidates:
            add_stmt(s, out, seen)
            if len(out) >= 40:
                break

        # Parse grammar and generate additional
        grammar_path = os.path.join(resources_path, "sql_grammar.txt")
        grammar_text = _safe_read_text(grammar_path)
        productions = _parse_grammar_file(grammar_text)

        if productions:
            starts = _infer_start_symbols(productions)
            # Generate from multiple starts for diversity
            for st in starts[:8]:
                gen = _generate_from_grammar(productions, st, max_results=160)
                for g in gen:
                    # Encourage statement termination variation
                    add_stmt(g, out, seen)
                    if len(out) >= 55:
                        break
                if len(out) >= 55:
                    break

        # If still too few, add some fallbacks with semicolons/case variations
        if len(out) < 10:
            fallbacks = [
                "select 1;",
                "select * from t1;",
                "SELECT 1;",
                "SELECT * FROM t1;",
            ]
            for s in fallbacks:
                add_stmt(s, out, seen)

        # Diversify / limit count for efficiency without knowing coverage:
        # keep a mix by first keyword, plus some long/tricky ones
        if len(out) > 50:
            by_kw = defaultdict(list)
            for s in out:
                by_kw[_first_keyword(s)].append(s)

            selected = []
            # Always keep at least one per keyword, up to 3
            for kw in sorted(by_kw.keys()):
                group = by_kw[kw]
                group_sorted = sorted(group, key=lambda x: (-len(x), x))
                selected.extend(group_sorted[:3])

            # Add a few extra long/tricky statements overall
            remaining = [s for s in out if s not in set(selected)]
            remaining.sort(key=lambda x: (-len(x), x))
            selected.extend(remaining[: max(0, 50 - len(selected))])

            # preserve original order while filtering to selected set
            selected_set = set(selected)
            new_out = []
            for s in out:
                if s in selected_set and s not in new_out:
                    new_out.append(s)
                if len(new_out) >= 50:
                    break
            out = new_out

        return out