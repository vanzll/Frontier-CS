import os
import sys
import re
import random
from typing import List, Tuple, Dict, Set, Optional
from collections import Counter

try:
    import coverage  # type: ignore
    from coverage.parser import PythonParser  # type: ignore
except Exception:  # pragma: no cover
    coverage = None
    PythonParser = None


def _safe_read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _canon(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def _import_sql_engine(resources_path: str):
    rp = _canon(resources_path)
    if rp not in sys.path:
        sys.path.insert(0, rp)

    import importlib

    sql_engine = importlib.import_module("sql_engine")
    try:
        parse_sql = getattr(sql_engine, "parse_sql")
    except Exception:
        parse_sql = importlib.import_module("sql_engine.parser").parse_sql

    parser_mod = importlib.import_module("sql_engine.parser")
    tokenizer_mod = importlib.import_module("sql_engine.tokenizer")
    ast_nodes_mod = importlib.import_module("sql_engine.ast_nodes")

    files = [_canon(parser_mod.__file__), _canon(tokenizer_mod.__file__), _canon(ast_nodes_mod.__file__)]
    return parse_sql, files, parser_mod, tokenizer_mod, ast_nodes_mod


def _extract_upper_keywords_from_code(text: str) -> Set[str]:
    out = set()
    for m in re.finditer(r"""(['"])([A-Z_]{2,30})\1""", text):
        s = m.group(2)
        if "_" in s or s.isalpha():
            out.add(s)
    return out


def _try_parse(parse_sql, stmt: str) -> bool:
    try:
        parse_sql(stmt)
        return True
    except Exception:
        return False


def _pythonparser_for_file(filename: str):
    text = _safe_read_text(filename)
    try:
        p = PythonParser(text=text, filename=filename)
    except TypeError:  # pragma: no cover
        p = PythonParser(text, filename)
    p.parse_source()
    return p


def _get_totals(target_files: List[str]) -> Tuple[Set[Tuple[str, int]], Set[Tuple[str, int, int]]]:
    total_lines: Set[Tuple[str, int]] = set()
    total_arcs: Set[Tuple[str, int, int]] = set()
    if PythonParser is None:  # pragma: no cover
        return total_lines, total_arcs
    for fn in target_files:
        p = _pythonparser_for_file(fn)
        for ln in getattr(p, "statements", set()) or set():
            total_lines.add((fn, int(ln)))
        arcs = set()
        try:
            arcs = set(p.arcs() or [])
        except Exception:  # pragma: no cover
            arcs = set()
        for a, b in arcs:
            try:
                total_arcs.add((fn, int(a), int(b)))
            except Exception:
                pass
    return total_lines, total_arcs


def _measure_stmt_coverage(parse_sql, stmt: str, include_files: List[str]) -> Tuple[Set[Tuple[str, int]], Set[Tuple[str, int, int]]]:
    if coverage is None:  # pragma: no cover
        return set(), set()

    cov = coverage.Coverage(branch=True, include=[_canon(f) for f in include_files], data_file=None)
    cov.start()
    try:
        parse_sql(stmt)
    except Exception:
        pass
    finally:
        cov.stop()

    data = cov.get_data()
    lines_cov: Set[Tuple[str, int]] = set()
    arcs_cov: Set[Tuple[str, int, int]] = set()

    for fn in include_files:
        fnc = _canon(fn)
        try:
            lns = data.lines(fnc)
        except Exception:
            lns = None
        if lns:
            for ln in lns:
                try:
                    lines_cov.add((fnc, int(ln)))
                except Exception:
                    pass

        try:
            arcs = data.arcs(fnc)
        except Exception:
            arcs = None
        if arcs:
            for a, b in arcs:
                try:
                    arcs_cov.add((fnc, int(a), int(b)))
                except Exception:
                    pass

    return lines_cov, arcs_cov


def _weighted_cov(covered_lines: Set[Tuple[str, int]], covered_arcs: Set[Tuple[str, int, int]],
                  total_lines: Set[Tuple[str, int]], total_arcs: Set[Tuple[str, int, int]]) -> float:
    tl = len(total_lines) if total_lines else 1
    ta = len(total_arcs) if total_arcs else 1
    lc = len(covered_lines) / tl
    ac = len(covered_arcs) / ta
    return 0.6 * lc + 0.4 * ac


def _dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in items:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _normalize_for_set(stmt: str) -> str:
    s = re.sub(r"\s+", " ", stmt.strip())
    return s


def _variants(stmt: str) -> List[str]:
    s = stmt.strip()
    out = [s]
    if not s.endswith(";"):
        out.append(s + ";")
    if s.endswith(";"):
        out.append(s[:-1].strip())
    out.append("\n" + s + "\n")
    out.append("-- leading comment\n" + s)
    out.append("/*leading*/" + s)
    out.append(s.replace("SELECT ", "SELECT/*c*/ ", 1) if "SELECT " in s else s)
    out.append(s.replace(" FROM ", " /*c*/ FROM ", 1) if " FROM " in s else s)
    out.append(re.sub(r"\s+", "  ", s))
    out.append(re.sub(r"\s+", "\t", s))
    return _dedup_keep_order([x for x in out if x and len(x) <= 4000])


def _build_handcrafted_candidates() -> List[str]:
    base = []

    base += [
        "SELECT 1",
        "SELECT 1+2*3",
        "SELECT (1 + 2) * 3",
        "SELECT 'a''b'",
        "SELECT NULL",
        "SELECT 1 AS x",
        "SELECT 1 x",
        "SELECT * FROM t",
        "SELECT a, b, c FROM t",
        "SELECT t.a, t.b FROM t",
        "SELECT DISTINCT a FROM t",
        "SELECT a FROM t WHERE a = 1",
        "SELECT a FROM t WHERE a <> 1",
        "SELECT a FROM t WHERE a != 1",
        "SELECT a FROM t WHERE a < 2",
        "SELECT a FROM t WHERE a <= 2",
        "SELECT a FROM t WHERE a > 0",
        "SELECT a FROM t WHERE a >= 0",
        "SELECT a FROM t WHERE a IS NULL",
        "SELECT a FROM t WHERE a IS NOT NULL",
        "SELECT a FROM t WHERE NOT a = 1",
        "SELECT a FROM t WHERE a BETWEEN 1 AND 3",
        "SELECT a FROM t WHERE a IN (1, 2, 3)",
        "SELECT a FROM t WHERE a NOT IN (1, 2)",
        "SELECT a FROM t WHERE a LIKE 'a%'",
        "SELECT a FROM t WHERE a = 1 AND b = 2",
        "SELECT a FROM t WHERE a = 1 OR b = 2",
        "SELECT a FROM t ORDER BY a",
        "SELECT a FROM t ORDER BY a ASC, b DESC",
        "SELECT a FROM t LIMIT 10",
        "SELECT a FROM t LIMIT 10 OFFSET 5",
        "SELECT a FROM t GROUP BY a",
        "SELECT a, COUNT(*) FROM t GROUP BY a",
        "SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(*) > 1",
        "SELECT CASE WHEN a=1 THEN 'one' ELSE 'other' END FROM t",
        "SELECT COALESCE(a, 1) FROM t",
        "SELECT NULLIF(a, b) FROM t",
        "SELECT ABS(-1) FROM t",
        "SELECT CAST(a AS INTEGER) FROM t",
        "SELECT a FROM (SELECT a FROM t) x",
        "SELECT x.a FROM (SELECT a FROM t) x WHERE x.a = 1",
        "SELECT a FROM t1 UNION SELECT a FROM t2",
        "SELECT a FROM t1 UNION ALL SELECT a FROM t2",
        "SELECT a FROM t1 INTERSECT SELECT a FROM t2",
        "SELECT a FROM t1 EXCEPT SELECT a FROM t2",
        "SELECT t1.a, t2.b FROM t1 JOIN t2 ON t1.id = t2.id",
        "SELECT t1.a, t2.b FROM t1 INNER JOIN t2 ON t1.id = t2.id",
        "SELECT t1.a, t2.b FROM t1 LEFT JOIN t2 ON t1.id = t2.id",
        "SELECT t1.a, t2.b FROM t1 LEFT OUTER JOIN t2 ON t1.id = t2.id",
        "SELECT t1.a FROM t1 CROSS JOIN t2",
        "SELECT a FROM t WHERE a IN (SELECT a FROM t2)",
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM t2)",
        "WITH c AS (SELECT 1 AS a) SELECT a FROM c",
        "WITH RECURSIVE c AS (SELECT 1 AS a UNION ALL SELECT a+1 FROM c) SELECT a FROM c",
        "VALUES (1, 'x')",
        "INSERT INTO t(a,b) VALUES (1,'x')",
        "INSERT INTO t VALUES (1,'x')",
        "INSERT INTO t(a,b) VALUES (1,'x'), (2,'y')",
        "INSERT INTO t(a) SELECT a FROM t2",
        "INSERT OR REPLACE INTO t(a,b) VALUES (1,'x')",
        "UPDATE t SET a = 1",
        "UPDATE t SET a = 1, b = b + 1 WHERE c IS NULL",
        "UPDATE OR IGNORE t SET a = 1 WHERE b = 2",
        "DELETE FROM t",
        "DELETE FROM t WHERE a = 1",
        "DELETE FROM t WHERE EXISTS (SELECT 1)",
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT, b REAL)",
        "CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY AUTOINCREMENT, a TEXT NOT NULL, b REAL DEFAULT 1.5, c BLOB, d TEXT UNIQUE)",
        "CREATE TABLE t (a INTEGER, b INTEGER, PRIMARY KEY (a, b))",
        "CREATE TABLE t (a INTEGER CHECK(a > 0), b TEXT)",
        "CREATE TABLE t (a INTEGER, b INTEGER, FOREIGN KEY (a) REFERENCES t2(id))",
        "ALTER TABLE t ADD COLUMN x INTEGER",
        "ALTER TABLE t RENAME TO t2",
        "CREATE INDEX idx_t_a ON t(a)",
        "CREATE UNIQUE INDEX idx_t_ab ON t(a, b)",
        "DROP INDEX idx_t_a",
        "DROP TABLE t",
        "DROP TABLE IF EXISTS t",
        "CREATE VIEW v AS SELECT 1 AS a",
        "DROP VIEW v",
        "BEGIN",
        "BEGIN TRANSACTION",
        "COMMIT",
        "ROLLBACK",
        "SAVEPOINT s1",
        "RELEASE s1",
        "ROLLBACK TO s1",
        "EXPLAIN SELECT 1",
        "EXPLAIN QUERY PLAN SELECT 1",
        "PRAGMA cache_size = 100",
        "PRAGMA user_version",
        "VACUUM",
        "ANALYZE",
        "REINDEX",
        "ATTACH DATABASE 'file.db' AS aux",
        "DETACH DATABASE aux",
    ]

    base += [
        'SELECT "a" FROM "t"',
        "SELECT `a` FROM `t`",
        "SELECT [a] FROM [t]",
        'CREATE TABLE "t" ("a" INTEGER, "b" TEXT)',
        "INSERT INTO `t`(`a`,`b`) VALUES (1,'x')",
        "UPDATE [t] SET [a] = 1 WHERE [b] = 'x'",
    ]

    out = []
    for s in base:
        out.extend(_variants(s))
    return _dedup_keep_order(out)


def _rand_choice(rng: random.Random, items: List[str]) -> str:
    return items[rng.randrange(len(items))]


def _random_select(rng: random.Random) -> str:
    idents = ["a", "b", "c", "id", "col1", "col2", "x", "y"]
    tables = ["t", "t1", "t2", "my_table"]
    nums = ["0", "1", "2", "3", "10", "-1", "1.5"]
    strs = ["'x'", "'y'", "'abc'", "'a''b'"]
    funcs = ["COUNT(*)", "MAX(a)", "MIN(b)", "SUM(c)", "ABS(-1)", "COALESCE(a, 1)", "NULLIF(a, b)"]

    def expr(depth: int = 0) -> str:
        if depth > 2:
            return _rand_choice(rng, idents + nums + strs + ["NULL"])
        k = rng.randrange(9)
        if k <= 3:
            return _rand_choice(rng, idents + nums + strs + ["NULL"])
        if k == 4:
            return _rand_choice(rng, funcs)
        if k == 5:
            return f"({_rand_choice(rng, idents)})"
        if k == 6:
            a = expr(depth + 1)
            b = expr(depth + 1)
            op = _rand_choice(rng, ["+", "-", "*", "/", "||"])
            return f"({a} {op} {b})"
        if k == 7:
            a = _rand_choice(rng, idents)
            return f"CAST({a} AS INTEGER)"
        return "CASE WHEN " + pred(depth + 1) + " THEN " + _rand_choice(rng, strs + nums) + " ELSE " + _rand_choice(rng, strs + nums) + " END"

    def pred(depth: int = 0) -> str:
        a = expr(depth + 1)
        b = expr(depth + 1)
        op = _rand_choice(rng, ["=", "<>", "!=", "<", "<=", ">", ">="])
        basep = f"{a} {op} {b}"
        extra = rng.randrange(10)
        if extra == 0:
            return f"{_rand_choice(rng, idents)} IS NULL"
        if extra == 1:
            return f"{_rand_choice(rng, idents)} IS NOT NULL"
        if extra == 2:
            return f"{_rand_choice(rng, idents)} BETWEEN {_rand_choice(rng, nums)} AND {_rand_choice(rng, nums)}"
        if extra == 3:
            return f"{_rand_choice(rng, idents)} IN ({_rand_choice(rng, nums)}, {_rand_choice(rng, nums)}, {_rand_choice(rng, nums)})"
        if extra == 4:
            return f"{_rand_choice(rng, idents)} LIKE {_rand_choice(rng, strs)}"
        if extra == 5:
            return f"EXISTS (SELECT 1 FROM {_rand_choice(rng, tables)})"
        if extra == 6:
            return f"{_rand_choice(rng, idents)} IN (SELECT {_rand_choice(rng, idents)} FROM {_rand_choice(rng, tables)})"
        if extra == 7 and depth <= 1:
            return f"NOT ({basep})"
        if extra == 8 and depth <= 1:
            return f"({basep}) AND ({pred(depth + 1)})"
        if extra == 9 and depth <= 1:
            return f"({basep}) OR ({pred(depth + 1)})"
        return basep

    select_items_pool = ["*", "1", "NULL", "a", "b", "c", "t.a", "t1.a", "t2.b", "COUNT(*)", "MAX(a)", "MIN(b)"]
    items = []
    for _ in range(1 + rng.randrange(4)):
        it = _rand_choice(rng, select_items_pool)
        if rng.randrange(4) == 0 and it not in ("*", "COUNT(*)"):
            it = f"{it} AS {_rand_choice(rng, ['x', 'y', 'z', 'alias'])}"
        items.append(it)
    if rng.randrange(5) == 0:
        items.append(expr(0) + " AS e")

    distinct = _rand_choice(rng, ["", "", "DISTINCT ", "ALL "])
    stmt = f"SELECT {distinct}" + ", ".join(items)

    from_kind = rng.randrange(6)
    if from_kind == 0:
        pass
    elif from_kind == 1:
        stmt += f" FROM {_rand_choice(rng, tables)}"
    elif from_kind == 2:
        stmt += f" FROM {_rand_choice(rng, tables)} AS {_rand_choice(rng, ['t', 'x', 'y'])}"
    elif from_kind == 3:
        stmt += f" FROM {_rand_choice(rng, tables)}, {_rand_choice(rng, tables)}"
    else:
        t1 = _rand_choice(rng, ["t1", "a", "x"])
        t2 = _rand_choice(rng, ["t2", "b", "y"])
        join = _rand_choice(rng, ["JOIN", "INNER JOIN", "LEFT JOIN", "LEFT OUTER JOIN", "CROSS JOIN"])
        on = "" if "CROSS" in join else f" ON {t1}.id = {t2}.id"
        stmt += f" FROM {_rand_choice(rng, tables)} {t1} {join} {_rand_choice(rng, tables)} {t2}{on}"

    if " FROM " in stmt and rng.randrange(2) == 0:
        stmt += " WHERE " + pred(0)

    if " FROM " in stmt and rng.randrange(4) == 0:
        grp = _rand_choice(rng, idents)
        stmt += f" GROUP BY {grp}"
        if rng.randrange(2) == 0:
            stmt += " HAVING COUNT(*) > 1"

    if " FROM " in stmt and rng.randrange(3) == 0:
        ob1 = _rand_choice(rng, idents)
        ob2 = _rand_choice(rng, idents)
        stmt += f" ORDER BY {ob1} {_rand_choice(rng, ['ASC', 'DESC'])}, {ob2} {_rand_choice(rng, ['ASC', 'DESC'])}"

    if rng.randrange(3) == 0:
        stmt += f" LIMIT {_rand_choice(rng, ['1', '2', '5', '10'])}"
        if rng.randrange(3) == 0:
            stmt += f" OFFSET {_rand_choice(rng, ['0', '1', '2', '5'])}"

    if rng.randrange(7) == 0:
        stmt2 = f"SELECT {_rand_choice(rng, idents + ['1'])} FROM {_rand_choice(rng, tables)}"
        setop = _rand_choice(rng, ["UNION", "UNION ALL", "INTERSECT", "EXCEPT"])
        stmt = f"{stmt} {setop} {stmt2}"

    if rng.randrange(8) == 0:
        stmt = f"WITH c AS (SELECT 1 AS a) {stmt}"

    return stmt


def _random_insert(rng: random.Random) -> str:
    tables = ["t", "t1", "my_table"]
    cols = ["a", "b", "c", "id"]
    nums = ["0", "1", "2", "3", "10", "-1", "1.5"]
    strs = ["'x'", "'y'", "'abc'", "'a''b'"]
    vals = nums + strs + ["NULL"]
    t = _rand_choice(rng, tables)

    kind = rng.randrange(6)
    if kind == 0:
        return f"INSERT INTO {t} VALUES ({_rand_choice(rng, vals)})"
    if kind == 1:
        c = _rand_choice(rng, cols)
        return f"INSERT INTO {t}({c}) VALUES ({_rand_choice(rng, vals)})"
    if kind == 2:
        c1 = _rand_choice(rng, cols)
        c2 = _rand_choice(rng, cols)
        return f"INSERT INTO {t}({c1},{c2}) VALUES ({_rand_choice(rng, vals)},{_rand_choice(rng, vals)})"
    if kind == 3:
        c1 = _rand_choice(rng, cols)
        c2 = _rand_choice(rng, cols)
        return f"INSERT INTO {t}({c1},{c2}) VALUES ({_rand_choice(rng, vals)},{_rand_choice(rng, vals)}), ({_rand_choice(rng, vals)},{_rand_choice(rng, vals)})"
    if kind == 4:
        c = _rand_choice(rng, cols)
        t2 = _rand_choice(rng, tables)
        return f"INSERT INTO {t}({c}) SELECT {c} FROM {t2}"
    return f"INSERT OR REPLACE INTO {t}(a,b) VALUES (1,'x')"


def _random_update(rng: random.Random) -> str:
    tables = ["t", "t1", "my_table"]
    cols = ["a", "b", "c", "id"]
    nums = ["0", "1", "2", "3", "10", "-1", "1.5"]
    strs = ["'x'", "'y'", "'abc'", "'a''b'"]
    vals = nums + strs + ["NULL"]
    t = _rand_choice(rng, tables)
    c1 = _rand_choice(rng, cols)
    c2 = _rand_choice(rng, cols)
    set_exprs = [
        f"{c1} = {_rand_choice(rng, vals)}",
        f"{c1} = {c1} + 1",
        f"{c1} = COALESCE({c1}, 0)",
        f"{c1} = (SELECT MAX({c1}) FROM {_rand_choice(rng, tables)})",
        f"{c1} = CAST({_rand_choice(rng, vals)} AS INTEGER)",
    ]
    stmt = f"UPDATE {t} SET " + _rand_choice(rng, set_exprs)
    if rng.randrange(2) == 0:
        stmt += ", " + f"{c2} = {_rand_choice(rng, vals)}"
    if rng.randrange(2) == 0:
        stmt += " WHERE " + _rand_choice(
            rng,
            [
                f"{c1} IS NULL",
                f"{c1} IS NOT NULL",
                f"{c1} = {_rand_choice(rng, vals)}",
                f"{c1} IN (1, 2, 3)",
                "EXISTS (SELECT 1)",
            ],
        )
    if rng.randrange(5) == 0:
        stmt = "UPDATE OR IGNORE " + stmt[len("UPDATE "):]
    return stmt


def _random_delete(rng: random.Random) -> str:
    tables = ["t", "t1", "my_table"]
    cols = ["a", "b", "c", "id"]
    t = _rand_choice(rng, tables)
    stmt = f"DELETE FROM {t}"
    if rng.randrange(2) == 0:
        c = _rand_choice(rng, cols)
        stmt += " WHERE " + _rand_choice(
            rng,
            [
                f"{c} = 1",
                f"{c} <> 1",
                f"{c} BETWEEN 1 AND 3",
                f"{c} IN (SELECT {c} FROM {_rand_choice(rng, tables)})",
                "EXISTS (SELECT 1 FROM t2)",
            ],
        )
    return stmt


def _random_create_table(rng: random.Random) -> str:
    t = _rand_choice(rng, ["t", "t1", "my_table", "t_new"])
    cols = ["a", "b", "c", "id", "x", "y"]
    types = ["INTEGER", "TEXT", "REAL", "BLOB"]
    coldefs = []
    n = 2 + rng.randrange(4)
    used = set()
    for _ in range(n):
        c = _rand_choice(rng, cols)
        while c in used:
            c = _rand_choice(rng, cols)
        used.add(c)
        ty = _rand_choice(rng, types)
        extra = []
        if rng.randrange(5) == 0:
            extra.append("NOT NULL")
        if rng.randrange(7) == 0:
            extra.append("UNIQUE")
        if rng.randrange(7) == 0:
            extra.append("PRIMARY KEY")
        if rng.randrange(7) == 0:
            extra.append("DEFAULT 1")
        if rng.randrange(9) == 0:
            extra.append("CHECK(" + c + " > 0)")
        coldefs.append(" ".join([c, ty] + extra).strip())
    if rng.randrange(4) == 0:
        pkcols = list(used)[: min(2, len(used))]
        if pkcols:
            coldefs.append("PRIMARY KEY (" + ", ".join(pkcols) + ")")
    prefix = "CREATE TABLE"
    if rng.randrange(3) == 0:
        prefix += " IF NOT EXISTS"
    return f"{prefix} {t} (" + ", ".join(coldefs) + ")"


def _random_create_index(rng: random.Random) -> str:
    idx = _rand_choice(rng, ["idx_a", "idx_ab", "idx_t_a", "idx_my"])
    t = _rand_choice(rng, ["t", "t1", "my_table"])
    cols = ["a", "b", "c", "id", "x"]
    c1 = _rand_choice(rng, cols)
    c2 = _rand_choice(rng, cols)
    unique = "UNIQUE " if rng.randrange(3) == 0 else ""
    if rng.randrange(2) == 0:
        return f"CREATE {unique}INDEX {idx} ON {t}({c1})"
    return f"CREATE {unique}INDEX {idx} ON {t}({c1}, {c2})"


def _build_random_candidates(seed: int = 0, n_select: int = 250, n_other: int = 160) -> List[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n_select):
        out.append(_random_select(rng))
    for _ in range(n_other):
        kind = rng.randrange(6)
        if kind == 0:
            out.append(_random_insert(rng))
        elif kind == 1:
            out.append(_random_update(rng))
        elif kind == 2:
            out.append(_random_delete(rng))
        elif kind == 3:
            out.append(_random_create_table(rng))
        elif kind == 4:
            out.append(_random_create_index(rng))
        else:
            out.append(_rand_choice(rng, [
                "DROP TABLE IF EXISTS t",
                "DROP INDEX idx_a",
                "ALTER TABLE t ADD COLUMN x INTEGER",
                "BEGIN",
                "COMMIT",
                "ROLLBACK",
                "SAVEPOINT s1",
                "RELEASE s1",
                "ROLLBACK TO s1",
            ]))
    final = []
    for s in out:
        final.extend(_variants(s))
    return _dedup_keep_order(final)


def _greedy_select(statements: List[str],
                   stmt_lines: Dict[str, Set[Tuple[str, int]]],
                   stmt_arcs: Dict[str, Set[Tuple[str, int, int]]],
                   total_lines: Set[Tuple[str, int]],
                   total_arcs: Set[Tuple[str, int, int]],
                   max_n: int = 40,
                   min_gain: float = 1e-5) -> List[str]:
    remaining = list(statements)
    selected: List[str] = []
    covered_lines: Set[Tuple[str, int]] = set()
    covered_arcs: Set[Tuple[str, int, int]] = set()

    current_score = _weighted_cov(covered_lines, covered_arcs, total_lines, total_arcs)

    while remaining and len(selected) < max_n:
        best_stmt = None
        best_score = current_score
        best_gain = 0.0

        for s in remaining:
            nl = covered_lines | stmt_lines.get(s, set())
            na = covered_arcs | stmt_arcs.get(s, set())
            sc = _weighted_cov(nl, na, total_lines, total_arcs)
            gain = sc - current_score
            if gain > best_gain + 1e-15:
                best_gain = gain
                best_score = sc
                best_stmt = s

        if best_stmt is None or best_gain < min_gain:
            break

        selected.append(best_stmt)
        covered_lines |= stmt_lines.get(best_stmt, set())
        covered_arcs |= stmt_arcs.get(best_stmt, set())
        current_score = best_score
        remaining.remove(best_stmt)

    return selected


def _prune_redundant(selected: List[str],
                     stmt_lines: Dict[str, Set[Tuple[str, int]]],
                     stmt_arcs: Dict[str, Set[Tuple[str, int, int]]]) -> List[str]:
    if not selected:
        return selected

    line_counts = Counter()
    arc_counts = Counter()
    for s in selected:
        for e in stmt_lines.get(s, set()):
            line_counts[e] += 1
        for e in stmt_arcs.get(s, set()):
            arc_counts[e] += 1

    pruned = []
    changed = True
    keep = selected[:]
    while changed:
        changed = False
        new_keep = []
        for s in keep:
            lset = stmt_lines.get(s, set())
            aset = stmt_arcs.get(s, set())
            unique_line = any(line_counts[e] == 1 for e in lset) if lset else False
            unique_arc = any(arc_counts[e] == 1 for e in aset) if aset else False
            if unique_line or unique_arc:
                new_keep.append(s)
                continue
            # remove redundant
            for e in lset:
                line_counts[e] -= 1
                if line_counts[e] <= 0:
                    del line_counts[e]
            for e in aset:
                arc_counts[e] -= 1
                if arc_counts[e] <= 0:
                    del arc_counts[e]
            changed = True
        keep = new_keep
    pruned = keep
    return pruned


class Solution:
    def solve(self, resources_path: str) -> list[str]:
        parse_sql, target_files, parser_mod, tokenizer_mod, ast_nodes_mod = _import_sql_engine(resources_path)

        handcrafted = _build_handcrafted_candidates()
        random_cands = _build_random_candidates(seed=12345, n_select=260, n_other=180)

        code_keywords = set()
        try:
            code_keywords |= _extract_upper_keywords_from_code(_safe_read_text(_canon(parser_mod.__file__)))
            code_keywords |= _extract_upper_keywords_from_code(_safe_read_text(_canon(tokenizer_mod.__file__)))
        except Exception:
            pass

        extra = []
        if "TRUNCATE" in code_keywords:
            extra += ["TRUNCATE TABLE t", "TRUNCATE t"]
        if "RENAME" in code_keywords:
            extra += ["ALTER TABLE t RENAME COLUMN a TO b"]
        if "CREATE" in code_keywords and "TRIGGER" in code_keywords:
            extra += [
                "CREATE TRIGGER tr AFTER INSERT ON t BEGIN SELECT 1; END",
                "DROP TRIGGER tr",
            ]
        if "CREATE" in code_keywords and "VIRTUAL" in code_keywords:
            extra += ["CREATE VIRTUAL TABLE t USING fts5(a)"]

        extra_variants = []
        for s in extra:
            extra_variants.extend(_variants(s))

        candidates = _dedup_keep_order(handcrafted + random_cands + extra_variants)

        valid = []
        seen_norm = set()
        for s in candidates:
            n = _normalize_for_set(s)
            if n in seen_norm:
                continue
            if len(s) > 8000:
                continue
            if _try_parse(parse_sql, s):
                valid.append(s)
                seen_norm.add(n)
            if len(valid) >= 220:
                break

        if not valid:
            for fallback in ["SELECT 1", "SELECT 1;"]:
                if _try_parse(parse_sql, fallback):
                    return [fallback]
            return ["SELECT 1"]

        if coverage is None or PythonParser is None:  # pragma: no cover
            return valid[:35]

        total_lines, total_arcs = _get_totals([_canon(f) for f in target_files])

        stmt_lines: Dict[str, Set[Tuple[str, int]]] = {}
        stmt_arcs: Dict[str, Set[Tuple[str, int, int]]] = {}

        # Measure a limited set, prioritizing longer (often more complex) statements
        valid_sorted = sorted(valid, key=lambda x: (-len(x), x))[:200]

        for s in valid_sorted:
            lns, arcs = _measure_stmt_coverage(parse_sql, s, target_files)
            stmt_lines[s] = lns
            stmt_arcs[s] = arcs

        selected = _greedy_select(valid_sorted, stmt_lines, stmt_arcs, total_lines, total_arcs, max_n=40, min_gain=1e-6)
        selected = _prune_redundant(selected, stmt_lines, stmt_arcs)

        # If pruning removed too much, keep a small base set
        if not selected:
            selected = valid_sorted[:10]

        # Ensure at least one comment-containing stmt if valid and available (tokenizer coverage)
        if not any(("/*" in s) or ("--" in s) for s in selected):
            for s in valid_sorted:
                if ("/*" in s) or ("--" in s):
                    selected.append(s)
                    break

        # Final sanity: all returned must parse
        final = []
        for s in selected:
            if _try_parse(parse_sql, s):
                final.append(s)
        if not final:
            final = [valid_sorted[0]]

        return final[:45]