import os
import re
import ast
import textwrap
from typing import Dict, List, Set, Tuple, Optional


def _extract_py_string_constants(py_path: str) -> List[str]:
    try:
        with open(py_path, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception:
        return []
    out: List[str] = []
    try:
        tree = ast.parse(src, filename=py_path)
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                out.append(n.value)
    except Exception:
        # Fallback: crude regex for quoted strings
        for m in re.finditer(r"""(?s)(?:'([^'\\]*(?:\\.[^'\\]*)*)'|"([^"\\]*(?:\\.[^"\\]*)*)")""", src):
            s = m.group(1) if m.group(1) is not None else m.group(2)
            if s is not None:
                out.append(s)
    return out


def _extract_grammar_terminals(grammar_path: str, limit: int = 2000) -> List[str]:
    try:
        with open(grammar_path, "r", encoding="utf-8") as f:
            txt = f.read()
    except Exception:
        return []
    # capture 'TOKEN' or "TOKEN"
    terms: List[str] = []
    for m in re.finditer(r"""(?s)(?:'([^'\\]{1,64})'|"([^"\\]{1,64})")""", txt):
        s = m.group(1) if m.group(1) is not None else m.group(2)
        if not s:
            continue
        s2 = s.strip()
        if not s2:
            continue
        terms.append(s2)
        if len(terms) >= limit:
            break
    return terms


def _normalize_keywords(strings: List[str]) -> Tuple[List[str], List[str], List[str]]:
    kw: Set[str] = set()
    funcs: Set[str] = set()
    types: Set[str] = set()

    common_funcs = {
        "COUNT", "SUM", "MIN", "MAX", "AVG", "ABS", "ROUND", "LOWER", "UPPER", "LENGTH",
        "SUBSTR", "SUBSTRING", "TRIM", "LTRIM", "RTRIM", "COALESCE", "NULLIF", "IFNULL",
        "CAST", "DATE", "TIME", "DATETIME", "STRFTIME", "RANDOM",
    }
    common_types = {
        "INT", "INTEGER", "TINYINT", "SMALLINT", "BIGINT",
        "REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL",
        "TEXT", "CHAR", "VARCHAR", "NCHAR", "NVARCHAR", "BLOB", "BOOLEAN", "DATE", "TIME", "TIMESTAMP",
    }
    common_keywords = {
        "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER",
        "TABLE", "INDEX", "VIEW", "TRIGGER",
        "FROM", "WHERE", "GROUP", "BY", "HAVING", "ORDER", "LIMIT", "OFFSET",
        "DISTINCT", "ALL", "AS", "INTO", "VALUES", "SET",
        "JOIN", "LEFT", "RIGHT", "FULL", "OUTER", "INNER", "CROSS", "NATURAL", "ON", "USING",
        "UNION", "INTERSECT", "EXCEPT",
        "WITH", "RECURSIVE",
        "NULL", "TRUE", "FALSE",
        "AND", "OR", "NOT",
        "IN", "IS", "LIKE", "BETWEEN", "EXISTS",
        "CASE", "WHEN", "THEN", "ELSE", "END",
        "PRIMARY", "KEY", "FOREIGN", "REFERENCES", "UNIQUE", "CHECK", "DEFAULT", "CONSTRAINT",
        "ASC", "DESC",
        "BEGIN", "COMMIT", "ROLLBACK", "TRANSACTION",
        "IF", "EXISTS", "TEMP", "TEMPORARY",
        "RENAME", "ADD", "COLUMN",
    }

    rx_word = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    for s in strings:
        if not s or len(s) > 64:
            continue
        t = s.strip()
        if not t or len(t) > 64:
            continue
        if rx_word.match(t):
            u = t.upper()
            if u in common_types or u.endswith("INT") or u in {"TEXT", "BLOB", "REAL", "NUMERIC"}:
                types.add(u)
            if u in common_funcs or u in {"CAST", "COALESCE", "NULLIF", "IFNULL"}:
                funcs.add(u)
            if len(u) >= 2:
                kw.add(u)

    kw |= common_keywords
    funcs |= common_funcs
    types |= common_types

    # Keep lists stable-ish for deterministic behavior
    kw_list = sorted(kw)
    func_list = sorted(funcs)
    type_list = sorted(types)
    # Trim oversized lists (code size / overhead)
    if len(kw_list) > 600:
        kw_list = kw_list[:600]
    if len(func_list) > 200:
        func_list = func_list[:200]
    if len(type_list) > 200:
        type_list = type_list[:200]
    return kw_list, func_list, type_list


class Solution:
    def solve(self, resources_path: str) -> dict:
        parser_py = os.path.join(resources_path, "sql_engine", "parser.py")
        tokenizer_py = os.path.join(resources_path, "sql_engine", "tokenizer.py")
        ast_nodes_py = os.path.join(resources_path, "sql_engine", "ast_nodes.py")
        grammar_txt = os.path.join(resources_path, "sql_grammar.txt")

        all_strings: List[str] = []
        for p in (parser_py, tokenizer_py, ast_nodes_py):
            all_strings.extend(_extract_py_string_constants(p))
        all_strings.extend(_extract_grammar_terminals(grammar_txt))

        kw_list, func_list, type_list = _normalize_keywords(all_strings)

        # Some operators / punctuation that tend to exercise tokenizer/parser branches
        op_list = [
            "+", "-", "*", "/", "%", "||",
            "=", "!=", "<>", "<", ">", "<=", ">=",
            "&", "|", "^", "~",
        ]

        # Seed statements (curated)
        seed_statements = [
            "SELECT 1",
            "SELECT 1+2*3",
            "SELECT NULL",
            "SELECT TRUE",
            "SELECT FALSE",
            "SELECT 'a'",
            "SELECT ''",
            "SELECT 'a''b'",
            "SELECT \"a\"",
            "SELECT `a`",
            "SELECT [a]",
            "SELECT * FROM t",
            "SELECT a FROM t",
            "SELECT a,b,c FROM t",
            "SELECT a AS x FROM t",
            "SELECT DISTINCT a FROM t",
            "SELECT a,b FROM t WHERE a=1",
            "SELECT a FROM t WHERE a<>1",
            "SELECT a FROM t WHERE a!=1",
            "SELECT a FROM t WHERE a<1 AND b>2",
            "SELECT a FROM t WHERE a<=1 OR b>=2",
            "SELECT a FROM t WHERE a IS NULL",
            "SELECT a FROM t WHERE a IS NOT NULL",
            "SELECT a FROM t WHERE a IN (1,2,3)",
            "SELECT a FROM t WHERE a NOT IN (1,2,3)",
            "SELECT a FROM t WHERE a BETWEEN 1 AND 3",
            "SELECT a FROM t WHERE a LIKE 'a%'",
            "SELECT a FROM t WHERE EXISTS (SELECT 1)",
            "SELECT a FROM t ORDER BY a",
            "SELECT a FROM t ORDER BY a DESC",
            "SELECT a FROM t ORDER BY a ASC, b DESC",
            "SELECT a FROM t GROUP BY a",
            "SELECT a, COUNT(*) FROM t GROUP BY a",
            "SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(*)>1",
            "SELECT a FROM t LIMIT 1",
            "SELECT a FROM t LIMIT 1 OFFSET 2",
            "SELECT a FROM t OFFSET 2",
            "SELECT a FROM t JOIN u ON t.id=u.id",
            "SELECT a FROM t LEFT JOIN u ON t.id=u.id",
            "SELECT a FROM t RIGHT JOIN u ON t.id=u.id",
            "SELECT a FROM t FULL OUTER JOIN u ON t.id=u.id",
            "SELECT a FROM t CROSS JOIN u",
            "SELECT a FROM t NATURAL JOIN u",
            "SELECT a FROM t JOIN u USING (id)",
            "SELECT a FROM (SELECT 1 AS a) x",
            "SELECT a FROM t WHERE (a=1 AND (b=2 OR c=3))",
            "SELECT CASE WHEN 1=1 THEN 2 ELSE 3 END",
            "SELECT CAST(1 AS INTEGER)",
            "SELECT COALESCE(NULL, 1)",
            "SELECT NULLIF(1,1)",
            "WITH x AS (SELECT 1 AS a) SELECT a FROM x",
            "WITH RECURSIVE x(a) AS (SELECT 1 UNION ALL SELECT a+1 FROM x) SELECT a FROM x LIMIT 3",
            "SELECT 1 UNION SELECT 2",
            "SELECT 1 UNION ALL SELECT 2",
            "SELECT 1 INTERSECT SELECT 1",
            "SELECT 1 EXCEPT SELECT 1",
            "INSERT INTO t VALUES (1)",
            "INSERT INTO t VALUES (1,'a',NULL)",
            "INSERT INTO t(a,b) VALUES (1,2)",
            "INSERT INTO t(a) SELECT 1",
            "UPDATE t SET a=1",
            "UPDATE t SET a=1, b=b+1 WHERE id=1",
            "DELETE FROM t",
            "DELETE FROM t WHERE id=1",
            "CREATE TABLE t (id INTEGER)",
            "CREATE TABLE t (id INTEGER PRIMARY KEY)",
            "CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT NOT NULL, b REAL DEFAULT 1.0)",
            "CREATE TABLE t (id INTEGER, a TEXT, UNIQUE(a))",
            "CREATE TABLE t (id INTEGER, a INTEGER, CHECK(a>0))",
            "CREATE TABLE t (id INTEGER, a INTEGER, FOREIGN KEY(a) REFERENCES u(id))",
            "DROP TABLE t",
            "DROP TABLE IF EXISTS t",
            "ALTER TABLE t ADD COLUMN a INTEGER",
            "ALTER TABLE t RENAME TO t2",
            "CREATE INDEX idx ON t(a)",
            "BEGIN",
            "BEGIN TRANSACTION",
            "COMMIT",
            "ROLLBACK",
            "-- comment\nSELECT 1",
            "/* comment */ SELECT 1",
            "SELECT 1 /* trailing */",
            "SELECT 0xFF",
            "SELECT 1e10",
            "SELECT 1.0",
            "SELECT .5",
            "SELECT -1",
            "SELECT +1",
            "SELECT 1/0",
            "SELECT (1)",
            "SELECT (((1)))",
            "SELECT a FROM t WHERE a=1;",

            # intentionally malformed / odd
            "SELECT",
            "INSERT",
            "CREATE",
            "SELECT * FROM",
            "SELECT FROM t",
            "SELECT 1,,2",
            "SELECT (",
            "SELECT )",
            "SELECT '",
            "SELECT \"",
            "SELECT /* unterminated",
            "SELECT -- unterminated",
            "SELECT 1e",
            "SELECT 0x",
            "SELECT `",
            "SELECT [",
        ]

        # Add a few keyword-only probes to hit tokenizer / dispatch error paths
        for k in kw_list[:120]:
            if len(k) <= 20 and k.isupper() and k.isalpha():
                seed_statements.append(k)
                seed_statements.append(k + ";")

        code = f"""
import time
import random
import re

KW = {kw_list!r}
FUNCS = {func_list!r}
TYPES = {type_list!r}
OPS = {op_list!r}

SEEDS = {seed_statements!r}

_state = None

_rx_num = re.compile(r"\\b(?:0x[0-9A-Fa-f]+|\\d+\\.\\d+|\\d+\\.|\\.\\d+|\\d+)(?:[eE][+-]?\\d+)?\\b")
_rx_ident = re.compile(r"\\b[a-zA-Z_][a-zA-Z0-9_]*\\b")
_rx_str = re.compile(r"'(?:''|[^'])*'")

def _rand_case(rng, s: str) -> str:
    mode = rng.randrange(5)
    if mode == 0:
        return s
    if mode == 1:
        return s.lower()
    if mode == 2:
        return s.upper()
    # mixed
    out = []
    for ch in s:
        if 'a' <= ch <= 'z' or 'A' <= ch <= 'Z':
            out.append(ch.upper() if rng.getrandbits(1) else ch.lower())
        else:
            out.append(ch)
    return ''.join(out)

def _ident(rng) -> str:
    base = rng.choice([
        "t","u","v","w","users","orders","items","products","log","x","y","z",
        "col","col1","col2","id","name","value","amount","price","qty","created_at","updated_at",
        "select","from","where","group","order"
    ])
    if rng.random() < 0.55:
        base = base + str(rng.randrange(0, 50))
    style = rng.randrange(6)
    if style == 0:
        return base
    if style == 1:
        return base.upper()
    if style == 2:
        return '"' + base.replace('"', '""') + '"'
    if style == 3:
        return '`' + base.replace('`', '``') + '`'
    if style == 4:
        return '[' + base.replace(']', ']]') + ']'
    return base

def _type_name(rng) -> str:
    t = rng.choice(TYPES) if TYPES else rng.choice(["INTEGER","TEXT","REAL","NUMERIC"])
    if t in ("CHAR","VARCHAR","NCHAR","NVARCHAR","DECIMAL","NUMERIC") and rng.random() < 0.6:
        if rng.random() < 0.4:
            return f"{{t}}({{rng.randrange(1, 40)}})"
        return f"{{t}}({{rng.randrange(1, 20)}},{{rng.randrange(0, 10)}})"
    return t

def _num(rng) -> str:
    r = rng.random()
    if r < 0.25:
        return str(rng.randrange(-10, 2000))
    if r < 0.45:
        return str(rng.randrange(0, 999)) + "." + str(rng.randrange(0, 999))
    if r < 0.65:
        return "." + str(rng.randrange(0, 999))
    if r < 0.8:
        return str(rng.randrange(1, 999)) + "e" + str(rng.randrange(-30, 30))
    if r < 0.92:
        return "0x" + format(rng.randrange(0, 1<<32), "x")
    return str(rng.randrange(0, 999999999))

def _str_lit(rng) -> str:
    choices = [
        "a", "b", "test", "hello", "world", "O'Reilly", "x\\ny", "tab\\tsep",
        "", " ", "''", "/*c*/", "--c", "NaN", "∞", "汉字", "\\\\", "\\0"
    ]
    s = rng.choice(choices)
    # represent with single quotes SQL-style (double single-quotes to escape)
    s2 = s.replace("'", "''")
    return "'" + s2 + "'"

def _literal(rng) -> str:
    r = rng.random()
    if r < 0.15:
        return "NULL"
    if r < 0.25:
        return "TRUE"
    if r < 0.35:
        return "FALSE"
    if r < 0.65:
        return _num(rng)
    return _str_lit(rng)

def _func_call(rng, depth: int) -> str:
    fn = rng.choice(FUNCS) if FUNCS else rng.choice(["COUNT","SUM","LOWER","UPPER","COALESCE","NULLIF","CAST"])
    fn_u = fn.upper()
    if fn_u == "COUNT" and rng.random() < 0.5:
        arg = "*" if rng.random() < 0.5 else _expr(rng, depth-1)
        return f"COUNT({{arg}})"
    if fn_u == "CAST":
        return f"CAST({{_expr(rng, depth-1)}} AS {{_type_name(rng)}})"
    if fn_u in ("COALESCE","NULLIF","IFNULL"):
        a = _expr(rng, depth-1)
        b = _expr(rng, depth-1)
        if fn_u == "COALESCE" and rng.random() < 0.4:
            c = _expr(rng, depth-1)
            return f"COALESCE({{a}},{{b}},{{c}})"
        return f"{{fn_u}}({{a}},{{b}})"
    argc = 1 if rng.random() < 0.6 else 2
    args = [_expr(rng, depth-1) for _ in range(argc)]
    return f"{{fn_u}}(" + ",".join(args) + ")"

def _colref(rng) -> str:
    if rng.random() < 0.35:
        return _ident(rng) + "." + _ident(rng)
    return _ident(rng)

def _case_expr(rng, depth: int) -> str:
    parts = ["CASE"]
    n = 1 + (1 if rng.random() < 0.35 else 0)
    for _ in range(n):
        parts.append("WHEN")
        parts.append(_expr(rng, depth-1))
        parts.append("THEN")
        parts.append(_expr(rng, depth-1))
    if rng.random() < 0.7:
        parts.append("ELSE")
        parts.append(_expr(rng, depth-1))
    parts.append("END")
    return " ".join(parts)

def _expr(rng, depth: int) -> str:
    if depth <= 0:
        r = rng.random()
        if r < 0.4:
            return _literal(rng)
        if r < 0.75:
            return _colref(rng)
        return "(" + _literal(rng) + ")"

    r = rng.random()
    if r < 0.14:
        return _literal(rng)
    if r < 0.28:
        return _colref(rng)
    if r < 0.40:
        return "(" + _expr(rng, depth-1) + ")"
    if r < 0.52:
        return _func_call(rng, depth)
    if r < 0.62:
        op = rng.choice(["NOT", "+", "-", "~"])
        if op == "NOT":
            return "NOT " + _expr(rng, depth-1)
        return op + _expr(rng, depth-1)
    if r < 0.75:
        a = _expr(rng, depth-1)
        b = _expr(rng, depth-1)
        op = rng.choice(OPS + ["AND","OR"])
        if op in ("AND","OR"):
            return a + " " + op + " " + b
        return a + op + b
    if r < 0.86:
        a = _expr(rng, depth-1)
        b = _expr(rng, depth-1)
        c = _expr(rng, depth-1)
        return a + " BETWEEN " + b + " AND " + c
    if r < 0.94:
        a = _expr(rng, depth-1)
        if rng.random() < 0.6:
            # IN list
            n = 1 + (rng.randrange(1, 5))
            lst = ",".join(_expr(rng, depth-1) for _ in range(n))
            neg = " NOT" if rng.random() < 0.35 else ""
            return a + neg + " IN (" + lst + ")"
        # IN (subquery)
        neg = " NOT" if rng.random() < 0.35 else ""
        return a + neg + " IN (" + _select_stmt(rng, depth-1, as_subquery=True) + ")"
    # EXISTS subquery / LIKE / IS NULL
    kind = rng.randrange(3)
    if kind == 0:
        return "EXISTS (" + _select_stmt(rng, depth-1, as_subquery=True) + ")"
    if kind == 1:
        return _expr(rng, depth-1) + " LIKE " + _str_lit(rng)
    return _expr(rng, depth-1) + (" IS NOT NULL" if rng.random() < 0.5 else " IS NULL")

def _select_list(rng, depth: int) -> str:
    if rng.random() < 0.15:
        return "*"
    n = 1 + rng.randrange(1, 5)
    cols = []
    for _ in range(n):
        e = _expr(rng, depth-1)
        if rng.random() < 0.3:
            e = e + " AS " + _ident(rng)
        cols.append(e)
    return ", ".join(cols)

def _from_clause(rng, depth: int) -> str:
    if rng.random() < 0.25:
        return ""
    if rng.random() < 0.3 and depth > 0:
        base = "(" + _select_stmt(rng, depth-1, as_subquery=True) + ") " + _ident(rng)
    else:
        base = _ident(rng)
        if rng.random() < 0.25:
            base = base + " " + _ident(rng)
    parts = ["FROM", base]
    # joins
    if rng.random() < 0.55:
        jn = 1 + rng.randrange(0, 3)
        for _ in range(jn):
            jt = rng.choice(["JOIN","LEFT JOIN","LEFT OUTER JOIN","INNER JOIN","CROSS JOIN","NATURAL JOIN"])
            tbl = _ident(rng)
            if rng.random() < 0.25:
                tbl = tbl + " " + _ident(rng)
            if "NATURAL" in jt or "CROSS" in jt:
                parts.append(jt)
                parts.append(tbl)
                continue
            parts.append(jt)
            parts.append(tbl)
            if rng.random() < 0.25:
                parts.append("USING (" + _ident(rng) + ")")
            else:
                parts.append("ON " + _colref(rng) + "=" + _colref(rng))
    return " " + " ".join(parts)

def _where_clause(rng, depth: int) -> str:
    if rng.random() < 0.55:
        return ""
    return " WHERE " + _expr(rng, depth-1)

def _group_having(rng, depth: int) -> str:
    if rng.random() < 0.7:
        return ""
    n = 1 + rng.randrange(0, 3)
    cols = ", ".join(_colref(rng) for _ in range(n))
    s = " GROUP BY " + cols
    if rng.random() < 0.5:
        s += " HAVING " + _expr(rng, depth-1)
    return s

def _order_limit(rng, depth: int) -> str:
    out = []
    if rng.random() < 0.6:
        n = 1 + rng.randrange(0, 3)
        items = []
        for _ in range(n):
            it = _colref(rng) if rng.random() < 0.7 else _expr(rng, depth-1)
            if rng.random() < 0.5:
                it += " " + rng.choice(["ASC","DESC"])
            items.append(it)
        out.append(" ORDER BY " + ", ".join(items))
    if rng.random() < 0.65:
        if rng.random() < 0.2:
            out.append(" OFFSET " + _num(rng))
        else:
            out.append(" LIMIT " + _num(rng))
            if rng.random() < 0.5:
                out.append(" OFFSET " + _num(rng))
    return "".join(out)

def _select_stmt(rng, depth: int, as_subquery: bool = False) -> str:
    parts = []
    if not as_subquery and rng.random() < 0.25:
        # WITH clause
        cten = 1 + rng.randrange(0, 2)
        rec = "RECURSIVE " if rng.random() < 0.25 else ""
        ctes = []
        for _ in range(cten):
            name = _ident(rng)
            if rng.random() < 0.35:
                cols = "(" + ",".join(_ident(rng) for _ in range(1 + rng.randrange(0, 3))) + ")"
            else:
                cols = ""
            ctes.append(name + cols + " AS (" + _select_stmt(rng, depth-1, as_subquery=True) + ")")
        parts.append("WITH " + rec + ", ".join(ctes) + " ")
    parts.append("SELECT ")
    if rng.random() < 0.2:
        parts.append(rng.choice(["DISTINCT ", "ALL "]))
    parts.append(_select_list(rng, depth))
    parts.append(_from_clause(rng, depth))
    parts.append(_where_clause(rng, depth))
    parts.append(_group_having(rng, depth))
    parts.append(_order_limit(rng, depth))
    s = "".join(parts)
    if rng.random() < 0.2 and depth > 0:
        # compound
        op = rng.choice([" UNION ", " UNION ALL ", " INTERSECT ", " EXCEPT "])
        s = s + op + _select_stmt(rng, depth-1, as_subquery=True)
    return s

def _insert_stmt(rng, depth: int) -> str:
    t = _ident(rng)
    cols = ""
    if rng.random() < 0.55:
        cols = "(" + ",".join(_ident(rng) for _ in range(1 + rng.randrange(0, 4))) + ")"
    if rng.random() < 0.6:
        # VALUES
        rows = 1 + rng.randrange(0, 3)
        row_txt = []
        for _ in range(rows):
            n = 1 + rng.randrange(0, 4)
            row_txt.append("(" + ",".join(_expr(rng, depth-1) for _ in range(n)) + ")")
        return "INSERT INTO " + t + cols + " VALUES " + ", ".join(row_txt)
    # INSERT ... SELECT
    return "INSERT INTO " + t + cols + " " + _select_stmt(rng, depth-1, as_subquery=True)

def _update_stmt(rng, depth: int) -> str:
    t = _ident(rng)
    n = 1 + rng.randrange(0, 4)
    assigns = []
    for _ in range(n):
        assigns.append(_ident(rng) + "=" + _expr(rng, depth-1))
    s = "UPDATE " + t + " SET " + ", ".join(assigns)
    if rng.random() < 0.7:
        s += " WHERE " + _expr(rng, depth-1)
    return s

def _delete_stmt(rng, depth: int) -> str:
    t = _ident(rng)
    s = "DELETE FROM " + t
    if rng.random() < 0.7:
        s += " WHERE " + _expr(rng, depth-1)
    return s

def _create_table_stmt(rng, depth: int) -> str:
    t = _ident(rng)
    ncols = 1 + rng.randrange(0, 6)
    cols = []
    for i in range(ncols):
        col = _ident(rng) + " " + _type_name(rng)
        if rng.random() < 0.25:
            col += " NOT NULL"
        if rng.random() < 0.18:
            col += " UNIQUE"
        if rng.random() < 0.16:
            col += " DEFAULT " + (_literal(rng) if rng.random() < 0.7 else _expr(rng, depth-1))
        if rng.random() < 0.12:
            col += " PRIMARY KEY"
        cols.append(col)
    tbl_constraints = []
    if rng.random() < 0.22:
        tbl_constraints.append("PRIMARY KEY (" + _ident(rng) + ")")
    if rng.random() < 0.18:
        tbl_constraints.append("UNIQUE (" + _ident(rng) + ")")
    if rng.random() < 0.20:
        tbl_constraints.append("CHECK (" + _expr(rng, depth-1) + ")")
    if rng.random() < 0.12:
        tbl_constraints.append("FOREIGN KEY (" + _ident(rng) + ") REFERENCES " + _ident(rng) + "(" + _ident(rng) + ")")
    all_defs = cols + tbl_constraints
    prefix = "CREATE "
    if rng.random() < 0.2:
        prefix += rng.choice(["TEMP ", "TEMPORARY "])
    prefix += "TABLE "
    if rng.random() < 0.25:
        prefix += "IF NOT EXISTS "
    return prefix + t + " (" + ", ".join(all_defs) + ")"

def _drop_stmt(rng) -> str:
    what = rng.choice(["TABLE", "INDEX", "VIEW", "TRIGGER"])
    s = "DROP " + what + " "
    if rng.random() < 0.35:
        s += "IF EXISTS "
    s += _ident(rng)
    return s

def _alter_stmt(rng) -> str:
    t = _ident(rng)
    if rng.random() < 0.5:
        return "ALTER TABLE " + t + " ADD COLUMN " + _ident(rng) + " " + _type_name(rng)
    return "ALTER TABLE " + t + " RENAME TO " + _ident(rng)

def _txn_stmt(rng) -> str:
    return rng.choice(["BEGIN", "BEGIN TRANSACTION", "COMMIT", "ROLLBACK"])

def _malformed(rng) -> str:
    # Build weird token soup (still exercises tokenizer/error branches)
    toks = []
    n = 2 + rng.randrange(0, 18)
    for _ in range(n):
        r = rng.random()
        if r < 0.35:
            toks.append(rng.choice(KW) if KW else rng.choice(["SELECT","FROM","WHERE","JOIN"]))
        elif r < 0.55:
            toks.append(_ident(rng))
        elif r < 0.75:
            toks.append(_literal(rng))
        else:
            toks.append(rng.choice(OPS + ["(",")",",",";"]))
    s = " ".join(toks)
    if rng.random() < 0.35:
        s = s.replace(" ", "")
    if rng.random() < 0.5:
        s += rng.choice(["", ";", ";;", ")"])
    return s

def _decorate(rng, s: str) -> str:
    if rng.random() < 0.35:
        s = _rand_case(rng, s)
    if rng.random() < 0.20:
        s = ("--" + _ident(rng) + "\\n") + s
    if rng.random() < 0.20:
        s = ("/*" + _ident(rng) + "*/ ") + s
    if rng.random() < 0.40 and not s.rstrip().endswith(";"):
        s += ";"
    if rng.random() < 0.15:
        s = "  \\t\\n" + s + "\\n\\t  "
    return s

def _mutate(rng, s: str) -> str:
    if not s:
        return s
    r = rng.random()
    if r < 0.25:
        # replace a number
        return _rx_num.sub(lambda m: _num(rng), s, count=1)
    if r < 0.45:
        # replace string literal
        if _rx_str.search(s):
            return _rx_str.sub(lambda m: _str_lit(rng), s, count=1)
        return s + " " + _str_lit(rng)
    if r < 0.62:
        # replace identifier
        return _rx_ident.sub(lambda m: _ident(rng), s, count=1)
    if r < 0.78:
        # insert keyword/operator at random spot
        insert = rng.choice(KW) if (KW and rng.random() < 0.7) else rng.choice(OPS + ["(",")",","])
        pos = rng.randrange(0, len(s)+1)
        return s[:pos] + " " + insert + " " + s[pos:]
    if r < 0.90:
        # delete a slice
        if len(s) < 6:
            return s
        a = rng.randrange(0, len(s)-1)
        b = rng.randrange(a+1, min(len(s), a + 1 + rng.randrange(1, 20)))
        return s[:a] + s[b:]
    # duplicate a clause-ish
    frag = rng.choice([" WHERE ", " FROM ", " ORDER BY ", " GROUP BY ", " LIMIT ", " JOIN ", " VALUES "])
    return s + frag + _ident(rng)

def fuzz(parse_sql):
    global _state
    if _state is None:
        _state = {{
            "t0": time.perf_counter(),
            "calls": 0,
            "target_sec": 3.8,
            "batch": 900,
            "seed_i": 0,
            "corpus": [],
            "rng": random.Random(0xC0FFEE),
            "max_calls": 26,
        }}

    st = _state
    rng = st["rng"]

    now = time.perf_counter()
    if now - st["t0"] > 59.0:
        return False
    if st["calls"] >= st["max_calls"]:
        return False

    batch = int(st["batch"])
    if batch < 250:
        batch = 250
    if batch > 6000:
        batch = 6000

    stmts = []
    seen = set()

    # Always include some seeds early, then taper
    seed_chunk = 0
    if st["seed_i"] < len(SEEDS):
        if st["calls"] == 0:
            seed_chunk = min(500, len(SEEDS) - st["seed_i"])
        elif st["calls"] < 4:
            seed_chunk = min(200, len(SEEDS) - st["seed_i"])
        else:
            seed_chunk = min(60, len(SEEDS) - st["seed_i"])

    for _ in range(seed_chunk):
        s = SEEDS[st["seed_i"]]
        st["seed_i"] += 1
        s = _decorate(rng, s)
        if s not in seen:
            seen.add(s)
            stmts.append(s)

    # Fill remainder with generated/mutated/malformed
    while len(stmts) < batch:
        r = rng.random()
        if r < 0.08:
            s = _malformed(rng)
        elif r < 0.56:
            k = rng.randrange(6)
            if k == 0:
                s = _select_stmt(rng, 4, as_subquery=False)
            elif k == 1:
                s = _insert_stmt(rng, 4)
            elif k == 2:
                s = _update_stmt(rng, 4)
            elif k == 3:
                s = _delete_stmt(rng, 4)
            elif k == 4:
                s = _create_table_stmt(rng, 4)
            else:
                s = _drop_stmt(rng) if rng.random() < 0.6 else _alter_stmt(rng)
        else:
            # mutate existing
            if st["corpus"]:
                base = st["corpus"][rng.randrange(0, len(st["corpus"]))]
                s = _mutate(rng, base)
            else:
                s = _select_stmt(rng, 3, as_subquery=False)

        s = _decorate(rng, s)
        if s in seen:
            continue
        seen.add(s)
        stmts.append(s)

    # one parse_sql call per fuzz() invocation for efficiency
    t1 = time.perf_counter()
    parse_sql(stmts)
    dt = time.perf_counter() - t1

    # update corpus (keep small)
    if len(st["corpus"]) < 2500:
        st["corpus"].extend(stmts[: min(len(stmts), 200)])
    else:
        # random replacement
        for s in stmts[:120]:
            st["corpus"][rng.randrange(0, len(st["corpus"]))] = s

    st["calls"] += 1

    # adjust batch size toward target call time
    if dt > 0.05:
        scale = st["target_sec"] / dt
        # conservative updates
        if scale > 1.25:
            scale = 1.25
        elif scale < 0.80:
            scale = 0.80
        st["batch"] = int(max(250, min(6000, int(st["batch"] * scale))))

    # stop early if near time budget
    if time.perf_counter() - st["t0"] > 59.2:
        return False
    return True
"""
        return {"code": textwrap.dedent(code).lstrip()}