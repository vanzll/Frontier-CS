import os
import random
import time
import re
import string
import itertools

class Solution:
    def solve(self, resources_path: str) -> dict:
        code = """
import random
import time
import re
import string
import itertools

# Global state to minimize parse_sql invocations
_FUZZ_DONE = False

def _rng():
    return random.Random(time.time_ns() ^ (hash(time.time()) & 0xFFFFFFFF))

def _choose(rnd, seq):
    return seq[rnd.randrange(len(seq))]

def _maybe(rnd, p=0.5):
    return rnd.random() < p

def _rand_int(rnd, a=0, b=1000):
    return rnd.randint(a, b)

def _rand_float(rnd):
    # Diverse numeric formats, including extremes
    forms = []
    forms.append(str(rnd.randint(-2**31, 2**31-1)))
    forms.append(str(rnd.uniform(-1e6, 1e6)))
    e = rnd.randint(-308, 308)
    mant = rnd.uniform(-10, 10)
    forms.append(f"{mant:.6f}e{e}")
    # hex-like
    forms.append(hex(rnd.randint(0, 2**32-1)))
    # decimals with leading/trailing dot
    forms.append("." + str(rnd.randint(0, 999)))
    forms.append(str(rnd.randint(0, 999)) + ".")
    return _choose(rnd, forms)

def _rand_blob(rnd):
    n = rnd.randint(1, 8)
    hexchars = "0123456789ABCDEF"
    s = "".join(rnd.choice(hexchars) for _ in range(n*2))
    return "X'" + s + "'"

def _rand_string(rnd, max_len=20):
    # generate a SQL string literal with escapes
    length = rnd.randint(0, max_len)
    pool = string.ascii_letters + string.digits + " _-!@#$%^&*()[]{}:,.?/\\|"
    raw = "".join(rnd.choice(pool) for _ in range(length))
    # escape single quotes by doubling
    escaped = raw.replace("'", "''")
    prefix = ""
    if _maybe(rnd, 0.1):
        # E'' or N'' prefixes (Postgres / SQL Server)
        prefix = _choose(rnd, ["E", "N"])
    return f"{prefix}'" + escaped + "'"

def _rand_identifier(rnd, allow_keywords=True):
    # sometimes use reserved keywords as identifiers to exercise quoting
    keywords = [
        "select","from","where","group","order","by","insert","update","delete","join","on",
        "left","right","full","inner","outer","cross","into","values","table","create",
        "drop","alter","index","view","as","and","or","not","null","like","is","in","between",
        "having","distinct","union","intersect","except","limit","offset","exists","case","when",
        "then","else","end","primary","key","unique","check","foreign","references"
    ]
    if allow_keywords and _maybe(rnd, 0.2):
        base = _choose(rnd, keywords)
    else:
        start = rnd.choice(string.ascii_letters + "_")
        rest = "".join(rnd.choice(string.ascii_letters + string.digits + "_") for _ in range(rnd.randint(0, 10)))
        base = start + rest
    # randomly quote the identifier to test tokenizer/identifier parsing
    if _maybe(rnd, 0.3):
        # double quotes
        return '"' + base.replace('"', '""') + '"'
    elif _maybe(rnd, 0.2):
        return '[' + base.replace(']', ']]') + ']'
    elif _maybe(rnd, 0.2):
        return '`' + base.replace('`', '``') + '`'
    else:
        return base

def _rand_qualified_name(rnd):
    if _maybe(rnd, 0.3):
        return _rand_identifier(rnd) + "." + _rand_identifier(rnd)
    return _rand_identifier(rnd)

def _rand_type(rnd):
    numeric = ["INT", "INTEGER", "SMALLINT", "BIGINT", "NUMERIC", "DECIMAL", "REAL", "DOUBLE", "FLOAT"]
    text = ["TEXT", "VARCHAR", "CHAR", "NCHAR", "NVARCHAR", "CLOB"]
    datetime = ["DATE", "TIME", "TIMESTAMP", "DATETIME"]
    blob = ["BLOB"]
    types = numeric + text + datetime + blob + ["BOOLEAN"]
    t = _choose(rnd, types)
    if t in ["VARCHAR", "CHAR", "NCHAR", "NVARCHAR"] and _maybe(rnd, 0.8):
        t += f"({_rand_int(rnd,1,255)})"
    elif t in ["DECIMAL", "NUMERIC"] and _maybe(rnd, 0.7):
        t += f"({_rand_int(rnd,1,18)},{_rand_int(rnd,0,6)})"
    return t

def _rand_bool(rnd):
    return _choose(rnd, ["TRUE", "FALSE"])

def _rand_literal(rnd):
    choices = []
    choices.append("NULL")
    choices.append(_rand_float(rnd))
    choices.append(_rand_string(rnd))
    choices.append(_rand_blob(rnd))
    choices.append(_rand_bool(rnd))
    # Some date/time-ish
    choices.append(f"DATE({_rand_string(rnd,10)})")
    choices.append(f"TIME({_rand_string(rnd,10)})")
    return _choose(rnd, choices)

def _rand_param(rnd):
    forms = ["?", "?"+str(_rand_int(rnd,1,10)), ":"+_rand_identifier(rnd, allow_keywords=False),
             "@"+_rand_identifier(rnd, allow_keywords=False), "$"+_rand_identifier(rnd, allow_keywords=False)]
    return _choose(rnd, forms)

def _rand_collation(rnd):
    return _choose(rnd, ["BINARY","NOCASE","RTRIM","UNICODE","LOCALIZED","UNICASE"])

def _rand_function_name(rnd):
    funcs = [
        "ABS","ROUND","LENGTH","SUBSTR","LOWER","UPPER","TRIM","LTRIM","RTRIM","COALESCE","NULLIF",
        "IFNULL","RANDOM","RANDOMBLOB","HEX","QUOTE","CHAR","PRINTF","REPLACE","STRFTIME","DATE","DATETIME",
        "JULIANDAY","TIME","TOTAL","AVG","SUM","MIN","MAX","COUNT"
    ]
    return _choose(rnd, funcs)

def _rand_unary_op(rnd):
    return _choose(rnd, ["-","+","~","NOT"])

def _rand_binary_op(rnd):
    return _choose(rnd, [
        "||","+","-","*","/","%","&","|","^","<<",">>",
        "=","!=","<>","<",">","<=",">=",
        "AND","OR","IS","IS NOT","LIKE","GLOB","REGEXP","MATCH"
    ])

def _rand_ordering_term(rnd, depth):
    s = _rand_expr(rnd, depth+1)
    if _maybe(rnd, 0.5):
        s += " COLLATE " + _rand_collation(rnd)
    if _maybe(rnd, 0.7):
        s += " " + _choose(rnd, ["ASC","DESC"])
    if _maybe(rnd, 0.3):
        s += " NULLS " + _choose(rnd, ["FIRST","LAST"])
    return s

def _rand_case_expr(rnd, depth):
    n_when = _rand_int(rnd,1,3)
    parts = ["CASE"]
    if _maybe(rnd, 0.3):
        parts.append(_rand_expr(rnd, depth+1))
    for _ in range(n_when):
        parts.append("WHEN " + _rand_expr(rnd, depth+1) + " THEN " + _rand_expr(rnd, depth+1))
    if _maybe(rnd, 0.6):
        parts.append("ELSE " + _rand_expr(rnd, depth+1))
    parts.append("END")
    return " ".join(parts)

def _rand_expr(rnd, depth=0):
    max_depth = 3
    if depth >= max_depth:
        # base terminals
        base_choices = [
            _rand_literal(rnd),
            _rand_param(rnd),
            _rand_qualified_name(rnd)
        ]
        if _maybe(rnd, 0.2):
            base_choices.append("(" + _rand_expr(rnd, depth) + ")")
        return _choose(rnd, base_choices)
    choice = rnd.random()
    if choice < 0.18:
        # function call
        fn = _rand_function_name(rnd)
        arg_count = _rand_int(rnd, 0, 3)
        if fn.upper() in ["COUNT"] and _maybe(rnd, 0.5):
            args = ["*"]
        else:
            args = [_rand_expr(rnd, depth+1) for _ in range(arg_count)]
        if _maybe(rnd, 0.2):
            # DISTINCT in aggregate
            args = ["DISTINCT " + ", ".join(args)] if args else ["DISTINCT *"]
        return f"{fn}(" + ", ".join(args) + ")"
    elif choice < 0.36:
        # unary
        return _rand_unary_op(rnd) + " " + _rand_expr(rnd, depth+1)
    elif choice < 0.62:
        # binary
        left = _rand_expr(rnd, depth+1)
        op = _rand_binary_op(rnd)
        right = _rand_expr(rnd, depth+1)
        return f"{left} {op} {right}"
    elif choice < 0.76:
        # parens
        return "(" + _rand_expr(rnd, depth+1) + ")"
    elif choice < 0.88:
        # IN list
        n = _rand_int(rnd, 0, 4)
        if _maybe(rnd, 0.3):
            subq = _rand_select(rnd, depth+1, allow_set=False)
            lst = "(" + subq + ")"
        else:
            lst = "(" + ", ".join(_rand_expr(rnd, depth+1) for _ in range(n)) + ")"
        return _rand_expr(rnd, depth+1) + (" NOT" if _maybe(rnd,0.3) else "") + " IN " + lst
    elif choice < 0.96:
        # BETWEEN
        e1 = _rand_expr(rnd, depth+1)
        e2 = _rand_expr(rnd, depth+1)
        e3 = _rand_expr(rnd, depth+1)
        return f"{e1} {'NOT ' if _maybe(rnd,0.3) else ''}BETWEEN {e2} AND {e3}"
    else:
        # CASE
        return _rand_case_expr(rnd, depth+1)

def _rand_table_constraint(rnd):
    kinds = []
    if _maybe(rnd, 0.7):
        cols = "(" + ", ".join(_rand_identifier(rnd) for _ in range(_rand_int(rnd,1,3))) + ")"
        kinds.append("PRIMARY KEY " + cols)
    if _maybe(rnd, 0.5):
        cols = "(" + ", ".join(_rand_identifier(rnd) for _ in range(_rand_int(rnd,1,3))) + ")"
        kinds.append("UNIQUE " + cols)
    if _maybe(rnd, 0.5):
        kinds.append("CHECK (" + _rand_expr(rnd) + ")")
    if _maybe(rnd, 0.5):
        cols = "(" + ", ".join(_rand_identifier(rnd) for _ in range(_rand_int(rnd,1,2))) + ")"
        kinds.append("FOREIGN KEY " + cols + " REFERENCES " + _rand_qualified_name(rnd) +
                     "(" + ", ".join(_rand_identifier(rnd) for _ in range(_rand_int(rnd,1,2))) + ")")
    if not kinds:
        kinds.append("CHECK (" + _rand_expr(rnd) + ")")
    return _choose(rnd, kinds)

def _rand_column_def(rnd):
    name = _rand_identifier(rnd)
    t = _rand_type(rnd)
    parts = [name, t]
    if _maybe(rnd, 0.4):
        parts.append("PRIMARY KEY")
        if _maybe(rnd, 0.2):
            parts.append("AUTOINCREMENT")
    if _maybe(rnd, 0.4):
        parts.append("NOT NULL")
    if _maybe(rnd, 0.3):
        parts.append("UNIQUE")
    if _maybe(rnd, 0.4):
        parts.append("DEFAULT " + _rand_expr(rnd))
    if _maybe(rnd, 0.3):
        parts.append("CHECK (" + _rand_expr(rnd) + ")")
    if _maybe(rnd, 0.3):
        parts.append("REFERENCES " + _rand_qualified_name(rnd) + "(" + _rand_identifier(rnd) + ")")
    return " ".join(parts)

def _rand_table_def(rnd):
    cols = [_rand_column_def(rnd) for _ in range(_rand_int(rnd,1,5))]
    if _maybe(rnd, 0.6):
        cols.append(_rand_table_constraint(rnd))
    return "(" + ", ".join(cols) + ")"

def _rand_join_clause(rnd, depth):
    join_types = ["JOIN","INNER JOIN","LEFT JOIN","LEFT OUTER JOIN","RIGHT JOIN","FULL JOIN","CROSS JOIN"]
    t1 = _rand_qualified_name(rnd) + (" " + _rand_identifier(rnd) if _maybe(rnd,0.5) else "")
    t2 = _rand_qualified_name(rnd) + (" " + _rand_identifier(rnd) if _maybe(rnd,0.5) else "")
    join = t1 + " " + _choose(rnd, join_types) + " " + t2
    if _maybe(rnd, 0.7):
        join += " ON " + _rand_expr(rnd, depth+1)
    else:
        join += " USING (" + ", ".join(_rand_identifier(rnd) for _ in range(_rand_int(rnd,1,3))) + ")"
    return join

def _rand_with_clause(rnd, depth):
    if not _maybe(rnd, 0.3):
        return None
    recursive = "RECURSIVE " if _maybe(rnd, 0.3) else ""
    n = _rand_int(rnd,1,3)
    ctes = []
    for _ in range(n):
        name = _rand_identifier(rnd, allow_keywords=False)
        cols = ""
        if _maybe(rnd, 0.4):
            cols = "(" + ", ".join(_rand_identifier(rnd) for _ in range(_rand_int(rnd,1,4))) + ")"
        cte = f"{name}{cols} AS (" + _rand_select(rnd, depth+1, allow_set=False) + ")"
        ctes.append(cte)
    return f"WITH {recursive}" + ", ".join(ctes)

def _rand_select_core(rnd, depth):
    parts = []
    if _maybe(rnd, 0.4):
        parts.append("DISTINCT")
    elif _maybe(rnd, 0.2):
        parts.append("ALL")
    ncols = _rand_int(rnd,1,5)
    cols = []
    for _ in range(ncols):
        if _maybe(rnd, 0.2):
            cols.append("*")
        else:
            e = _rand_expr(rnd, depth+1)
            if _maybe(rnd, 0.4):
                e += " AS " + _rand_identifier(rnd)
            cols.append(e)
    parts.append(", ".join(cols))
    core = "SELECT " + " ".join(parts)
    if _maybe(rnd, 0.8):
        if _maybe(rnd, 0.7):
            frm = _rand_join_clause(rnd, depth+1)
        else:
            tables = []
            for _ in range(_rand_int(rnd,1,3)):
                t = _rand_qualified_name(rnd)
                if _maybe(rnd, 0.5):
                    t += " " + _rand_identifier(rnd)
                tables.append(t)
            frm = ", ".join(tables)
        core += " FROM " + frm
    if _maybe(rnd, 0.6):
        core += " WHERE " + _rand_expr(rnd, depth+1)
    if _maybe(rnd, 0.4):
        n = _rand_int(rnd,1,3)
        group_cols = ", ".join(_rand_expr(rnd, depth+1) for _ in range(n))
        core += " GROUP BY " + group_cols
        if _maybe(rnd, 0.5):
            core += " HAVING " + _rand_expr(rnd, depth+1)
    if _maybe(rnd, 0.5):
        n = _rand_int(rnd,1,3)
        order = ", ".join(_rand_ordering_term(rnd, depth+1) for _ in range(n))
        core += " ORDER BY " + order
    if _maybe(rnd, 0.5):
        core += " LIMIT " + str(_rand_int(rnd,0,1000))
        if _maybe(rnd, 0.5):
            if _maybe(rnd, 0.5):
                core += " OFFSET " + str(_rand_int(rnd,0,1000))
            else:
                core = core.replace(" LIMIT ", " LIMIT " + str(_rand_int(rnd,0,1000)) + ", ", 1)
    return core

def _rand_select(rnd, depth=0, allow_set=True):
    with_clause = _rand_with_clause(rnd, depth)
    core = _rand_select_core(rnd, depth+1)
    if allow_set and _maybe(rnd, 0.4):
        ops = ["UNION","UNION ALL","INTERSECT","EXCEPT"]
        other = _rand_select_core(rnd, depth+1)
        core = "(" + core + ") " + _choose(rnd, ops) + " (" + other + ")"
    if with_clause:
        return with_clause + " " + core
    return core

def _rand_insert(rnd):
    tbl = _rand_qualified_name(rnd)
    cols = ""
    if _maybe(rnd, 0.6):
        cols = "(" + ", ".join(_rand_identifier(rnd) for _ in range(_rand_int(rnd,1,5))) + ")"
    if _maybe(rnd, 0.5):
        rows = []
        for _ in range(_rand_int(rnd,1,3)):
            row = "(" + ", ".join(_rand_expr(rnd) for _ in range(_rand_int(rnd,1,5))) + ")"
            rows.append(row)
        return f"INSERT INTO {tbl} {cols} VALUES " + ", ".join(rows)
    else:
        return f"INSERT INTO {tbl} {cols} " + _rand_select(rnd)

def _rand_update(rnd):
    tbl = _rand_qualified_name(rnd)
    sets = []
    for _ in range(_rand_int(rnd,1,4)):
        sets.append(_rand_identifier(rnd) + " = " + _rand_expr(rnd))
    sql = f"UPDATE {tbl} SET " + ", ".join(sets)
    if _maybe(rnd, 0.7):
        sql += " WHERE " + _rand_expr(rnd)
    return sql

def _rand_delete(rnd):
    tbl = _rand_qualified_name(rnd)
    sql = f"DELETE FROM {tbl}"
    if _maybe(rnd, 0.7):
        sql += " WHERE " + _rand_expr(rnd)
    if _maybe(rnd, 0.5):
        sql += " ORDER BY " + ", ".join(_rand_ordering_term(rnd,0) for _ in range(_rand_int(rnd,1,3)))
    if _maybe(rnd, 0.4):
        sql += " LIMIT " + str(_rand_int(rnd,0,1000))
    return sql

def _rand_create_table(rnd):
    tbl = _rand_qualified_name(rnd)
    temp = "TEMP " if _maybe(rnd, 0.2) else ""
    if_exists = "IF NOT EXISTS " if _maybe(rnd, 0.4) else ""
    return f"CREATE {temp}TABLE {if_exists}{tbl} " + _rand_table_def(rnd)

def _rand_alter_table(rnd):
    tbl = _rand_qualified_name(rnd)
    actions = []
    if _maybe(rnd, 0.5):
        actions.append("RENAME TO " + _rand_identifier(rnd))
    if _maybe(rnd, 0.7):
        actions.append("ADD COLUMN " + _rand_column_def(rnd))
    if _maybe(rnd, 0.3):
        actions.append("DROP COLUMN " + _rand_identifier(rnd))
    if not actions:
        actions.append("ADD COLUMN " + _rand_column_def(rnd))
    return "ALTER TABLE " + tbl + " " + ", ".join(actions)

def _rand_drop_table(rnd):
    return "DROP TABLE " + ("IF EXISTS " if _maybe(rnd,0.5) else "") + _rand_qualified_name(rnd)

def _rand_index(rnd):
    unique = "UNIQUE " if _maybe(rnd,0.4) else ""
    name = _rand_identifier(rnd)
    tbl = _rand_qualified_name(rnd)
    cols = "(" + ", ".join(_rand_identifier(rnd) for _ in range(_rand_int(rnd,1,4))) + ")"
    where = " WHERE " + _rand_expr(rnd) if _maybe(rnd,0.4) else ""
    return f"CREATE {unique}INDEX " + ("IF NOT EXISTS " if _maybe(rnd,0.4) else "") + f"{name} ON {tbl} {cols}{where}"

def _rand_view(rnd):
    name = _rand_identifier(rnd)
    if_exists = "IF NOT EXISTS " if _maybe(rnd,0.4) else ""
    return f"CREATE VIEW {if_exists}{name} AS " + _rand_select(rnd)

def _rand_transaction(rnd):
    return _choose(rnd, ["BEGIN", "BEGIN TRANSACTION", "COMMIT", "ROLLBACK", "END", "SAVEPOINT " + _rand_identifier(rnd), "RELEASE " + _rand_identifier(rnd)])

def _rand_misc(rnd):
    return _choose(rnd, [
        "EXPLAIN " + _rand_select(rnd),
        "ANALYZE " + _rand_qualified_name(rnd),
        "VACUUM",
        "PRAGMA " + _rand_identifier(rnd) + "=" + str(_rand_int(rnd,0,100)),
        "PRAGMA " + _rand_identifier(rnd) + "(" + _rand_identifier(rnd) + "=" + str(_rand_int(rnd,0,1)) + ")"
    ])

def _base_statements():
    rnd = _rng()
    stmts = []
    # Simple selects
    for kw in ["", "DISTINCT ", "ALL "]:
        for star in [True, False]:
            s = "SELECT " + kw
            if star:
                s += "*"
            else:
                s += "1, 'text', NULL, 42.5"
            if _maybe(rnd, 0.7):
                s += " FROM " + _rand_identifier(rnd)
            stmts.append(s)
    # Joins
    join_types = ["JOIN","INNER JOIN","LEFT JOIN","LEFT OUTER JOIN","RIGHT JOIN","FULL JOIN","CROSS JOIN"]
    for jt in join_types:
        s = "SELECT * FROM a " + jt + " b ON a.id = b.id"
        stmts.append(s)
    # Group by, having
    stmts.append("SELECT a, COUNT(*), SUM(b) FROM t GROUP BY a")
    stmts.append("SELECT a, COUNT(*), SUM(b) FROM t GROUP BY a HAVING SUM(b) > 10")
    # Order by variations
    stmts.append("SELECT a, b FROM t ORDER BY a ASC, b DESC NULLS LAST")
    # Subqueries
    stmts.append("SELECT * FROM (SELECT 1 AS x, 2 AS y)")
    stmts.append("SELECT * FROM t WHERE a IN (SELECT a FROM t2)")
    stmts.append("SELECT EXISTS (SELECT 1 FROM t WHERE x=1)")
    # Set operations
    stmts.append("(SELECT 1 AS x) UNION (SELECT 2)")
    stmts.append("(SELECT 1 AS x) UNION ALL (SELECT 2)")
    stmts.append("(SELECT 1 AS x) INTERSECT (SELECT 1)")
    stmts.append("(SELECT 1 AS x) EXCEPT (SELECT 1)")
    # Insert
    stmts.append("INSERT INTO t(a,b) VALUES (1,2)")
    stmts.append("INSERT INTO t VALUES (1,'x'), (2,'y'), (3,'z')")
    stmts.append("INSERT INTO t(a) SELECT a FROM t2")
    # Update/Delete
    stmts.append("UPDATE t SET a = a+1 WHERE id = 1")
    stmts.append("DELETE FROM t WHERE a BETWEEN 1 AND 5")
    stmts.append("DELETE FROM t ORDER BY a LIMIT 10")
    # DDL
    stmts.append("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL, val REAL DEFAULT 0.0)")
    stmts.append("CREATE TABLE IF NOT EXISTS t2 (a INT, b TEXT, CHECK (a > 0), UNIQUE(a))")
    stmts.append("CREATE TEMP TABLE tt (x BLOB, y DATE, z BOOLEAN DEFAULT TRUE)")
    stmts.append("ALTER TABLE t ADD COLUMN c TEXT")
    stmts.append("ALTER TABLE t RENAME TO t_renamed")
    stmts.append("DROP TABLE IF EXISTS t3")
    stmts.append("CREATE INDEX idx_t_a ON t(a)")
    stmts.append("CREATE UNIQUE INDEX IF NOT EXISTS idx_t_b ON t(b) WHERE b IS NOT NULL")
    stmts.append("CREATE VIEW v AS SELECT * FROM t")
    # Transactions and pragmas
    stmts.append("BEGIN")
    stmts.append("COMMIT")
    stmts.append("ROLLBACK")
    stmts.append("PRAGMA journal_mode=WAL")
    stmts.append("PRAGMA foreign_keys=ON")
    # Expressions covering operators
    stmts.append("SELECT -1, +2, ~3, NOT 0")
    stmts.append("SELECT 1+2*3/4%5, 1<<2, 8>>1, 1|2, 1&3, 1^1")
    stmts.append("SELECT 'a'||'b', 'O''Reilly', X'ABCD'")
    stmts.append("SELECT 1 IS NULL, 1 IS NOT NULL, 'a' LIKE '%a%' ESCAPE '\\\\'")
    stmts.append("SELECT CASE WHEN 1=1 THEN 'yes' ELSE 'no' END")
    stmts.append('SELECT "from", [select], `table` FROM t')
    # WITH CTE
    stmts.append("WITH cte AS (SELECT 1 AS x) SELECT x FROM cte")
    stmts.append("WITH RECURSIVE cte(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM cte WHERE n<5) SELECT * FROM cte")
    # Explain
    stmts.append("EXPLAIN SELECT * FROM t")
    # Comments
    stmts.append("/* comment */ SELECT 1 -- trailing")
    return stmts

def _edge_token_statements():
    rnd = _rng()
    stmts = []
    # Numbers edge
    nums = ["0", "00", "0001", "1.0", ".5", "5.", "-0.0", "+.0", "1e-5", "-1E+5", "1e309", "0xFF", "0XdeadBEEF"]
    stmts.append("SELECT " + ", ".join(nums))
    # Strings edge
    strings = [
        "'simple'", "'O''Reilly'", "E'\\\\n\\\\t'", "N'Unicode'", "'multi\\nline'", "''", "' '"
    ]
    stmts.append("SELECT " + ", ".join(strings))
    # BLOBs and params
    stmts.append("SELECT X'00FF', X'ABCD', ?, ?1, :name, @var, $dollar")
    # Operators sequences
    stmts.append("SELECT 1--2\\n")
    stmts.append("SELECT 1 /* nested /* not really */ comment */ + 2")
    stmts.append("/* leading comment */ SELECT /* mid */ 1 /* tail */")
    # Casts and collation
    stmts.append("SELECT '10'::INT, CAST('10' AS INTEGER), a COLLATE NOCASE FROM t")
    # IS / LIKE / GLOB / REGEXP
    stmts.append("SELECT NULL IS NULL, 'a' GLOB '[a-z]', 'a' REGEXP 'a+'")
    # Null ordering
    stmts.append("SELECT * FROM t ORDER BY a NULLS FIRST")
    # Complex order-by expressions
    stmts.append("SELECT * FROM t ORDER BY (a + b) DESC, c COLLATE BINARY ASC")
    # Escaped quotes
    stmts.append("SELECT 'It''s fine', 'quote\\\\'' inside'")
    # Parenthesis stress
    stmts.append("SELECT (((1+2)*3) - ((4/5)+6))")
    # Like with ESCAPE
    stmts.append("SELECT 'a!b' LIKE '%!b' ESCAPE '!'")
    # Weird whitespace
    stmts.append("SELECT\\t1,\\n2,\\r3,\\f4,\\v5")
    return stmts

def _random_statements(count=1200):
    rnd = _rng()
    stmts = []
    gens = []
    gens.extend([_rand_select]*5)
    gens.extend([_rand_insert]*2)
    gens.extend([_rand_update])
    gens.extend([_rand_delete])
    gens.extend([_rand_create_table])
    gens.extend([_rand_alter_table])
    gens.extend([_rand_drop_table])
    gens.extend([_rand_index])
    gens.extend([_rand_view])
    gens.extend([_rand_transaction])
    gens.extend([_rand_misc])
    for _ in range(count):
        g = _choose(rnd, gens)
        try:
            if g == _rand_select:
                s = g(rnd)
            else:
                s = g(rnd)
            if _maybe(rnd, 0.7):
                s = s  # leave as is
        except Exception:
            # In case generator fails, fall back to a simple select
            s = "SELECT 1"
        stmts.append(s)
    return stmts

def _insert_comments_randomly(rnd, s):
    # Insert block or line comments at random boundaries (between tokens)
    tokens = re.split(r"(\\s+)", s)
    out = []
    for t in tokens:
        if t.strip() == "":
            out.append(t)
            continue
        out.append(t)
        if _maybe(rnd, 0.1):
            if _maybe(rnd, 0.5):
                out.append(" /*" + _rand_identifier(rnd, allow_keywords=False) + "*/ ")
            else:
                out.append(" --" + _rand_identifier(rnd, allow_keywords=False) + "\\n")
    return "".join(out)

def _randomize_whitespace(rnd, s):
    # Replace spaces with random whitespace sequences
    ws = [" ", "\\t", "\\n", "  ", "\\r", " \\t ", "\\n\\n", "\\f", "\\v"]
    return re.sub(r"\\s+", lambda m: _choose(rnd, ws), s)

def _toggle_case_random(rnd, s):
    # Randomly upper/lower each alphabetic character
    chars = []
    for c in s:
        if c.isalpha():
            if _maybe(rnd, 0.5):
                chars.append(c.upper())
            else:
                chars.append(c.lower())
        else:
            chars.append(c)
    return "".join(chars)

def _random_parentheses(rnd, s):
    # Wrap random tokens with parentheses
    parts = s.split()
    for i in range(len(parts)):
        if _maybe(rnd, 0.15) and len(parts[i]) < 40:
            parts[i] = "(" + parts[i] + ")"
    return " ".join(parts)

def _typo_keywords(rnd, s):
    # Introduce small typos in keywords for invalid variants
    def typo(word):
        if len(word) <= 3:
            return word
        idx = rnd.randint(0, len(word)-1)
        return word[:idx] + word[idx+1:]
    return re.sub(r"\\b(SELECT|FROM|WHERE|INSERT|UPDATE|DELETE|CREATE|TABLE|JOIN|GROUP|ORDER|LIMIT|OFFSET|VALUES|INTO|AS|AND|OR)\\b",
                  lambda m: typo(m.group(1)) if _maybe(rnd, 0.2) else m.group(1),
                  s, flags=re.IGNORECASE)

def _unbalance_parentheses(rnd, s):
    if "(" in s and _maybe(rnd, 0.3):
        # remove a random closing paren
        return s.replace(")", "", 1)
    if ")" in s and _maybe(rnd, 0.3):
        # remove a random opening paren
        return s.replace("(", "", 1)
    return s

def _append_stray_trailing(rnd, s):
    tails = [";", " ;", ";;", "; --end\\n", "; /*end*/", "\\0", "#", "::", ",", "/", "\\"]
    if _maybe(rnd, 0.7):
        return s + _choose(rnd, tails)
    return s

def _mutate_statement(rnd, s):
    ops = []
    ops.append(lambda x: _insert_comments_randomly(rnd, x))
    ops.append(lambda x: _randomize_whitespace(rnd, x))
    ops.append(lambda x: _toggle_case_random(rnd, x))
    ops.append(lambda x: _random_parentheses(rnd, x))
    ops.append(lambda x: _typo_keywords(rnd, x))
    ops.append(lambda x: _unbalance_parentheses(rnd, x))
    ops.append(lambda x: _append_stray_trailing(rnd, x))
    rnd.shuffle(ops)
    out = s
    # Apply 1 to 4 random ops
    for i in range(rnd.randint(1, 4)):
        out = ops[i](out)
    return out

def _mutations(base_stmts, count=1800):
    rnd = _rng()
    out = []
    if not base_stmts:
        return out
    for _ in range(count):
        s = _choose(rnd, base_stmts)
        out.append(_mutate_statement(rnd, s))
    return out

def _bad_statements(count=300):
    rnd = _rng()
    out = []
    pools = string.ascii_letters + string.digits + " _-!@#$%^&*()[]{}:,.?/\\|;\\'\\\\" + "\\n\\t\\r"
    for _ in range(count):
        length = rnd.randint(5, 120)
        s = "".join(rnd.choice(pools) for _ in range(length))
        # ensure some unmatched quotes or comment starts
        if _maybe(rnd, 0.3):
            s = "SELECT '" + s
        if _maybe(rnd, 0.3):
            s = "/* " + s
        out.append(s)
    return out

def _ensure_semicolon_variants(stmts, proportion=0.7):
    rnd = _rng()
    out = []
    for s in stmts:
        if _maybe(rnd, proportion):
            out.append(s + ";")
        out.append(s)
    return out

def _dedup_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def fuzz(parse_sql):
    global _FUZZ_DONE
    if _FUZZ_DONE:
        return False
    rnd = _rng()
    # Generate corpus
    base = _base_statements()
    edges = _edge_token_statements()
    randoms = _random_statements(count=1400)
    muts = _mutations(base + edges + randoms, count=1900)
    bads = _bad_statements(count=350)
    corpus = base + edges + randoms + muts + bads
    # Add some extra focused expressions-only selects to deeply test expression parsing
    extra_exprs = []
    for _ in range(200):
        extra_exprs.append("SELECT " + ", ".join(_rand_expr(rnd) for __ in range(rnd.randint(1,5))))
    corpus.extend(extra_exprs)
    # Semicolon variants
    corpus = _ensure_semicolon_variants(corpus, proportion=0.5)
    # Deduplicate to reduce redundant parses
    corpus = _dedup_preserve_order(corpus)
    # Limit size if somehow exploded
    if len(corpus) > 6000:
        corpus = corpus[:6000]
    # Execute through parser
    parse_sql(corpus)
    _FUZZ_DONE = True
    return False
"""
        return {"code": code}