import os
import re
from textwrap import dedent


class Solution:
    def solve(self, resources_path: str) -> dict:
        # Extract uppercase keywords from the grammar, ignoring non-terminals in <...>
        keywords = []
        grammar_path = os.path.join(resources_path, "sql_grammar.txt")
        try:
            with open(grammar_path, "r", encoding="utf-8") as f:
                data = f.read()

            spans = []
            for m in re.finditer(r"<[^>]*>", data):
                spans.append((m.start(), m.end()))

            def in_span(idx: int) -> bool:
                for s, e in spans:
                    if s <= idx < e:
                        return True
                return False

            raw = []
            for m in re.finditer(r"\b([A-Z][A-Z0-9_]{1,})\b", data):
                if not in_span(m.start()):
                    raw.append(m.group(1))

            seen = set()
            keywords = [k for k in raw if not (k in seen or seen.add(k))]
        except Exception:
            keywords = []

        # Default generic SQL keywords to supplement grammar-derived ones
        default_keywords = [
            "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "TABLE", "INDEX", "VIEW",
            "TRIGGER", "DROP", "ALTER", "WHERE", "FROM", "GROUP", "BY", "ORDER", "HAVING",
            "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "FULL", "CROSS", "NATURAL", "ON",
            "USING", "UNION", "ALL", "EXCEPT", "INTERSECT", "VALUES", "INTO", "AS",
            "DISTINCT", "LIMIT", "OFFSET", "AND", "OR", "NOT", "NULL", "IS", "IN",
            "BETWEEN", "LIKE", "GLOB", "CASE", "WHEN", "THEN", "ELSE", "END", "PRIMARY",
            "KEY", "FOREIGN", "REFERENCES", "CHECK", "DEFAULT", "UNIQUE", "IF", "EXISTS",
            "BEGIN", "TRANSACTION", "COMMIT", "ROLLBACK"
        ]

        seen = set()
        merged = []
        for lst in (keywords, default_keywords):
            for k in lst:
                if k not in seen:
                    seen.add(k)
                    merged.append(k)
        keywords = merged

        base_code = '''
import random
import string

GRAMMAR_KEYWORDS = __KEYWORDS__


_initialized = False
_corpus = []
_corpus_limit = 2000
_iteration = 0

_table_names = []
_column_names = {}


def _ensure_initialized():
    global _initialized, _table_names, _column_names, _corpus
    if _initialized:
        return
    _initialized = True

    # Predefined table and column names
    _table_names = [
        "t", "users", "orders", "products", "accounts", "logs",
        "sessions", "events", "items", "messages", "posts"
    ]
    _column_names = {
        "users": ["id", "name", "email", "age", "status", "created_at"],
        "orders": ["id", "user_id", "amount", "created_at", "status"],
        "products": ["id", "name", "price", "stock", "category"],
        "accounts": ["id", "balance", "currency", "created_at"],
        "logs": ["id", "msg", "level", "created_at"],
    }

    # Initial seed corpus with a variety of basic statements
    seeds = [
        "SELECT 1",
        "SELECT * FROM t",
        "SELECT a, b, c FROM t WHERE a > 10",
        "SELECT name, age FROM users WHERE age >= 18 ORDER BY age DESC",
        "INSERT INTO t VALUES (1, 2, 3)",
        "INSERT INTO users (id, name, email) VALUES (1, 'alice', 'a@example.com')",
        "UPDATE users SET name = 'bob' WHERE id = 1",
        "DELETE FROM users WHERE id = 2",
        "CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, val TEXT)",
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, age INTEGER)",
        "CREATE INDEX idx_users_email ON users(email)",
        "DROP TABLE IF EXISTS old_table",
        "BEGIN TRANSACTION",
        "COMMIT",
        "ROLLBACK",
        "SELECT COUNT(*) FROM users",
        "SELECT DISTINCT status FROM orders",
        "SELECT u.name, o.amount FROM users u JOIN orders o ON u.id = o.user_id",
        "SELECT * FROM products WHERE price BETWEEN 10 AND 100",
    ]
    _corpus.extend(seeds)


SQL_BASE_KEYWORDS = set([
    "SELECT","INSERT","UPDATE","DELETE","CREATE","TABLE","INDEX","VIEW","TRIGGER",
    "DROP","ALTER","WHERE","FROM","GROUP","BY","ORDER","HAVING","JOIN","LEFT","RIGHT",
    "INNER","OUTER","FULL","CROSS","NATURAL","ON","USING","UNION","ALL","EXCEPT",
    "INTERSECT","VALUES","INTO","AS","DISTINCT","LIMIT","OFFSET","AND","OR","NOT",
    "NULL","IS","IN","BETWEEN","LIKE","CASE","WHEN","THEN","ELSE","END","PRIMARY",
    "KEY","FOREIGN","REFERENCES","CHECK","DEFAULT","UNIQUE","IF","EXISTS","BEGIN",
    "TRANSACTION","COMMIT","ROLLBACK"
])


ALL_KEYWORDS = sorted(set(GRAMMAR_KEYWORDS) | SQL_BASE_KEYWORDS)


SQL_FUNCTIONS = [
    "ABS", "MAX", "MIN", "SUM", "COUNT", "AVG", "COALESCE",
    "LENGTH", "SUBSTR", "UPPER", "LOWER", "ROUND",
    "RANDOM", "PRINTF", "DATE", "TIME", "DATETIME",
    "IFNULL", "HEX", "TOTAL"
]


SQL_TYPES = [
    "INTEGER", "REAL", "TEXT", "BLOB", "NUMERIC",
    "VARCHAR(255)", "CHAR(20)", "BOOLEAN",
    "DATE", "TIMESTAMP"
]


BINARY_OPERATORS = [
    "+", "-", "*", "/", "%", "=", "!=", "<>", "<", "<=", ">", ">=",
    "AND", "OR", "LIKE", "GLOB", "IS", "IS NOT"
]


UNARY_OPERATORS = [
    "NOT", "+", "-"
]


def _rand_identifier():
    length = random.randint(1, 8)
    first = random.choice(string.ascii_letters + "_")
    rest = "".join(random.choices(string.ascii_letters + string.digits + "_", k=length-1))
    ident = first + rest
    if random.random() < 0.1:
        # quoted identifier
        ident = '"%s"' % ident.replace('"', '""')
    return ident


def _rand_table_name():
    global _table_names
    if _table_names and random.random() < 0.7:
        return random.choice(_table_names)
    name = _rand_identifier()
    if len(_table_names) < 50:
        _table_names.append(name)
    return name


def _rand_column_name(table=None):
    if table and table in _column_names and _column_names[table]:
        if random.random() < 0.8:
            return random.choice(_column_names[table])
    if random.random() < 0.5:
        base_cols = [
            "id","name","value","val","col1","col2","x","y","z",
            "created_at","updated_at","status","flag","count","price"
        ]
        return random.choice(base_cols)
    return _rand_identifier()


def _rand_string_literal():
    length = random.randint(0, 20)
    chars = []
    for _ in range(length):
        ch = random.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789")
        chars.append(ch)
    s = "".join(chars)
    s = s.replace("'", "''")
    return "'%s'" % s


def _rand_numeric_literal():
    if random.random() < 0.3:
        # floating point
        int_part = random.randint(0, 100000)
        frac_part = random.randint(0, 100000)
        lit = "%d.%d" % (int_part, frac_part)
    else:
        lit = str(random.randint(-1000000, 1000000))
    return lit


def _rand_literal():
    r = random.random()
    if r < 0.4:
        return _rand_numeric_literal()
    elif r < 0.8:
        return _rand_string_literal()
    else:
        return random.choice(["NULL", "TRUE", "FALSE"])


def _rand_simple_expr():
    choice = random.random()
    if choice < 0.4:
        return _rand_literal()
    elif choice < 0.8:
        col = _rand_column_name()
        if random.random() < 0.3:
            tbl = _rand_table_name()
            return "%s.%s" % (tbl, col)
        return col
    else:
        return "?"  # parameter placeholder


def _gen_expr(depth=0, max_depth=3):
    if depth >= max_depth:
        return _rand_simple_expr()

    r = random.random()
    if r < 0.25:
        return _rand_simple_expr()
    elif r < 0.45:
        # parenthesized
        return "(%s)" % _gen_expr(depth+1, max_depth)
    elif r < 0.7:
        # binary operator
        left = _gen_expr(depth+1, max_depth)
        op = random.choice(BINARY_OPERATORS)
        right = _gen_expr(depth+1, max_depth)
        return "%s %s %s" % (left, op, right)
    elif r < 0.85:
        # unary
        op = random.choice(UNARY_OPERATORS)
        expr = _gen_expr(depth+1, max_depth)
        if op == "NOT":
            return "NOT %s" % expr
        else:
            return "%s%s" % (op, expr)
    elif r < 0.93:
        # function call
        func = random.choice(SQL_FUNCTIONS)
        arg_count = random.randint(0, 3)
        args = []
        for _ in range(arg_count):
            args.append(_gen_expr(depth+1, max_depth))
        return "%s(%s)" % (func, ", ".join(args))
    else:
        # CASE expression or BETWEEN
        if random.random() < 0.5:
            expr = _gen_expr(depth+1, max_depth)
            low = _gen_expr(depth+1, max_depth)
            high = _gen_expr(depth+1, max_depth)
            return "%s BETWEEN %s AND %s" % (expr, low, high)
        else:
            parts = []
            for _ in range(random.randint(1, 3)):
                cond = _gen_expr(depth+1, max_depth)
                val = _gen_expr(depth+1, max_depth)
                parts.append("WHEN %s THEN %s" % (cond, val))
            if random.random() < 0.5:
                else_expr = _gen_expr(depth+1, max_depth)
                parts.append("ELSE %s" % else_expr)
            return "CASE %s END" % " ".join(parts)


def _gen_order_expr():
    expr = _gen_expr(0, 2)
    direction = random.choice(["", " ASC", " DESC"])
    return expr + direction


def _gen_select_core(depth=0, max_depth=2):
    # SELECT [DISTINCT] expr_list FROM ...
    parts = ["SELECT"]
    if random.random() < 0.2:
        parts.append("DISTINCT")

    # select list
    cols = []
    if random.random() < 0.1:
        cols.append("*")
    else:
        count = random.randint(1, 5)
        for _ in range(count):
            expr = _gen_expr(0, 2)
            if random.random() < 0.3:
                alias = _rand_identifier()
                expr = "%s AS %s" % (expr, alias)
            cols.append(expr)
    parts.append(", ".join(cols))

    # FROM clause
    table_terms = []
    table_count = random.choices([0, 1, 2, 3], weights=[1, 3, 3, 2])[0]
    for _ in range(table_count):
        if depth < max_depth and random.random() < 0.2:
            # subquery as table
            sub = _gen_select_stmt(depth+1, max_depth)
            alias = _rand_identifier()
            term = "(%s) AS %s" % (sub, alias)
        else:
            tbl = _rand_table_name()
            alias = None
            if random.random() < 0.5:
                alias = _rand_identifier()
            term = tbl if not alias else "%s AS %s" % (tbl, alias)
        table_terms.append(term)

    if table_terms:
        # joins or simple list
        if len(table_terms) == 1:
            parts.append("FROM " + table_terms[0])
        else:
            join_str = table_terms[0]
            for next_term in table_terms[1:]:
                join_type = random.choice([
                    "JOIN", "LEFT JOIN", "RIGHT JOIN",
                    "INNER JOIN", "FULL JOIN", "CROSS JOIN"
                ])
                condition = _gen_expr(0, 2)
                join_str = "%s %s %s ON %s" % (
                    join_str, join_type, next_term, condition
                )
            parts.append("FROM " + join_str)

    # WHERE
    if random.random() < 0.7:
        where_expr = _gen_expr(0, 3)
        parts.append("WHERE " + where_expr)

    # GROUP BY / HAVING
    if random.random() < 0.4:
        group_exprs = [_gen_expr(0, 2) for _ in range(random.randint(1, 3))]
        parts.append("GROUP BY " + ", ".join(group_exprs))
        if random.random() < 0.5:
            having_expr = _gen_expr(0, 2)
            parts.append("HAVING " + having_expr)

    # ORDER BY
    if random.random() < 0.5:
        order_exprs = [_gen_order_expr() for _ in range(random.randint(1, 3))]
        parts.append("ORDER BY " + ", ".join(order_exprs))

    # LIMIT/OFFSET
    if random.random() < 0.5:
        limit = random.randint(0, 1000)
        parts.append("LIMIT %d" % limit)
        if random.random() < 0.5:
            offset = random.randint(0, 1000)
            parts.append("OFFSET %d" % offset)

    return " ".join(parts)


def _gen_select_stmt(depth=0, max_depth=2):
    stmt = _gen_select_core(depth, max_depth)
    # set operations: UNION / INTERSECT / EXCEPT
    if depth < max_depth and random.random() < 0.4:
        op = random.choice(["UNION", "UNION ALL", "INTERSECT", "EXCEPT"])
        right = _gen_select_core(depth+1, max_depth)
        stmt = "%s %s %s" % (stmt, op, right)
    return stmt


def _gen_insert_stmt():
    table = _rand_table_name()
    col_count = random.randint(0, 5)
    if col_count == 0 or random.random() < 0.2:
        cols = ""
    else:
        cols = []
        for _ in range(col_count):
            cols.append(_rand_column_name(table))
        cols = "(" + ", ".join(cols) + ")"

    if random.random() < 0.3:
        # INSERT ... DEFAULT VALUES
        stmt = "INSERT INTO %s %s DEFAULT VALUES" % (table, cols)
        return stmt.strip()

    row_count = random.randint(1, 4)
    rows = []
    for _ in range(row_count):
        if col_count == 0:
            value_count = random.randint(1, 5)
        else:
            value_count = col_count
        vals = []
        for _ in range(value_count):
            vals.append(_gen_expr(0, 2))
        rows.append("(%s)" % ", ".join(vals))
    stmt = "INSERT INTO %s %s VALUES %s" % (table, cols, ", ".join(rows))
    return stmt.strip()


def _gen_update_stmt():
    table = _rand_table_name()
    set_count = random.randint(1, 5)
    assigns = []
    for _ in range(set_count):
        col = _rand_column_name(table)
        expr = _gen_expr(0, 2)
        assigns.append("%s = %s" % (col, expr))
    stmt = "UPDATE %s SET %s" % (table, ", ".join(assigns))
    if random.random() < 0.8:
        cond = _gen_expr(0, 3)
        stmt += " WHERE " + cond
    return stmt


def _gen_delete_stmt():
    table = _rand_table_name()
    stmt = "DELETE FROM %s" % table
    if random.random() < 0.8:
        cond = _gen_expr(0, 3)
        stmt += " WHERE " + cond
    return stmt


def _register_table(name, cols):
    global _table_names, _column_names
    if name not in _table_names:
        _table_names.append(name)
    if name not in _column_names:
        _column_names[name] = []
    for c in cols:
        if c not in _column_names[name]:
            _column_names[name].append(c)


def _gen_create_table_stmt():
    name = _rand_table_name()
    col_count = random.randint(1, 6)
    cols = []
    col_names = []
    for _ in range(col_count):
        col_name = _rand_column_name()
        col_names.append(col_name)
        col_type = random.choice(SQL_TYPES)
        pieces = [col_name, col_type]
        if random.random() < 0.3:
            pieces.append("NOT NULL")
        if random.random() < 0.2:
            pieces.append("UNIQUE")
        if random.random() < 0.2:
            pieces.append("DEFAULT " + _rand_literal())
        if random.random() < 0.2:
            pieces.append("CHECK (" + _gen_expr(0, 2) + ")")
        cols.append(" ".join(pieces))

    table_constraints = []
    if random.random() < 0.3 and len(col_names) >= 1:
        k = min(len(col_names), random.randint(1, len(col_names)))
        pk_cols = random.sample(col_names, k=k)
        table_constraints.append("PRIMARY KEY (%s)" % ", ".join(pk_cols))
    if random.random() < 0.2 and len(col_names) >= 2 and _table_names:
        fk_col = random.choice(col_names)
        ref_table = random.choice(_table_names)
        ref_col = _rand_column_name(ref_table)
        table_constraints.append("FOREIGN KEY (%s) REFERENCES %s(%s)" % (fk_col, ref_table, ref_col))

    all_defs = cols + table_constraints
    stmt = "CREATE TABLE"
    if random.random() < 0.5:
        stmt += " IF NOT EXISTS"
    stmt += " %s (%s)" % (name, ", ".join(all_defs))

    _register_table(name, col_names)

    return stmt


def _gen_drop_table_stmt():
    name = _rand_table_name()
    stmt = "DROP TABLE"
    if random.random() < 0.5:
        stmt += " IF EXISTS"
    stmt += " %s" % name
    return stmt


def _gen_index_stmt():
    name = _rand_identifier()
    table = _rand_table_name()
    col_count = random.randint(1, 3)
    cols = [_rand_column_name(table) for _ in range(col_count)]
    unique = "UNIQUE " if random.random() < 0.3 else ""
    stmt = "CREATE %sINDEX %s ON %s (%s)" % (
        unique, name, table, ", ".join(cols)
    )
    return stmt


def _gen_random_keyword_soup():
    tokens = []
    count = random.randint(3, 20)
    for _ in range(count):
        r = random.random()
        if r < 0.4 and ALL_KEYWORDS:
            tokens.append(random.choice(ALL_KEYWORDS))
        elif r < 0.7:
            tokens.append(_rand_identifier())
        elif r < 0.85:
            tokens.append(_rand_numeric_literal())
        else:
            tokens.append(random.choice([",", "(", ")", "=", "*", ".", "+", "-", "/", "%"]))
    return " ".join(tokens)


def _mutate_statement(stmt):
    if not stmt:
        return stmt
    # Simple mutations: token-level and character-level
    if random.random() < 0.5:
        # token-level
        parts = []
        current = ""
        for ch in stmt:
            if ch.isalnum() or ch == "_":
                current += ch
            else:
                if current:
                    parts.append(current)
                    current = ""
                if not ch.isspace():
                    parts.append(ch)
        if current:
            parts.append(current)

        if not parts:
            return stmt

        op = random.random()
        if op < 0.33 and len(parts) > 1:
            # delete a random token
            idx = random.randrange(len(parts))
            del parts[idx]
        elif op < 0.66:
            # insert a random token
            idx = random.randrange(len(parts) + 1)
            new_tok_choice = random.random()
            if new_tok_choice < 0.4 and ALL_KEYWORDS:
                tok = random.choice(ALL_KEYWORDS)
            elif new_tok_choice < 0.7:
                tok = _rand_identifier()
            else:
                tok = random.choice(["(", ")", ",", "=", "+", "-", "*", "/", "%"])
            parts.insert(idx, tok)
        else:
            # replace a token
            idx = random.randrange(len(parts))
            if ALL_KEYWORDS and random.random() < 0.5:
                parts[idx] = random.choice(ALL_KEYWORDS)
            else:
                parts[idx] = _rand_identifier()
        return " ".join(parts)
    else:
        # character-level mutation
        s = list(stmt)
        op = random.random()
        if op < 0.33 and len(s) > 0:
            # flip a character
            idx = random.randrange(len(s))
            s[idx] = random.choice(string.printable[:94])
        elif op < 0.66:
            # insert a character
            idx = random.randrange(len(s) + 1)
            s.insert(idx, random.choice(string.printable[:94]))
        else:
            # delete a range
            if len(s) > 1:
                start = random.randrange(len(s) - 1)
                end = min(len(s), start + random.randint(1, 5))
                del s[start:end]
        return "".join(s)


def _gen_statement():
    r = random.random()
    if r < 0.5:
        return _gen_select_stmt()
    elif r < 0.65:
        return _gen_insert_stmt()
    elif r < 0.8:
        return _gen_update_stmt()
    elif r < 0.9:
        # DDL
        ddl_choice = random.random()
        if ddl_choice < 0.5:
            return _gen_create_table_stmt()
        elif ddl_choice < 0.7:
            return _gen_index_stmt()
        else:
            return _gen_drop_table_stmt()
    else:
        return _gen_random_keyword_soup()


def fuzz(parse_sql):
    global _iteration, _corpus
    _ensure_initialized()
    _iteration += 1

    # Determine batch size based on iteration
    if _iteration < 5:
        batch_size = 64
    elif _iteration < 20:
        batch_size = 128
    elif _iteration < 100:
        batch_size = 192
    else:
        batch_size = 256

    statements = []

    # Mutate existing corpus entries
    if _corpus:
        mutate_count = min(len(_corpus), batch_size // 3)
        for _ in range(mutate_count):
            base = random.choice(_corpus)
            mutated = _mutate_statement(base)
            statements.append(mutated)

    # Generate fresh statements
    while len(statements) < batch_size:
        stmt = _gen_statement()
        statements.append(stmt)
        # Maintain corpus with a bias toward more complex statements
        if len(_corpus) < _corpus_limit:
            if len(stmt) > 20 or random.random() < 0.3:
                _corpus.append(stmt)

    try:
        parse_sql(statements)
    except Exception:
        # parse_sql is expected to swallow parser errors, but guard just in case
        pass

    return True
'''
        code = dedent(base_code).replace("__KEYWORDS__", repr(keywords))
        return {"code": code}