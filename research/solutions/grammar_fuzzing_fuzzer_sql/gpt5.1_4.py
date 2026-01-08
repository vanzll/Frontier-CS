import os
import re


class Solution:
    def solve(self, resources_path: str) -> dict:
        grammar_path = os.path.join(resources_path, "sql_grammar.txt")
        grammar_text = ""
        if os.path.isfile(grammar_path):
            try:
                with open(grammar_path, "r", encoding="utf-8", errors="ignore") as f:
                    grammar_text = f.read()
            except Exception:
                grammar_text = ""
        keywords = []
        if grammar_text:
            candidates = re.findall(r"\b([A-Z][A-Z0-9_]*)\b", grammar_text)
            if candidates:
                keywords = sorted(set(candidates))
        default_keywords = [
            "SELECT", "INSERT", "UPDATE", "DELETE", "FROM", "WHERE",
            "GROUP", "BY", "HAVING", "ORDER", "LIMIT", "OFFSET",
            "JOIN", "LEFT", "RIGHT", "FULL", "OUTER", "INNER",
            "CROSS", "ON", "AS", "AND", "OR", "NOT",
            "NULL", "IS", "IN", "EXISTS", "BETWEEN", "LIKE",
            "UNION", "ALL", "DISTINCT", "CREATE", "TABLE",
            "DROP", "ALTER", "ADD", "COLUMN", "INDEX",
            "VIEW", "TRIGGER", "PRIMARY", "KEY", "FOREIGN",
            "REFERENCES", "DEFAULT", "CHECK", "UNIQUE",
            "VALUES", "INTO", "SET", "CASE", "WHEN",
            "THEN", "ELSE", "END", "BEGIN", "COMMIT",
            "ROLLBACK", "TRUE", "FALSE", "IF"
        ]
        if keywords:
            merged = set(default_keywords)
            merged.update(keywords)
            keywords = sorted(merged)
        else:
            keywords = default_keywords

        base_code = '''import random
import time
import string

KEYWORDS = __KEYWORDS__

TABLE_NAMES = [
    "users",
    "orders",
    "products",
    "t",
    "t1",
    "t2",
    "logs",
    "events",
    "metrics",
    "sessions",
    "items",
    "categories",
    "accounts",
    "payments",
    "invoices",
]

COLUMN_NAMES = [
    "id",
    "user_id",
    "order_id",
    "product_id",
    "name",
    "email",
    "price",
    "amount",
    "quantity",
    "status",
    "created_at",
    "updated_at",
    "description",
    "category",
    "is_active",
    "count",
    "total",
]

NUMERIC_TYPES = [
    "INT",
    "INTEGER",
    "BIGINT",
    "SMALLINT",
    "DECIMAL(10,2)",
    "NUMERIC",
    "FLOAT",
    "DOUBLE",
    "REAL",
]

STRING_TYPES = [
    "VARCHAR(255)",
    "VARCHAR(100)",
    "CHAR(10)",
    "TEXT",
]

OTHER_TYPES = [
    "DATE",
    "TIMESTAMP",
    "DATETIME",
    "BOOLEAN",
    "BLOB",
]

ALL_TYPES = NUMERIC_TYPES + STRING_TYPES + OTHER_TYPES

COMPARISON_OPERATORS = ["=", "<>", "!=", "<", ">", "<=", ">="]
ARITHMETIC_OPERATORS = ["+", "-", "*", "/", "%"]
UNARY_OPERATORS = ["+", "-", "NOT"]
LOGICAL_OPERATORS = ["AND", "OR"]
JOIN_TYPES = [
    "JOIN",
    "LEFT JOIN",
    "RIGHT JOIN",
    "FULL JOIN",
    "FULL OUTER JOIN",
    "INNER JOIN",
    "CROSS JOIN",
    "LEFT OUTER JOIN",
]
SET_OPERATORS = ["UNION", "UNION ALL", "INTERSECT", "EXCEPT"]
ORDER_DIRECTIONS = ["ASC", "DESC"]
BOOLEAN_LITERALS = ["TRUE", "FALSE"]

_rng = random.Random(time.time())

_MAX_CORPUS_SIZE = 1024
_BASE_BATCH_SIZE = 256

_CALL_COUNT = 0

_BASE_SEEDS = [
    "SELECT 1",
    "SELECT * FROM users",
    "SELECT id, name FROM users WHERE id = 1",
    "SELECT COUNT(*) FROM orders WHERE status = 'open'",
    "SELECT DISTINCT status FROM orders",
    "SELECT u.name, o.amount FROM users u JOIN orders o ON u.id = o.user_id",
    "SELECT name FROM products WHERE price > 10 ORDER BY price DESC",
    "SELECT category, SUM(amount) FROM orders GROUP BY category HAVING SUM(amount) > 1000",
    "INSERT INTO users (id, name, email) VALUES (1, 'alice', 'a@example.com')",
    "INSERT INTO orders (id, user_id, amount) VALUES (1, 1, 100.50)",
    "UPDATE users SET name = 'bob' WHERE id = 1",
    "DELETE FROM users WHERE id = 1",
    "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)",
    "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount DECIMAL(10,2))",
    "ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT TRUE",
    "CREATE INDEX idx_users_name ON users (name)",
    "DROP TABLE old_users",
    "BEGIN",
    "COMMIT",
    "ROLLBACK",
    "SELECT * FROM (SELECT id FROM users) sub WHERE id > 10",
    "SELECT CASE WHEN amount > 0 THEN 1 ELSE 0 END AS positive FROM orders",
    "SELECT name FROM users WHERE name LIKE 'a%'",
    "SELECT * FROM users WHERE email IS NULL",
    "SELECT * FROM users WHERE id IN (1,2,3)",
]

_CORPUS = list(_BASE_SEEDS)

def _rand():
    return _rng.random()

def _randint(a, b):
    return _rng.randint(a, b)

def _choice(seq):
    return _rng.choice(seq)

def _maybe(prob):
    return _rand() < prob

def _randomize_keyword_case(kw):
    mode = _randint(0, 2)
    if mode == 0:
        return kw.lower()
    elif mode == 1:
        return kw.upper()
    else:
        chars = []
        for c in kw:
            if c.isalpha() and _maybe(0.5):
                chars.append(c.lower())
            else:
                chars.append(c.upper())
        return "".join(chars)

def _random_identifier(max_parts=3):
    parts = []
    n_parts = 1
    if max_parts > 1:
        n_parts = _randint(1, max_parts)
    first_chars = string.ascii_letters + "_"
    later_chars = first_chars + string.digits + "$"
    for _ in range(n_parts):
        length = _randint(1, 10)
        s = _choice(first_chars)
        for _i in range(length - 1):
            s += _choice(later_chars)
        if _maybe(0.1):
            s = '"' + s + '"'
        parts.append(s)
    return ".".join(parts)

def _random_table_name():
    if _maybe(0.7):
        name = _choice(TABLE_NAMES)
    else:
        name = _random_identifier()
    if _maybe(0.3):
        schema = _choice(["public", "main", "dbo"])
        return schema + "." + name
    return name

def _random_column_name():
    if _maybe(0.7):
        return _choice(COLUMN_NAMES)
    return _random_identifier(max_parts=1)

def _random_int():
    if _maybe(0.2):
        return str(_randint(-2**31, 2**31 - 1))
    else:
        return str(_randint(-1000, 1000))

def _random_number():
    if _maybe(0.7):
        return _random_int()
    v = (_rand() - 0.5) * 1e6
    s = "{0:.3f}".format(v)
    if _maybe(0.3):
        exp = _randint(-10, 10)
        s += "e{0}".format(exp)
    return s

def _random_string_literal():
    length = _randint(0, 20)
    chars = string.ascii_letters + string.digits + " _-!@#$%^&*()[]{}:,./?|"
    s = ""
    for _ in range(length):
        c = _choice(chars)
        if c == "'":
            c = "''"
        s += c
    if _maybe(0.05):
        return "'" + s
    return "'" + s + "'"

def _random_literal():
    r = _rand()
    if r < 0.4:
        return _random_number()
    elif r < 0.8:
        return _random_string_literal()
    elif r < 0.9:
        return _choice(BOOLEAN_LITERALS)
    else:
        return "NULL"

def _simple_value_expr():
    r = _rand()
    if r < 0.4:
        return _random_column_name()
    elif r < 0.8:
        return _random_literal()
    else:
        op = _choice(UNARY_OPERATORS)
        return op + " " + _random_column_name()

def _value_expr(depth=0):
    if depth > 2:
        return _simple_value_expr()
    r = _rand()
    if r < 0.4:
        return _simple_value_expr()
    elif r < 0.8:
        left = _value_expr(depth + 1)
        right = _value_expr(depth + 1)
        op = _choice(ARITHMETIC_OPERATORS + COMPARISON_OPERATORS)
        return "(" + left + " " + op + " " + right + ")"
    else:
        funcs = [
            "COUNT",
            "SUM",
            "AVG",
            "MIN",
            "MAX",
            "LOWER",
            "UPPER",
            "COALESCE",
            "ABS",
            "ROUND",
            "LENGTH",
            "SUBSTR",
        ]
        func = _choice(funcs)
        if func in ("COUNT", "SUM", "AVG", "MIN", "MAX") and _maybe(0.2):
            arg = "*"
        else:
            arg = _simple_value_expr()
        return func + "(" + arg + ")"

def _simple_condition():
    col = _random_column_name()
    r = _rand()
    if r < 0.3:
        op = _choice(["=", "<>", "!=", "<", ">", "<=", ">="])
        return col + " " + op + " " + _value_expr()
    elif r < 0.45:
        if _maybe(0.5):
            return col + " IS NULL"
        else:
            return col + " IS NOT NULL"
    elif r < 0.7:
        neg = " NOT" if _maybe(0.3) else ""
        values = ", ".join(_random_literal() for _ in range(_randint(1, 4)))
        return col + neg + " IN (" + values + ")"
    elif r < 0.9:
        a = _random_literal()
        b = _random_literal()
        if _maybe(0.5):
            a, b = b, a
        return col + " BETWEEN " + a + " AND " + b
    else:
        pat = _random_string_literal()
        neg = " NOT" if _maybe(0.3) else ""
        return col + neg + " LIKE " + pat

def _condition_expr(depth=0):
    if depth > 2:
        return _simple_condition()
    if _maybe(0.6):
        left = _condition_expr(depth + 1)
        right = _condition_expr(depth + 1)
        op = _choice(LOGICAL_OPERATORS)
        return "(" + left + " " + op + " " + right + ")"
    else:
        return _simple_condition()

def _random_comment():
    if _maybe(0.5):
        text = "".join(_choice(string.ascii_letters + " ") for _ in range(_randint(0, 20)))
        return "-- " + text
    else:
        text = "".join(_choice(string.ascii_letters + " /*-") for _ in range(_randint(0, 30)))
        if _maybe(0.2):
            return "/*" + text
        return "/*" + text + "*/"

def _maybe_add_comments(sql):
    if _maybe(0.2):
        sql = _random_comment() + " " + sql
    if _maybe(0.2):
        sql = sql + " " + _random_comment()
    if _maybe(0.1):
        parts = sql.split(" ", 1)
        if len(parts) == 2:
            sql = parts[0] + " " + _random_comment() + " " + parts[1]
    return sql

def gen_simple_subselect():
    col = _random_column_name()
    table = _random_table_name()
    return "SELECT " + col + " FROM " + table

def gen_select_stmt():
    distinct = "DISTINCT " if _maybe(0.3) else ""
    n_cols = _randint(1, 5)
    cols = []
    for _i in range(n_cols):
        if _maybe(0.2):
            if _maybe(0.3):
                cols.append("*")
            else:
                cols.append(_value_expr())
        else:
            expr = _value_expr()
            if _maybe(0.3):
                alias = _random_identifier(max_parts=1)
                expr = expr + " AS " + alias
            cols.append(expr)
    select_list = ", ".join(cols)
    from_clause = ""
    if _maybe(0.8):
        from_clause = "FROM " + _random_table_name()
        join_parts = []
        if _maybe(0.5):
            n_joins = _randint(1, 3)
            for _j in range(n_joins):
                join_type = _choice(JOIN_TYPES)
                table = _random_table_name()
                cond = _simple_condition()
                join_parts.append(" " + join_type + " " + table + " ON " + cond)
        from_clause = from_clause + "".join(join_parts)
    where_clause = ""
    if _maybe(0.6):
        where_clause = " WHERE " + _condition_expr()
    group_by_clause = ""
    if _maybe(0.3):
        n = _randint(1, 3)
        cols_g = ", ".join(_random_column_name() for _ in range(n))
        group_by_clause = " GROUP BY " + cols_g
    having_clause = ""
    if group_by_clause and _maybe(0.5):
        having_clause = " HAVING " + _condition_expr()
    order_by_clause = ""
    if _maybe(0.5):
        n = _randint(1, 3)
        parts = []
        for _i in range(n):
            expr = _random_column_name() if _maybe(0.6) else _value_expr()
            if _maybe(0.7):
                expr = expr + " " + _choice(ORDER_DIRECTIONS)
            parts.append(expr)
        order_by_clause = " ORDER BY " + ", ".join(parts)
    limit_clause = ""
    if _maybe(0.5):
        limit = str(_randint(0, 1000))
        limit_clause = " LIMIT " + limit
        if _maybe(0.5):
            offset = str(_randint(0, 1000))
            limit_clause = limit_clause + " OFFSET " + offset
    tail = ""
    if _maybe(0.2):
        op = _choice(SET_OPERATORS)
        tail = " " + op + " " + gen_simple_subselect()
    sql = "SELECT " + distinct + select_list
    if from_clause:
        sql = sql + " " + from_clause
    sql = sql + where_clause + group_by_clause + having_clause + order_by_clause + limit_clause + tail
    if _maybe(0.3):
        sql = "(" + sql + ")"
    if _maybe(0.5):
        sql = sql + ";"
    return _maybe_add_comments(sql)

def gen_insert_stmt():
    table = _random_table_name()
    max_cols = min(5, len(COLUMN_NAMES))
    n_cols = _randint(1, max_cols) if max_cols > 0 else 1
    if COLUMN_NAMES and n_cols <= len(COLUMN_NAMES):
        cols = _rng.sample(COLUMN_NAMES, n_cols)
    else:
        cols = [_random_column_name() for _ in range(n_cols)]
    col_list = ", ".join(cols)
    n_rows = 1 if _maybe(0.7) else _randint(2, 4)
    rows = []
    for _i in range(n_rows):
        vals = []
        for _c in cols:
            vals.append(_random_literal())
        rows.append("(" + ", ".join(vals) + ")")
    values_clause = ", ".join(rows)
    sql = "INSERT INTO " + table + " (" + col_list + ") VALUES " + values_clause
    if _maybe(0.5):
        sql = sql + ";"
    return _maybe_add_comments(sql)

def gen_update_stmt():
    table = _random_table_name()
    n_sets = _randint(1, 4)
    assigns = []
    for _i in range(n_sets):
        col = _random_column_name()
        val = _value_expr()
        assigns.append(col + " = " + val)
    sql = "UPDATE " + table + " SET " + ", ".join(assigns)
    if _maybe(0.7):
        sql = sql + " WHERE " + _condition_expr()
    if _maybe(0.5):
        sql = sql + ";"
    return _maybe_add_comments(sql)

def gen_delete_stmt():
    table = _random_table_name()
    sql = "DELETE FROM " + table
    if _maybe(0.7):
        sql = sql + " WHERE " + _condition_expr()
    if _maybe(0.5):
        sql = sql + ";"
    return _maybe_add_comments(sql)

def gen_create_table_stmt():
    table = _random_table_name()
    n_cols = _randint(1, 8)
    cols = []
    for _i in range(n_cols):
        col_name = _random_column_name()
        col_type = _choice(ALL_TYPES)
        constraints = []
        if _maybe(0.3):
            constraints.append("NOT NULL")
        if _maybe(0.1):
            constraints.append("UNIQUE")
        if _maybe(0.1):
            constraints.append("DEFAULT " + _random_literal())
        col_def = col_name + " " + col_type
        if constraints:
            col_def = col_def + " " + " ".join(constraints)
        cols.append(col_def)
    if _maybe(0.3) and cols:
        pk_cols = [c.split(" ")[0] for c in cols]
        k = _randint(1, min(2, len(pk_cols)))
        pk = ", ".join(_rng.sample(pk_cols, k))
        cols.append("PRIMARY KEY (" + pk + ")")
    if _maybe(0.2) and cols:
        fk_col = cols[0].split(" ")[0]
        ref_table = _random_table_name()
        ref_col = _random_column_name()
        cols.append("FOREIGN KEY (" + fk_col + ") REFERENCES " + ref_table + " (" + ref_col + ")")
    body = ", ".join(cols)
    sql = "CREATE TABLE " + table + " (" + body + ")"
    if _maybe(0.5):
        sql = sql + ";"
    return _maybe_add_comments(sql)

def gen_create_index_stmt():
    idx_name = _random_identifier(max_parts=1)
    table = _random_table_name()
    n_cols = _randint(1, 3)
    cols = ", ".join(_random_column_name() for _i in range(n_cols))
    sql = "CREATE "
    if _maybe(0.5):
        sql = sql + "UNIQUE "
    sql = sql + "INDEX " + idx_name + " ON " + table + " (" + cols + ")"
    if _maybe(0.5):
        sql = sql + ";"
    return _maybe_add_comments(sql)

def gen_alter_table_stmt():
    table = _random_table_name()
    action_type = _randint(0, 2)
    if action_type == 0:
        col_name = _random_column_name()
        col_type = _choice(ALL_TYPES)
        sql = "ALTER TABLE " + table + " ADD COLUMN " + col_name + " " + col_type
    elif action_type == 1:
        col_name = _random_column_name()
        sql = "ALTER TABLE " + table + " DROP COLUMN " + col_name
    else:
        if _maybe(0.5):
            new_name = _random_table_name()
            sql = "ALTER TABLE " + table + " RENAME TO " + new_name
        else:
            col_name = _random_column_name()
            new_col = _random_column_name()
            sql = "ALTER TABLE " + table + " RENAME COLUMN " + col_name + " TO " + new_col
    if _maybe(0.5):
        sql = sql + ";"
    return _maybe_add_comments(sql)

def gen_drop_stmt():
    kind = _randint(0, 2)
    if kind == 0:
        sql = "DROP TABLE " + _random_table_name()
    elif kind == 1:
        sql = "DROP INDEX " + _random_identifier(max_parts=1)
    else:
        sql = "DROP VIEW " + _random_identifier(max_parts=1)
    if _maybe(0.5):
        sql = sql + ";"
    return _maybe_add_comments(sql)

def gen_transaction_stmt():
    kind = _randint(0, 3)
    if kind == 0:
        sql = "BEGIN"
    elif kind == 1:
        sql = "COMMIT"
    elif kind == 2:
        sql = "ROLLBACK"
    else:
        sql = "SAVEPOINT " + _random_identifier(max_parts=1)
    if _maybe(0.5):
        sql = sql + ";"
    return _maybe_add_comments(sql)

_TEMPLATE_GENERATORS = [
    gen_select_stmt,
    gen_insert_stmt,
    gen_update_stmt,
    gen_delete_stmt,
    gen_create_table_stmt,
    gen_create_index_stmt,
    gen_alter_table_stmt,
    gen_drop_stmt,
    gen_transaction_stmt,
]

def _random_mutation_chunk():
    r = _rand()
    if r < 0.4:
        if KEYWORDS:
            return " " + _randomize_keyword_case(_choice(KEYWORDS)) + " "
        else:
            return " " + _random_identifier() + " "
    elif r < 0.7:
        return " " + _random_literal() + " "
    else:
        return _choice(["*", ",", "(", ")", "=", "+", "-", "/", "%", ";", "?", "||"])

def mutate_statement(stmt):
    if not stmt:
        stmt = "SELECT 1"
    s = stmt
    max_len = 2000
    if len(s) > max_len:
        start = _randint(0, len(s) - max_len)
        s = s[start:start + max_len]
    n_mut = _randint(1, 3)
    for _i in range(n_mut):
        op = _randint(0, 5)
        if op == 0:
            insert_pos = _randint(0, len(s))
            chunk = _random_mutation_chunk()
            s = s[:insert_pos] + chunk + s[insert_pos:]
        elif op == 1 and len(s) > 1:
            start = _randint(0, len(s) - 1)
            end = min(len(s), start + _randint(1, 10))
            s = s[:start] + s[end:]
        elif op == 2 and len(s) > 0:
            idx = _randint(0, len(s) - 1)
            ch = s[idx]
            if ch.islower():
                ch2 = ch.upper()
            elif ch.isupper():
                ch2 = ch.lower()
            else:
                ch2 = ch
            s = s[:idx] + ch2 + s[idx + 1:]
        elif op == 3 and len(s) > 1:
            start = _randint(0, len(s) - 1)
            end = min(len(s), start + _randint(1, 10))
            seg = s[start:end]
            insert_pos = _randint(0, len(s))
            s = s[:insert_pos] + seg + s[insert_pos:]
        elif op == 4 and " " in s:
            parts = s.split(" ")
            if len(parts) >= 2:
                i = _randint(0, len(parts) - 1)
                j = _randint(0, len(parts) - 1)
                parts[i], parts[j] = parts[j], parts[i]
                s = " ".join(parts)
        else:
            if _maybe(0.5):
                s = s + " /*" + "".join(_choice(string.ascii_letters) for _ in range(_randint(0, 10)))
            else:
                s = s + " '" + "".join(_choice(string.ascii_letters) for _ in range(_randint(0, 10)))
    return s

def gen_keyword_soup_stmt():
    length = _randint(3, 25)
    tokens = []
    for _i in range(length):
        r = _rand()
        if r < 0.5 and KEYWORDS:
            kw = _choice(KEYWORDS)
            tokens.append(_randomize_keyword_case(kw))
        elif r < 0.7:
            tokens.append(_random_identifier())
        elif r < 0.85:
            tokens.append(_random_literal())
        else:
            tokens.append(_choice(["*", ",", "(", ")", "=", "+", "-", "/", "%", ";"]))
    sql = " ".join(tokens)
    if _maybe(0.3):
        sql = _maybe_add_comments(sql)
    if _maybe(0.3):
        count = _randint(1, 3)
        for _i in range(count):
            sql = sql.replace(" ", "\\n", 1)
    return sql

def generate_statement():
    r = _rand()
    if r < 0.6:
        gen = _choice(_TEMPLATE_GENERATORS)
        try:
            return gen()
        except Exception:
            return gen_keyword_soup_stmt()
    elif r < 0.85 and _CORPUS:
        base = _choice(_CORPUS)
        return mutate_statement(base)
    else:
        return gen_keyword_soup_stmt()

def fuzz(parse_sql):
    global _CALL_COUNT, _CORPUS
    _CALL_COUNT += 1
    batch_size = _BASE_BATCH_SIZE
    if _CALL_COUNT <= 10:
        batch_size = 128 + _CALL_COUNT * 16
    elif _CALL_COUNT > 200:
        batch_size = 512
    statements = []
    for _i in range(batch_size):
        stmt = generate_statement()
        statements.append(stmt)
    try:
        parse_sql(statements)
    except Exception:
        pass
    if len(_CORPUS) < _MAX_CORPUS_SIZE:
        step = max(1, len(statements) // 4)
        for idx in range(0, len(statements), step):
            if len(_CORPUS) >= _MAX_CORPUS_SIZE:
                break
            _CORPUS.append(statements[idx])
    else:
        step = max(1, len(statements) // 8)
        for idx in range(0, len(statements), step):
            replace_index = _randint(0, _MAX_CORPUS_SIZE - 1)
            _CORPUS[replace_index] = statements[idx]
    return True
'''
        code = base_code.replace("__KEYWORDS__", repr(keywords))
        return {"code": code}