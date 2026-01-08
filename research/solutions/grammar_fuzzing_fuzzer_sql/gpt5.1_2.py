import textwrap


class Solution:
    def solve(self, resources_path: str) -> dict:
        code = textwrap.dedent(r'''
            import random
            import string
            import time


            _rng = random.Random()

            _start_time = None
            _batch_size = 200
            _time_per_stmt = None

            _TARGET_CALL_TIME = 0.2  # seconds per parse_sql call (approximate)
            _MIN_BATCH_SIZE = 50
            _MAX_BATCH_SIZE = 4000

            _known_tables = [
                "users", "orders", "products", "t", "foo", "bar", "baz"
            ]
            _known_columns = [
                "id", "name", "value", "price",
                "created_at", "updated_at",
                "flag", "count", "status",
                "description", "qty", "amount"
            ]

            _corpus = []
            _MAX_CORPUS = 512

            _SQL_TYPES = [
                "INT",
                "INTEGER",
                "SMALLINT",
                "BIGINT",
                "TINYINT",
                "VARCHAR(10)",
                "VARCHAR(255)",
                "CHAR(20)",
                "TEXT",
                "DATE",
                "TIME",
                "TIMESTAMP",
                "NUMERIC(10,2)",
                "DECIMAL(10,3)",
                "REAL",
                "DOUBLE",
                "DOUBLE PRECISION",
                "BOOLEAN",
                "BLOB"
            ]

            _AGG_FUNCS = ["COUNT", "SUM", "AVG", "MIN", "MAX"]
            _SCALAR_FUNCS = [
                "ABS", "LOWER", "UPPER", "LENGTH",
                "SUBSTR", "SUBSTRING",
                "ROUND",
                "TRIM", "LTRIM", "RTRIM",
                "COALESCE", "IFNULL", "NULLIF",
                "RANDOM", "HEX", "QUOTE"
            ]

            _BINARY_ARITH_OPS = ["+", "-", "*", "/", "%", "||"]
            _COMPARE_OPS = ["=", "<>", "!=", "<", ">", "<=", ">=", "IS", "IS NOT", "LIKE", "GLOB", "MATCH", "REGEXP"]
            _LOGICAL_OPS = ["AND", "OR"]


            def _random_identifier(prefix: str = "") -> str:
                length = _rng.randint(1, 8)
                letters = string.ascii_lowercase + string.ascii_uppercase + "_"
                first = _rng.choice(letters)
                others = "".join(_rng.choice(letters + string.digits + "_$") for _ in range(length - 1))
                return prefix + first + others


            def _maybe_quote_identifier(name: str) -> str:
                r = _rng.random()
                if r < 0.2:
                    return '"' + name.replace('"', '""') + '"'
                elif r < 0.3:
                    return "`" + name.replace("`", "``") + "`"
                elif r < 0.35:
                    return "[" + name.replace("]", "]]") + "]"
                else:
                    return name


            def _gen_table_name() -> str:
                if _known_tables and _rng.random() < 0.7:
                    name = _rng.choice(_known_tables)
                else:
                    name = _random_identifier("t_")
                    if len(_known_tables) < 100:
                        _known_tables.append(name)
                return name


            def _gen_column_name() -> str:
                if _known_columns and _rng.random() < 0.7:
                    name = _rng.choice(_known_columns)
                else:
                    name = _random_identifier("c_")
                    if len(_known_columns) < 200:
                        _known_columns.append(name)
                return name


            def _gen_number_literal() -> str:
                k = _rng.random()
                if k < 0.5:
                    val = _rng.randint(-2 ** 31, 2 ** 31 - 1)
                    return str(val)
                elif k < 0.75:
                    val = (_rng.random() - 0.5) * 1e6
                    return "%.5f" % val
                elif k < 0.9:
                    base = _rng.uniform(0.1, 1000.0)
                    exp = _rng.randint(-10, 10)
                    return "%fe%d" % (base, exp)
                else:
                    val = _rng.randint(0, 2 ** 32 - 1)
                    return "0x%x" % val


            def _gen_string_literal() -> str:
                # Sometimes generate unclosed/invalid strings to stress tokenizer
                if _rng.random() < 0.15:
                    length = _rng.randint(0, 20)
                    s = []
                    for _ in range(length):
                        c = _rng.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-")
                        if c == "'":
                            s.append("''")
                        else:
                            s.append(c)
                    return "'" + "".join(s)
                length = _rng.randint(0, 20)
                s = []
                for _ in range(length):
                    c = _rng.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-.,/\\")
                    if c == "'":
                        s.append("''")
                    else:
                        s.append(c)
                body = "".join(s)
                return "'" + body + "'"


            def _gen_literal() -> str:
                r = _rng.random()
                if r < 0.4:
                    return _gen_number_literal()
                elif r < 0.8:
                    return _gen_string_literal()
                elif r < 0.9:
                    return "NULL"
                elif r < 0.95:
                    return _rng.choice(["TRUE", "FALSE"])
                else:
                    return _rng.choice(["CURRENT_TIMESTAMP", "CURRENT_DATE", "CURRENT_TIME"])


            def _gen_column_ref() -> str:
                col = _gen_column_name()
                if _rng.random() < 0.3:
                    table = _gen_table_name()
                    return _maybe_quote_identifier(table) + "." + _maybe_quote_identifier(col)
                else:
                    return _maybe_quote_identifier(col)


            def _gen_simple_expr() -> str:
                r = _rng.random()
                if r < 0.4:
                    return _gen_literal()
                elif r < 0.8:
                    return _gen_column_ref()
                else:
                    if _rng.random() < 0.5:
                        return "?"
                    else:
                        return "?" + str(_rng.randint(1, 10))


            def _gen_expr(depth: int = 0) -> str:
                if depth > 3:
                    return _gen_simple_expr()
                r = _rng.random()
                if r < 0.3:
                    return _gen_simple_expr()
                elif r < 0.55:
                    left = _gen_expr(depth + 1)
                    right = _gen_expr(depth + 1)
                    op_pool = _BINARY_ARITH_OPS + _COMPARE_OPS + _LOGICAL_OPS
                    op = _rng.choice(op_pool)
                    return "(" + left + " " + op + " " + right + ")"
                elif r < 0.65:
                    op = _rng.choice(["+", "-", "NOT", "~"])
                    inner = _gen_expr(depth + 1)
                    return op + " " + inner
                elif r < 0.78:
                    func = _rng.choice(_AGG_FUNCS + _SCALAR_FUNCS + ["CAST"])
                    if func == "CAST":
                        expr = _gen_expr(depth + 1)
                        typ = _rng.choice(_SQL_TYPES)
                        return "CAST(" + expr + " AS " + typ + ")"
                    else:
                        if func == "COUNT" and _rng.random() < 0.3:
                            args = ["*"]
                        else:
                            argc = _rng.randint(1, 3)
                            args = [_gen_expr(depth + 1) for _ in range(argc)]
                        return func + "(" + ", ".join(args) + ")"
                elif r < 0.88:
                    base_expr = ""
                    if _rng.random() < 0.5:
                        base_expr = " " + _gen_expr(depth + 1)
                    parts = ["CASE" + base_expr]
                    num_whens = _rng.randint(1, 3)
                    for _ in range(num_whens):
                        parts.append("WHEN " + _gen_expr(depth + 1) + " THEN " + _gen_expr(depth + 1))
                    if _rng.random() < 0.5:
                        parts.append("ELSE " + _gen_expr(depth + 1))
                    parts.append("END")
                    return " ".join(parts)
                else:
                    base = _gen_expr(depth + 1)
                    choice = _rng.random()
                    if choice < 0.33:
                        lo = _gen_expr(depth + 1)
                        hi = _gen_expr(depth + 1)
                        maybe_not = " NOT" if _rng.random() < 0.5 else ""
                        return "(" + base + maybe_not + " BETWEEN " + lo + " AND " + hi + ")"
                    elif choice < 0.66:
                        like = _gen_literal()
                        maybe_not = " NOT" if _rng.random() < 0.5 else ""
                        esc = ""
                        if _rng.random() < 0.3:
                            esc = " ESCAPE " + _gen_string_literal()
                        return "(" + base + maybe_not + " LIKE " + like + esc + ")"
                    else:
                        maybe_not = " NOT" if _rng.random() < 0.5 else ""
                        if depth > 1 or _rng.random() < 0.5:
                            count = _rng.randint(1, 4)
                            items = [_gen_expr(depth + 1) for _ in range(count)]
                            return "(" + base + maybe_not + " IN (" + ", ".join(items) + "))"
                        else:
                            sub = _gen_select(depth + 1)
                            return "(" + base + maybe_not + " IN (" + sub + "))"


            def _gen_table_factor(depth: int) -> str:
                if depth < 2 and _rng.random() < 0.2:
                    sub = _gen_select(depth + 1)
                    alias = _maybe_quote_identifier(_random_identifier("sub_"))
                    return "(" + sub + ") AS " + alias
                else:
                    table = _gen_table_name()
                    alias = ""
                    if _rng.random() < 0.6:
                        alias = " AS " + _maybe_quote_identifier(_random_identifier("t_"))
                    return _maybe_quote_identifier(table) + alias


            def _gen_from_clause(depth: int) -> str:
                num = _rng.randint(1, 3)
                if num == 1:
                    return _gen_table_factor(depth)
                current = _gen_table_factor(depth)
                for _ in range(num - 1):
                    join_type = _rng.choice([
                        "JOIN", "INNER JOIN",
                        "LEFT JOIN", "LEFT OUTER JOIN",
                        "RIGHT JOIN", "RIGHT OUTER JOIN",
                        "FULL JOIN", "FULL OUTER JOIN",
                        "CROSS JOIN"
                    ])
                    right = _gen_table_factor(depth)
                    clause = ""
                    r = _rng.random()
                    if r < 0.7:
                        clause = " ON " + _gen_expr(depth + 1)
                    elif r < 0.9:
                        count = _rng.randint(1, 2)
                        cols = [_maybe_quote_identifier(_gen_column_name()) for _ in range(count)]
                        clause = " USING (" + ", ".join(cols) + ")"
                    current = "(" + current + " " + join_type + " " + right + clause + ")"
                return current


            def _gen_select_core(depth: int = 0) -> str:
                parts = ["SELECT"]
                if _rng.random() < 0.3:
                    parts.append(_rng.choice(["DISTINCT", "ALL"]))
                n_items = _rng.randint(1, 4)
                items = []
                for _ in range(n_items):
                    r = _rng.random()
                    if r < 0.2:
                        items.append("*")
                    elif r < 0.35:
                        table = _gen_table_name()
                        items.append(_maybe_quote_identifier(table) + ".*")
                    else:
                        expr = _gen_expr(depth + 1)
                        if _rng.random() < 0.4:
                            alias = _maybe_quote_identifier(_random_identifier("c_"))
                            expr = expr + " AS " + alias
                        items.append(expr)
                parts.append(", ".join(items))
                if _rng.random() < 0.9 or depth == 0:
                    from_clause = _gen_from_clause(depth)
                    if from_clause:
                        parts.append("FROM " + from_clause)
                if _rng.random() < 0.7:
                    parts.append("WHERE " + _gen_expr(depth + 1))
                if _rng.random() < 0.5:
                    cnt = _rng.randint(1, 3)
                    gb = [_gen_expr(depth + 1) for _ in range(cnt)]
                    parts.append("GROUP BY " + ", ".join(gb))
                    if _rng.random() < 0.5:
                        parts.append("HAVING " + _gen_expr(depth + 1))
                if _rng.random() < 0.6:
                    cnt = _rng.randint(1, 3)
                    ob_items = []
                    for _ in range(cnt):
                        expr = _gen_expr(depth + 1)
                        direction = _rng.choice(["ASC", "DESC"]) if _rng.random() < 0.7 else ""
                        nulls = _rng.choice(["NULLS FIRST", "NULLS LAST"]) if _rng.random() < 0.3 else ""
                        s = expr
                        if direction:
                            s += " " + direction
                        if nulls:
                            s += " " + nulls
                        ob_items.append(s)
                    parts.append("ORDER BY " + ", ".join(ob_items))
                if _rng.random() < 0.5:
                    if _rng.random() < 0.5:
                        limit = str(_rng.randint(0, 100))
                        clause = "LIMIT " + limit
                        if _rng.random() < 0.5:
                            offset = str(_rng.randint(0, 100))
                            clause += " OFFSET " + offset
                    else:
                        offset = str(_rng.randint(0, 100))
                        limit = str(_rng.randint(0, 100))
                        clause = "LIMIT " + offset + ", " + limit
                    parts.append(clause)
                return " ".join(parts)


            def _gen_select(depth: int = 0) -> str:
                base = _gen_select_core(depth)
                if depth < 2 and _rng.random() < 0.4:
                    num_extra = _rng.randint(1, 2)
                    for _ in range(num_extra):
                        op = _rng.choice([
                            "UNION", "UNION ALL",
                            "EXCEPT", "EXCEPT ALL",
                            "INTERSECT", "INTERSECT ALL"
                        ])
                        right = _gen_select_core(depth + 1)
                        base = "(" + base + ") " + op + " (" + right + ")"
                return base


            def _gen_insert() -> str:
                table = _gen_table_name()
                num_cols = _rng.randint(1, 4)
                cols = [_maybe_quote_identifier(_gen_column_name()) for _ in range(num_cols)]
                if _rng.random() < 0.6:
                    rows = []
                    num_rows = _rng.randint(1, 3)
                    for _ in range(num_rows):
                        vals = [_gen_expr() for _ in range(num_cols)]
                        rows.append("(" + ", ".join(vals) + ")")
                    return "INSERT INTO " + _maybe_quote_identifier(table) + " (" + ", ".join(cols) + ") VALUES " + ", ".join(rows)
                else:
                    sub = _gen_select()
                    return "INSERT INTO " + _maybe_quote_identifier(table) + " (" + ", ".join(cols) + ") " + sub


            def _gen_update() -> str:
                table = _gen_table_name()
                num_cols = _rng.randint(1, 4)
                assignments = []
                for _ in range(num_cols):
                    col = _maybe_quote_identifier(_gen_column_name())
                    if _rng.random() < 0.3:
                        val = "DEFAULT"
                    else:
                        val = _gen_expr()
                    assignments.append(col + " = " + val)
                parts = ["UPDATE", _maybe_quote_identifier(table), "SET", ", ".join(assignments)]
                if _rng.random() < 0.7:
                    parts.append("WHERE " + _gen_expr())
                if _rng.random() < 0.3:
                    cnt = _rng.randint(1, 3)
                    ob = [_gen_expr() for _ in range(cnt)]
                    parts.append("ORDER BY " + ", ".join(ob))
                if _rng.random() < 0.3:
                    parts.append("LIMIT " + str(_rng.randint(1, 100)))
                return " ".join(parts)


            def _gen_delete() -> str:
                table = _gen_table_name()
                parts = ["DELETE FROM", _maybe_quote_identifier(table)]
                if _rng.random() < 0.7:
                    parts.append("WHERE " + _gen_expr())
                if _rng.random() < 0.3:
                    cnt = _rng.randint(1, 3)
                    ob = [_gen_expr() for _ in range(cnt)]
                    parts.append("ORDER BY " + ", ".join(ob))
                if _rng.random() < 0.3:
                    parts.append("LIMIT " + str(_rng.randint(1, 100)))
                return " ".join(parts)


            def _gen_column_def() -> str:
                col = _maybe_quote_identifier(_gen_column_name())
                typ = _rng.choice(_SQL_TYPES)
                parts = [col, typ]
                if _rng.random() < 0.3:
                    parts.append("NOT NULL")
                if _rng.random() < 0.2:
                    parts.append("PRIMARY KEY")
                    if _rng.random() < 0.5:
                        parts.append("AUTOINCREMENT")
                if _rng.random() < 0.2:
                    parts.append("UNIQUE")
                if _rng.random() < 0.3:
                    parts.append("DEFAULT " + _gen_literal())
                if _rng.random() < 0.2:
                    parts.append("CHECK (" + _gen_expr() + ")")
                return " ".join(parts)


            def _gen_table_constraint() -> str:
                kind = _rng.choice(["PRIMARY KEY", "UNIQUE", "CHECK", "FOREIGN KEY"])
                if kind in ("PRIMARY KEY", "UNIQUE"):
                    num = _rng.randint(1, 3)
                    cols = [_maybe_quote_identifier(_gen_column_name()) for _ in range(num)]
                    return kind + " (" + ", ".join(cols) + ")"
                elif kind == "CHECK":
                    return "CHECK (" + _gen_expr() + ")"
                else:
                    num = _rng.randint(1, 2)
                    cols = [_maybe_quote_identifier(_gen_column_name()) for _ in range(num)]
                    ref_table = _gen_table_name()
                    ref_cols = [_maybe_quote_identifier(_gen_column_name()) for _ in range(num)]
                    clause = "FOREIGN KEY (" + ", ".join(cols) + ") REFERENCES " + _maybe_quote_identifier(ref_table) + " (" + ", ".join(ref_cols) + ")"
                    actions = []
                    if _rng.random() < 0.5:
                        actions.append("ON DELETE " + _rng.choice(["CASCADE", "SET NULL", "SET DEFAULT", "RESTRICT", "NO ACTION"]))
                    if _rng.random() < 0.5:
                        actions.append("ON UPDATE " + _rng.choice(["CASCADE", "SET NULL", "SET DEFAULT", "RESTRICT", "NO ACTION"]))
                    if actions:
                        clause += " " + " ".join(actions)
                    return clause


            def _gen_create_table() -> str:
                table = _gen_table_name()
                temp = "TEMPORARY " if _rng.random() < 0.3 else ""
                if_not = "IF NOT EXISTS " if _rng.random() < 0.5 else ""
                if _rng.random() < 0.2:
                    sub = _gen_select()
                    return "CREATE " + temp + "TABLE " + if_not + _maybe_quote_identifier(table) + " AS " + sub
                num_cols = _rng.randint(1, 6)
                elements = [_gen_column_def() for _ in range(num_cols)]
                for _ in range(_rng.randint(0, 2)):
                    elements.append(_gen_table_constraint())
                body = ", ".join(elements)
                tail = ""
                w = []
                if _rng.random() < 0.3:
                    w.append("WITHOUT ROWID")
                if w:
                    tail = " " + " ".join(w)
                return "CREATE " + temp + "TABLE " + if_not + _maybe_quote_identifier(table) + " (" + body + ")" + tail


            def _gen_alter_table() -> str:
                table = _gen_table_name()
                choice = _rng.randint(0, 2)
                if choice == 0:
                    col_def = _gen_column_def()
                    action = "ADD COLUMN " + col_def
                elif choice == 1:
                    new_name = _maybe_quote_identifier(_gen_table_name())
                    action = "RENAME TO " + new_name
                else:
                    old_col = _maybe_quote_identifier(_gen_column_name())
                    new_col = _maybe_quote_identifier(_gen_column_name())
                    action = "RENAME COLUMN " + old_col + " TO " + new_col
                return "ALTER TABLE " + _maybe_quote_identifier(table) + " " + action


            def _gen_create_index() -> str:
                unique = "UNIQUE " if _rng.random() < 0.4 else ""
                if_not = "IF NOT EXISTS " if _rng.random() < 0.5 else ""
                name = _maybe_quote_identifier(_random_identifier("idx_"))
                table = _gen_table_name()
                num_cols = _rng.randint(1, 3)
                cols = []
                for _ in range(num_cols):
                    col = _maybe_quote_identifier(_gen_column_name())
                    if _rng.random() < 0.5:
                        col += " " + _rng.choice(["ASC", "DESC"])
                    cols.append(col)
                stmt = "CREATE " + unique + "INDEX " + if_not + name + " ON " + _maybe_quote_identifier(table) + " (" + ", ".join(cols) + ")"
                if _rng.random() < 0.4:
                    stmt += " WHERE " + _gen_expr()
                return stmt


            def _gen_drop_statement() -> str:
                kind = _rng.randint(0, 2)
                if kind == 0:
                    tbl = _gen_table_name()
                    if _rng.random() < 0.5:
                        return "DROP TABLE IF EXISTS " + _maybe_quote_identifier(tbl)
                    else:
                        return "DROP TABLE " + _maybe_quote_identifier(tbl)
                elif kind == 1:
                    idx = _maybe_quote_identifier(_random_identifier("idx_"))
                    if _rng.random() < 0.5:
                        return "DROP INDEX IF EXISTS " + idx
                    else:
                        return "DROP INDEX " + idx
                else:
                    view = _maybe_quote_identifier(_random_identifier("v_"))
                    if _rng.random() < 0.5:
                        return "DROP VIEW IF EXISTS " + view
                    else:
                        return "DROP VIEW " + view


            def _gen_transaction() -> str:
                k = _rng.randint(0, 3)
                if k == 0:
                    return "BEGIN TRANSACTION"
                elif k == 1:
                    return "COMMIT"
                elif k == 2:
                    return "ROLLBACK"
                else:
                    return "SAVEPOINT " + _maybe_quote_identifier(_random_identifier("sp_"))


            def _gen_comment_statement() -> str:
                if _rng.random() < 0.5:
                    length = _rng.randint(0, 40)
                    body = "".join(_rng.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ") for _ in range(length))
                    return "-- " + body
                else:
                    length = _rng.randint(0, 60)
                    body = "".join(_rng.choice("abcdefghijklmnopqrstuvwxyz0123456789*-/ ") for _ in range(length))
                    if _rng.random() < 0.1:
                        return "/* " + body
                    else:
                        return "/* " + body + " */"


            def _gen_pragma_or_set() -> str:
                name = _maybe_quote_identifier(_random_identifier("pragma_"))
                value = _gen_literal()
                if _rng.random() < 0.5:
                    return "PRAGMA " + name + " = " + value
                else:
                    return "SET " + name + " = " + value


            def _gen_random_garbage() -> str:
                length = _rng.randint(1, 40)
                body = "".join(_rng.choice("!@#$%^&*()_+-=[]{}|;:,<>/?\\") for _ in range(length))
                return body


            def _gen_misc_statement() -> str:
                r = _rng.random()
                if r < 0.25:
                    return _gen_transaction()
                elif r < 0.5:
                    return _gen_comment_statement()
                elif r < 0.75:
                    return _gen_pragma_or_set()
                else:
                    return _gen_random_garbage()


            def _gen_statement() -> str:
                r = _rng.random()
                if r < 0.35:
                    return _gen_select()
                elif r < 0.5:
                    return _gen_insert()
                elif r < 0.62:
                    return _gen_update()
                elif r < 0.74:
                    return _gen_delete()
                elif r < 0.82:
                    return _gen_create_table()
                elif r < 0.88:
                    return _gen_alter_table()
                elif r < 0.93:
                    return _gen_create_index()
                elif r < 0.97:
                    return _gen_drop_statement()
                else:
                    return _gen_misc_statement()


            def _maybe_embed_comment(stmt: str) -> str:
                if not stmt:
                    return stmt
                if _rng.random() < 0.3:
                    if _rng.random() < 0.5:
                        inner_len = _rng.randint(0, 20)
                        body = "".join(_rng.choice("abc123*/ ") for _ in range(inner_len))
                        comment = "/*" + body + "*/"
                    else:
                        inner_len = _rng.randint(0, 20)
                        body = "".join(_rng.choice("xyz789 ") for _ in range(inner_len))
                        comment = "--" + body + "\n"
                    pos = _rng.randint(0, len(stmt))
                    stmt = stmt[:pos] + comment + stmt[pos:]
                return stmt


            def _mutate_statement(s: str) -> str:
                if not s:
                    return _gen_statement()
                max_len = 500
                for _ in range(_rng.randint(1, 3)):
                    op = _rng.randint(0, 3)
                    if op == 0 and len(s) < max_len:
                        pos = _rng.randint(0, len(s))
                        insert = _rng.choice([
                            " NULL ",
                            " 0 ",
                            " 1 ",
                            " AND 1=1 ",
                            " OR 1=0 ",
                            " /*mut*/ ",
                            " --mut\n",
                            " , ",
                            " ( ",
                            " ) ",
                            " == ",
                            " != ",
                            " <> ",
                        ])
                        s = s[:pos] + insert + s[pos:]
                    elif op == 1 and len(s) > 1:
                        start = _rng.randint(0, len(s) - 1)
                        end = _rng.randint(start + 1, min(len(s), start + 1 + _rng.randint(1, 10)))
                        s = s[:start] + s[end:]
                    elif op == 2 and len(s) > 1:
                        start = _rng.randint(0, len(s) - 1)
                        end = _rng.randint(start + 1, min(len(s), start + 1 + _rng.randint(1, 12)))
                        replacement = _rng.choice([
                            _gen_literal(),
                            _gen_column_ref(),
                            _gen_table_name(),
                            _random_identifier(),
                            _rng.choice(["=", "<>", "!=", "<", ">", "<=", ">=", "LIKE", "IS", "IS NOT"])
                        ])
                        s = s[:start] + replacement + s[end:]
                    elif op == 3 and len(s) > 1:
                        start = _rng.randint(0, len(s) - 1)
                        end = _rng.randint(start + 1, min(len(s), start + 1 + _rng.randint(1, 15)))
                        segment = s[start:end]
                        if _rng.random() < 0.5:
                            segment = segment.upper()
                        else:
                            segment = segment.lower()
                        s = s[:start] + segment + s[end:]
                if _rng.random() < 0.1 and "'" in s:
                    idx = s.find("'")
                    s = s[:idx] + s[idx + 1:]
                return s


            def _generate_batch(n: int):
                res = []
                for _ in range(n):
                    if _corpus and _rng.random() < 0.5:
                        base = _rng.choice(_corpus)
                        stmt = _mutate_statement(base)
                    else:
                        stmt = _gen_statement()
                    stmt = _maybe_embed_comment(stmt)
                    if _rng.random() < 0.8 and not stmt.rstrip().endswith(";"):
                        stmt = stmt.rstrip() + ";"
                    if _rng.random() < 0.3:
                        prefix = "\n" * _rng.randint(0, 2) + " " * _rng.randint(0, 4)
                        suffix = " " * _rng.randint(0, 4) + "\n" * _rng.randint(0, 2)
                        stmt = prefix + stmt + suffix
                    res.append(stmt)
                    if 0 < len(stmt) < 400:
                        if len(_corpus) < _MAX_CORPUS:
                            _corpus.append(stmt)
                        elif _rng.random() < 0.1:
                            _corpus[_rng.randint(0, len(_corpus) - 1)] = stmt
                return res


            def fuzz(parse_sql):
                global _start_time, _batch_size, _time_per_stmt
                if _start_time is None:
                    _start_time = time.time()

                bs = int(_batch_size)
                if bs < _MIN_BATCH_SIZE:
                    bs = _MIN_BATCH_SIZE
                elif bs > _MAX_BATCH_SIZE:
                    bs = _MAX_BATCH_SIZE

                stmts = _generate_batch(bs)

                t0 = time.time()
                try:
                    parse_sql(stmts)
                except Exception:
                    # parse_sql is documented to catch its own exceptions, but be defensive
                    pass
                t1 = time.time()
                dt = t1 - t0

                if dt > 0 and bs > 0:
                    per_stmt = dt / float(bs)
                    if _time_per_stmt is None:
                        _time_per_stmt = per_stmt
                    else:
                        _time_per_stmt = 0.7 * _time_per_stmt + 0.3 * per_stmt
                    if _time_per_stmt > 0:
                        target_bs = int(_TARGET_CALL_TIME / _time_per_stmt)
                        if target_bs < _MIN_BATCH_SIZE:
                            target_bs = _MIN_BATCH_SIZE
                        elif target_bs > _MAX_BATCH_SIZE:
                            target_bs = _MAX_BATCH_SIZE
                        _batch_size = int(0.7 * _batch_size + 0.3 * target_bs)

                return True
        ''')
        return {"code": code}