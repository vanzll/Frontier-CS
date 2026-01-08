import typing


class Solution:
    def solve(self, resources_path: str) -> dict:
        code = '''import random
import string
import time

_SQL_KEYWORDS = [
    "SELECT", "INSERT", "UPDATE", "DELETE", "FROM", "WHERE", "GROUP", "BY",
    "HAVING", "ORDER", "LIMIT", "OFFSET", "JOIN", "LEFT", "RIGHT", "FULL",
    "OUTER", "INNER", "ON", "AS", "AND", "OR", "NOT", "NULL", "IS", "IN",
    "EXISTS", "CASE", "WHEN", "THEN", "ELSE", "END", "DISTINCT", "UNION",
    "ALL", "INTERSECT", "EXCEPT", "CREATE", "TABLE", "INDEX", "VIEW",
    "TRIGGER", "DROP", "ALTER", "ADD", "COLUMN", "PRIMARY", "KEY", "FOREIGN",
    "REFERENCES", "CHECK", "DEFAULT", "UNIQUE", "BEGIN", "COMMIT",
    "ROLLBACK", "TRANSACTION"
]

_SQL_TYPES = [
    "INT", "INTEGER", "SMALLINT", "BIGINT", "REAL", "DOUBLE", "FLOAT",
    "NUMERIC(10,2)", "DECIMAL(10,2)", "CHAR(10)", "VARCHAR(255)", "TEXT",
    "DATE", "TIME", "TIMESTAMP", "BOOLEAN", "BLOB"
]

_SQL_AGG_FUNCS = ["COUNT", "SUM", "AVG", "MIN", "MAX"]

_SQL_SCALAR_FUNCS = [
    "ABS", "ROUND", "UPPER", "LOWER", "LENGTH", "SUBSTR", "COALESCE",
    "TRIM", "RANDOM"
]

_SQL_BINARY_OPS = [
    "+", "-", "*", "/", "%", "||",
    "=", "<>", "!=", "<", "<=", ">", ">=",
    "AND", "OR", "LIKE"
]

_SQL_UNARY_OPS = ["NOT", "-", "+"]


class _FuzzerState:
    def __init__(self):
        self.rng = random.Random()
        self.rng.seed(int(time.time() * 1000) & 0x7FFFFFFF)
        self.call_count = 0
        self.total_statements = 0
        self.batch_size = 200
        self.max_batch_size = 600
        self.min_batch_size = 30
        self.avg_call_duration = 0.0
        self.smoothing = 0.2
        self.max_stmt_len = 1500
        self.max_expr_depth = 3
        self.corpus = []
        self.max_corpus = 1500
        self.initialized = False
        self.start_time = None
        self._init_seed_corpus()

    def _init_seed_corpus(self):
        seeds = [
            "SELECT 1",
            "SELECT * FROM users",
            "SELECT id, name FROM customers WHERE age > 30",
            "SELECT COUNT(*) FROM orders",
            "INSERT INTO logs (id, message) VALUES (1, 'hello')",
            "UPDATE products SET price = price * 1.1 WHERE category = 'books'",
            "DELETE FROM sessions WHERE last_access < NOW() - INTERVAL '1 day'",
            "CREATE TABLE t (id INT PRIMARY KEY, value TEXT)",
            "CREATE INDEX idx_t_value ON t (value)",
            "ALTER TABLE t ADD COLUMN created_at TIMESTAMP",
            "DROP TABLE IF EXISTS old_table",
            "BEGIN TRANSACTION",
            "COMMIT",
            "ROLLBACK",
            "CREATE VIEW v AS SELECT id, value FROM t WHERE value IS NOT NULL",
            "SELECT * FROM t WHERE id IN (1,2,3,4)",
            "SELECT a.id, b.name FROM a JOIN b ON a.bid = b.id",
            "SELECT id, SUM(amount) FROM payments GROUP BY id HAVING SUM(amount) > 100",
            "SELECT name FROM employees ORDER BY name DESC LIMIT 10 OFFSET 5",
            "SELECT CASE WHEN score > 90 THEN 'A' ELSE 'B' END AS grade FROM exams",
            "INSERT INTO t DEFAULT VALUES",
            "UPDATE t SET value = NULL WHERE id = 10",
            "DELETE FROM t",
            "SELECT DISTINCT category FROM products",
            "SELECT * FROM orders WHERE date BETWEEN '2020-01-01' AND '2020-12-31'",
            "SELECT * FROM logs WHERE message LIKE '%error%'",
            "SELECT * FROM t WHERE value IS NULL OR value = ''",
            "SELECT * FROM a LEFT JOIN b ON a.id = b.aid",
            "SELECT * FROM a RIGHT JOIN b ON a.id = b.aid",
            "SELECT * FROM t ORDER BY 1, 2",
            "SELECT * FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.id = t.uid)",
            "CREATE TABLE nums (n INT, m INT, PRIMARY KEY (n, m))",
            "ALTER TABLE nums RENAME TO nums_old",
            "DROP INDEX IF EXISTS idx_t_value",
            "CREATE UNIQUE INDEX idx_unique_value ON t (value)",
            "CREATE TABLE complex (id INT, data BLOB, created DATE DEFAULT CURRENT_DATE)"
        ]
        for s in seeds:
            if len(self.corpus) < self.max_corpus:
                self.corpus.append(s)

    def _rand_identifier(self, prefix=""):
        r = self.rng
        length = r.randint(1, 8)
        chars = []
        for _ in range(length):
            ch = r.choice(string.ascii_lowercase + string.digits + "_")
            chars.append(ch)
        name = prefix + "".join(chars)
        style = r.randint(0, 5)
        if style == 0:
            return name
        elif style == 1:
            return name.upper()
        elif style == 2:
            return '"' + name + '"'
        elif style == 3:
            return "`" + name + "`"
        elif style == 4:
            return "[" + name + "]"
        else:
            kw = r.choice(_SQL_KEYWORDS)
            if r.random() < 0.5:
                return '"' + kw + '"'
            else:
                return "`" + kw + "`"

    def _rand_integer(self):
        r = self.rng
        if r.random() < 0.2:
            return str(r.choice([0, -1, 1, 2**31 - 1, -2**31]))
        value = r.randint(-10**9, 10**9)
        return str(value)

    def _rand_float(self):
        r = self.rng
        value = (r.random() - 0.5) * 1e6
        s = "{:.6f}".format(value)
        if r.random() < 0.3 and value >= 0:
            s = "+" + s
        return s

    def _rand_string_literal(self):
        r = self.rng
        length = r.randint(0, 20)
        chars = []
        for _ in range(length):
            ch_type = r.random()
            if ch_type < 0.7:
                ch = r.choice(string.ascii_letters + string.digits + " _-")
            elif ch_type < 0.9:
                ch = r.choice(["\\n", "\\t", "\\r"])
            else:
                ch = r.choice(["'", "\"", "\\\\"])
            chars.append(ch)
        s = "".join(chars)
        s = s.replace("\\\\", "\\\\\\\\")
        s = s.replace("'", "''")
        return "'" + s + "'"

    def _rand_literal(self):
        r = self.rng
        p = r.random()
        if p < 0.3:
            return self._rand_integer()
        elif p < 0.5:
            return self._rand_float()
        elif p < 0.7:
            return self._rand_string_literal()
        elif p < 0.8:
            return "NULL"
        elif p < 0.9:
            return r.choice(["TRUE", "FALSE"])
        else:
            return r.choice([
                "CURRENT_DATE",
                "CURRENT_TIME",
                "CURRENT_TIMESTAMP",
                "NOW()"
            ])

    def _rand_column_ref(self):
        r = self.rng
        table = r.choice(["t", "u", "tbl", "users", "orders", "a", "b"])
        col = r.choice([
            "id", "name", "value", "col1", "col2",
            "price", "qty", "created_at", "updated_at", "flag"
        ])
        if r.random() < 0.6:
            return table + "." + col
        else:
            return col

    def _rand_expr(self, depth=0):
        r = self.rng
        if depth >= self.max_expr_depth:
            if r.random() < 0.5:
                return self._rand_literal()
            else:
                return self._rand_column_ref()
        p = r.random()
        if p < 0.2:
            return self._rand_literal()
        elif p < 0.35:
            return self._rand_column_ref()
        elif p < 0.65:
            return self._rand_binary_expr(depth)
        elif p < 0.8:
            return self._rand_function_call(depth)
        elif p < 0.9:
            return self._rand_unary_expr(depth)
        elif p < 0.97:
            return self._rand_case_expr(depth)
        else:
            return self._rand_subquery_expr(depth)

    def _rand_binary_expr(self, depth):
        r = self.rng
        left = self._rand_expr(depth + 1)
        op_choice = r.random()
        if op_choice < 0.15:
            exprs = []
            n = r.randint(1, 4)
            for _ in range(n):
                exprs.append(self._rand_expr(depth + 1))
            return left + " IN (" + ", ".join(exprs) + ")"
        elif op_choice < 0.25:
            low = self._rand_expr(depth + 1)
            high = self._rand_expr(depth + 1)
            return left + " BETWEEN " + low + " AND " + high
        elif op_choice < 0.35:
            return left + " IS " + r.choice(["NULL", "NOT NULL"])
        else:
            op = r.choice(_SQL_BINARY_OPS)
            right = self._rand_expr(depth + 1)
            return "(" + left + " " + op + " " + right + ")"

    def _rand_unary_expr(self, depth):
        r = self.rng
        op = r.choice(_SQL_UNARY_OPS)
        expr = self._rand_expr(depth + 1)
        if op in ["-", "+"]:
            return op + expr
        else:
            return op + " " + expr

    def _rand_function_call(self, depth):
        r = self.rng
        if r.random() < 0.4:
            func = r.choice(_SQL_AGG_FUNCS)
        else:
            func = r.choice(_SQL_SCALAR_FUNCS)
        if func == "COUNT":
            if r.random() < 0.3:
                args = ["*"]
            else:
                args = [self._rand_expr(depth + 1)]
        elif func == "COALESCE":
            arg_count = r.randint(2, 4)
            args = [self._rand_expr(depth + 1) for _ in range(arg_count)]
        elif func == "SUBSTR":
            args = [
                self._rand_expr(depth + 1),
                self._rand_integer(),
                self._rand_integer()
            ]
        elif func == "ROUND":
            args = [self._rand_expr(depth + 1), self._rand_integer()]
        else:
            arg_count = r.randint(1, 3)
            args = [self._rand_expr(depth + 1) for _ in range(arg_count)]
        return func + "(" + ", ".join(args) + ")"

    def _rand_case_expr(self, depth):
        r = self.rng
        when_count = r.randint(1, 3)
        parts = ["CASE"]
        for _ in range(when_count):
            cond = self._rand_expr(depth + 1)
            val = self._rand_expr(depth + 1)
            parts.append("WHEN " + cond + " THEN " + val)
        if r.random() < 0.7:
            parts.append("ELSE " + self._rand_expr(depth + 1))
        parts.append("END")
        return " ".join(parts)

    def _rand_subquery_expr(self, depth):
        select = self._rand_select(depth + 1, allow_set_ops=False)
        return "(" + select + ")"

    def _rand_select_list(self, depth):
        r = self.rng
        if r.random() < 0.2:
            if r.random() < 0.5:
                return "*"
            else:
                return r.choice(["1", "2", "3"])
        count = r.randint(1, 5)
        items = []
        for _ in range(count):
            expr = self._rand_expr(depth + 1)
            if r.random() < 0.4:
                alias = self._rand_identifier("c")
                expr = expr + " AS " + alias
            items.append(expr)
        return ", ".join(items)

    def _rand_table_ref(self, depth):
        r = self.rng
        name = self._rand_identifier("t")
        parts = [name]
        if r.random() < 0.7:
            if r.random() < 0.5:
                parts.append("AS")
            alias = self._rand_identifier("a")
            parts.append(alias)
        return " ".join(parts)

    def _rand_from_clause(self, depth):
        r = self.rng
        if r.random() < 0.15:
            sub = self._rand_select(depth + 1, allow_set_ops=False)
            alias = self._rand_identifier("s")
            base = "(" + sub + ") AS " + alias
        else:
            base = self._rand_table_ref(depth)
        join_count = 0
        while join_count < 3 and r.random() < 0.5:
            join_type = r.choice([
                "JOIN",
                "LEFT JOIN",
                "RIGHT JOIN",
                "INNER JOIN",
                "FULL JOIN",
                "LEFT OUTER JOIN",
                "RIGHT OUTER JOIN"
            ])
            right = self._rand_table_ref(depth)
            cond = self._rand_expr(depth + 1)
            base = "(" + base + " " + join_type + " " + right + " ON " + cond + ")"
            join_count += 1
        return " FROM " + base

    def _rand_group_by_clause(self, depth):
        r = self.rng
        if r.random() < 0.4:
            return ""
        count = r.randint(1, 3)
        exprs = [self._rand_expr(depth + 1) for _ in range(count)]
        clause = " GROUP BY " + ", ".join(exprs)
        if r.random() < 0.5:
            clause += " HAVING " + self._rand_expr(depth + 1)
        return clause

    def _rand_order_by_clause(self, depth):
        r = self.rng
        if r.random() < 0.5:
            return ""
        count = r.randint(1, 3)
        items = []
        for _ in range(count):
            expr = self._rand_expr(depth + 1)
            direction = r.choice(["ASC", "DESC", ""])
            if direction:
                expr = expr + " " + direction
            items.append(expr)
        return " ORDER BY " + ", ".join(items)

    def _rand_limit_clause(self):
        r = self.rng
        if r.random() < 0.5:
            return ""
        limit = self._rand_integer()
        clause = " LIMIT " + limit
        if r.random() < 0.5:
            clause += " OFFSET " + self._rand_integer()
        return clause

    def _rand_select(self, depth=0, allow_set_ops=True):
        r = self.rng
        distinct = "DISTINCT " if r.random() < 0.2 else ""
        select_list = self._rand_select_list(depth)
        if r.random() < 0.75:
            from_clause = self._rand_from_clause(depth)
        else:
            from_clause = ""
        where_clause = ""
        if r.random() < 0.7:
            where_clause = " WHERE " + self._rand_expr(depth + 1)
        group_by_clause = self._rand_group_by_clause(depth)
        order_by_clause = self._rand_order_by_clause(depth)
        limit_clause = self._rand_limit_clause()
        stmt = "SELECT " + distinct + select_list + from_clause + where_clause + group_by_clause + order_by_clause + limit_clause
        if allow_set_ops and depth == 0 and r.random() < 0.25:
            op = r.choice(["UNION", "UNION ALL", "INTERSECT", "EXCEPT"])
            other = self._rand_select(depth=1, allow_set_ops=False)
            stmt = "(" + stmt + ") " + op + " (" + other + ")"
        return stmt

    def _rand_insert(self, depth=0):
        r = self.rng
        table = self._rand_identifier("t")
        col_count = r.randint(0, 5)
        if col_count == 0 or r.random() < 0.2:
            cols = ""
        else:
            cols_list = [self._rand_identifier("c") for _ in range(col_count)]
            cols = " (" + ", ".join(cols_list) + ")"
        if r.random() < 0.2:
            values_part = " DEFAULT VALUES"
        elif r.random() < 0.5:
            row_count = r.randint(1, 5)
            rows = []
            for _ in range(row_count):
                values = [self._rand_expr(depth + 1) for _ in range(max(col_count, 1))]
                rows.append("(" + ", ".join(values) + ")")
            values_part = " VALUES " + ", ".join(rows)
        else:
            sub = self._rand_select(depth + 1, allow_set_ops=False)
            values_part = " " + sub
        prefix = "INSERT"
        if r.random() < 0.2:
            prefix += " OR " + r.choice(["REPLACE", "IGNORE", "ROLLBACK", "ABORT", "FAIL"])
        stmt = prefix + " INTO " + table + cols + values_part
        return stmt

    def _rand_update(self, depth=0):
        r = self.rng
        table = self._rand_table_ref(depth)
        set_count = r.randint(1, 5)
        sets = []
        for _ in range(set_count):
            col = self._rand_identifier("c")
            expr = self._rand_expr(depth + 1)
            sets.append(col + " = " + expr)
        where = ""
        if r.random() < 0.8:
            where = " WHERE " + self._rand_expr(depth + 1)
        prefix = "UPDATE"
        if r.random() < 0.2:
            prefix += " OR " + r.choice(["REPLACE", "IGNORE", "ROLLBACK", "ABORT", "FAIL"])
        stmt = prefix + " " + table + " SET " + ", ".join(sets) + where
        return stmt

    def _rand_delete(self, depth=0):
        r = self.rng
        table = self._rand_table_ref(depth)
        where = ""
        if r.random() < 0.9:
            where = " WHERE " + self._rand_expr(depth + 1)
        stmt = "DELETE FROM " + table + where
        return stmt

    def _rand_column_def(self):
        r = self.rng
        name = self._rand_identifier("c")
        col_type = r.choice(_SQL_TYPES)
        parts = [name, col_type]
        if r.random() < 0.3:
            parts.append("NOT NULL")
        if r.random() < 0.2:
            parts.append("UNIQUE")
        if r.random() < 0.2:
            parts.append("PRIMARY KEY")
        if r.random() < 0.2:
            parts.append("DEFAULT " + self._rand_literal())
        if r.random() < 0.1:
            parts.append("CHECK (" + self._rand_expr() + ")")
        return " ".join(parts)

    def _rand_create_table(self):
        r = self.rng
        name = self._rand_identifier("t")
        col_count = r.randint(1, 6)
        cols = [self._rand_column_def() for _ in range(col_count)]
        if r.random() < 0.3:
            cols.append("PRIMARY KEY (" + ", ".join(self._rand_identifier("c") for _ in range(r.randint(1, 2))) + ")")
        stmt = "CREATE TABLE "
        if r.random() < 0.2:
            stmt += "IF NOT EXISTS "
        stmt += name + " (" + ", ".join(cols) + ")"
        return stmt

    def _rand_alter_table(self):
        r = self.rng
        name = self._rand_identifier("t")
        action_type = r.random()
        if action_type < 0.33:
            col_def = self._rand_column_def()
            return "ALTER TABLE " + name + " ADD COLUMN " + col_def
        elif action_type < 0.66:
            old = self._rand_identifier("c")
            new = self._rand_identifier("c")
            if r.random() < 0.5:
                return "ALTER TABLE " + name + " RENAME COLUMN " + old + " TO " + new
            else:
                return "ALTER TABLE " + name + " RENAME TO " + new
        else:
            col = self._rand_identifier("c")
            return "ALTER TABLE " + name + " DROP COLUMN " + col

    def _rand_create_index(self):
        r = self.rng
        idx = self._rand_identifier("idx")
        table = self._rand_identifier("t")
        col_count = r.randint(1, 3)
        cols = [self._rand_identifier("c") for _ in range(col_count)]
        stmt = "CREATE "
        if r.random() < 0.3:
            stmt += "UNIQUE "
        stmt += "INDEX "
        if r.random() < 0.2:
            stmt += "IF NOT EXISTS "
        stmt += idx + " ON " + table + " (" + ", ".join(cols)
        if r.random() < 0.5:
            orders = []
            for c in cols:
                orders.append(c + " " + r.choice(["ASC", "DESC"]))
            stmt = "CREATE "
            if r.random() < 0.3:
                stmt += "UNIQUE "
            stmt += "INDEX "
            if r.random() < 0.2:
                stmt += "IF NOT EXISTS "
            stmt += idx + " ON " + table + " (" + ", ".join(orders) + ")"
        else:
            stmt += ")"
        return stmt

    def _rand_drop(self):
        r = self.rng
        obj_type = r.choice(["TABLE", "INDEX", "VIEW"])
        name = self._rand_identifier("t" if obj_type == "TABLE" else "idx")
        stmt = "DROP " + obj_type + " "
        if r.random() < 0.4:
            stmt += "IF EXISTS "
        stmt += name
        return stmt

    def _rand_create_view(self, depth=0):
        r = self.rng
        name = self._rand_identifier("v")
        select = self._rand_select(depth + 1, allow_set_ops=False)
        stmt = "CREATE VIEW "
        if r.random() < 0.2:
            stmt += "IF NOT EXISTS "
        stmt += name + " AS " + select
        return stmt

    def _rand_transaction(self):
        r = self.rng
        return r.choice([
            "BEGIN",
            "BEGIN TRANSACTION",
            "COMMIT",
            "ROLLBACK",
            "SAVEPOINT " + self._rand_identifier("sp"),
            "RELEASE " + self._rand_identifier("sp"),
            "ROLLBACK TO " + self._rand_identifier("sp")
        ])

    def _rand_random_garbage(self):
        r = self.rng
        length = r.randint(5, 60)
        chars = []
        alphabet = string.ascii_letters + string.digits + " \\t\\n.,;:+-*/%&|^!?$#@()[]{}<>=_'\\\"`"
        for _ in range(length):
            chars.append(r.choice(alphabet))
        return "".join(chars)

    def _rand_mutation_snippet(self):
        r = self.rng
        choice = r.random()
        if choice < 0.3:
            return " " + r.choice(_SQL_KEYWORDS) + " "
        elif choice < 0.6:
            return " " + self._rand_identifier("m") + " "
        elif choice < 0.8:
            return " " + self._rand_literal() + " "
        else:
            return " " + r.choice(["+", "-", "*", "/", "=", "<", ">", "<=", ">=", "<>", "!=", "AND", "OR"]) + " "

    def _mutate_statement(self, s):
        r = self.rng
        if not s or len(s) > self.max_stmt_len * 2:
            return self._rand_statement()
        s_list = list(s)
        max_len = self.max_stmt_len
        operations = r.randint(1, 3)
        for _ in range(operations):
            op = r.random()
            if op < 0.25:
                pos = r.randint(0, len(s_list))
                snippet = self._rand_mutation_snippet()
                for ch in snippet:
                    if len(s_list) >= max_len:
                        break
                    s_list.insert(pos, ch)
                    pos += 1
            elif op < 0.5 and s_list:
                if len(s_list) <= 1:
                    continue
                start = r.randint(0, len(s_list) - 1)
                end = min(len(s_list), start + r.randint(1, min(20, len(s_list) - start)))
                del s_list[start:end]
            elif op < 0.75 and s_list:
                idx = r.randint(0, len(s_list) - 1)
                ch = s_list[idx]
                if ch.islower():
                    s_list[idx] = ch.upper()
                elif ch.isupper():
                    s_list[idx] = ch.lower()
            else:
                if not s_list:
                    continue
                start = r.randint(0, len(s_list) - 1)
                end = min(len(s_list), start + r.randint(1, 10))
                snippet = list(self._rand_mutation_snippet())
                slice_len = max(1, min(len(snippet), max_len - len(s_list)))
                s_list[start:end] = snippet[:slice_len]
        mutated = "".join(s_list)
        if len(mutated) > max_len:
            mutated = mutated[:max_len]
        return mutated

    def _rand_statement(self):
        r = self.rng
        p = r.random()
        if p < 0.45:
            stmt = self._rand_select()
        elif p < 0.6:
            stmt = self._rand_insert()
        elif p < 0.72:
            stmt = self._rand_update()
        elif p < 0.82:
            stmt = self._rand_delete()
        elif p < 0.9:
            choice = r.random()
            if choice < 0.5:
                stmt = self._rand_create_table()
            elif choice < 0.7:
                stmt = self._rand_create_index()
            elif choice < 0.85:
                stmt = self._rand_alter_table()
            else:
                stmt = self._rand_create_view()
        elif p < 0.96:
            stmt = self._rand_drop()
        else:
            stmt = self._rand_transaction()
        if r.random() < 0.05:
            stmt = self._rand_random_garbage()
        return stmt

    def generate_batch(self):
        r = self.rng
        batch = []
        for _ in range(self.batch_size):
            if self.corpus and r.random() < 0.5:
                base = r.choice(self.corpus)
                stmt = self._mutate_statement(base)
            else:
                stmt = self._rand_statement()
                if len(self.corpus) < self.max_corpus and r.random() < 0.5:
                    self.corpus.append(stmt)
            if len(stmt) > self.max_stmt_len:
                stmt = stmt[: self.max_stmt_len]
            batch.append(stmt)
        self.total_statements += len(batch)
        return batch

    def adjust_batch_size(self, duration):
        if duration <= 0:
            return
        alpha = self.smoothing
        if self.avg_call_duration <= 0:
            self.avg_call_duration = duration
        else:
            self.avg_call_duration = alpha * duration + (1.0 - alpha) * self.avg_call_duration
        target = 0.15
        r = self.rng
        if self.avg_call_duration < target * 0.5 and self.batch_size < self.max_batch_size:
            scale = 1.0 + r.random() * 0.3
            new_size = int(self.batch_size * scale) + 1
            self.batch_size = min(self.max_batch_size, new_size)
        elif self.avg_call_duration > target * 2.5 and self.batch_size > self.min_batch_size:
            scale = 0.5 + r.random() * 0.3
            new_size = int(self.batch_size * scale)
            if new_size < self.min_batch_size:
                new_size = self.min_batch_size
            self.batch_size = new_size


_GLOBAL_STATE = None


def fuzz(parse_sql):
    global _GLOBAL_STATE
    if _GLOBAL_STATE is None:
        _GLOBAL_STATE = _FuzzerState()
    state = _GLOBAL_STATE
    state.call_count += 1
    if not state.initialized:
        state.initialized = True
        state.start_time = time.time()
    t0 = time.time()
    stmts = state.generate_batch()
    try:
        parse_sql(stmts)
    except Exception:
        pass
    t1 = time.time()
    state.adjust_batch_size(t1 - t0)
    return True
'''
        return {"code": code}