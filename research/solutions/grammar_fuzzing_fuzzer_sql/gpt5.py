import os
import random
import string
import time


# -----------------------------
# Internal fuzzer implementation
# -----------------------------

class _FastRandom:
    __slots__ = ("_r", "randint", "random", "choice", "choices", "shuffle", "uniform")

    def __init__(self, seed=None):
        r = random.Random(seed)
        self._r = r
        self.randint = r.randint
        self.random = r.random
        self.choice = r.choice
        self.choices = r.choices if hasattr(r, "choices") else None
        self.shuffle = r.shuffle
        self.uniform = r.uniform


class _FuzzerState:
    def __init__(self):
        seed = (int(time.time() * 1000) ^ os.getpid()) & 0xFFFFFFFF
        self.r = _FastRandom(seed)
        self.call_idx = 0
        self.corpus = []
        self.seen = set()
        self.max_depth_expr = 3
        self.max_depth_select = 2
        self._init_symbols()

    def _init_symbols(self):
        # Keywords and names to exercise tokenizer and parser code paths
        self.keywords = [
            "SELECT", "INSERT", "UPDATE", "DELETE", "FROM", "WHERE", "GROUP", "BY", "HAVING",
            "ORDER", "LIMIT", "OFFSET", "DISTINCT", "ALL", "AS", "AND", "OR", "NOT", "IN",
            "IS", "NULL", "LIKE", "ESCAPE", "BETWEEN", "EXISTS", "CASE", "WHEN", "THEN",
            "ELSE", "END", "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "OUTER", "CROSS",
            "ON", "USING", "WITH", "RECURSIVE", "UNION", "INTERSECT", "EXCEPT",
            "CREATE", "TABLE", "VIEW", "INDEX", "UNIQUE", "PRIMARY", "KEY", "FOREIGN",
            "REFERENCES", "CHECK", "DEFAULT", "CONSTRAINT", "IF", "EXISTS", "NOT", "DROP",
            "ALTER", "ADD", "COLUMN", "RENAME", "TO", "SET", "VALUES", "INTO", "RETURNING",
        ]

        # Common table and column names
        base_tables = [
            "users", "orders", "products", "inventory", "logs", "sessions", "events",
            "metrics", "visits", "accounts", "profiles", "messages", "comments", "likes",
            "follows", "payments", "transactions", "items", "categories", "suppliers",
            "customers", "employees", "departments", "regions", "countries", "states",
            "cities", "addresses", "roles", "permissions", "tokens", "reports", "tags",
            "articles", "posts", "files", "images", "videos", "audits", "backups",
            "archives", "refs", "map", "graph", "edges", "nodes", "feeds", "alerts", "jobs",
        ]
        # Add some quoted and schema-qualified names to stress tokenizer
        self.tables = []
        for t in base_tables:
            self.tables.append(t)
            self.tables.append(f"t_{t}")
            self.tables.append(f'"{t}"')
            self.tables.append(f"main.{t}")
            self.tables.append(f'temp."{t}"')

        base_cols = [
            "id", "user_id", "order_id", "product_id", "session_id", "event_id",
            "created_at", "updated_at", "deleted_at", "name", "title", "description",
            "content", "email", "phone", "status", "amount", "price", "quantity",
            "score", "rating", "count", "total", "flag", "active", "enabled", "role",
            "type", "category", "tag", "lat", "lng", "address", "city", "state",
            "zip", "country", "age", "gender", "notes", "data", "payload", "meta",
        ]
        self.columns = []
        for c in base_cols:
            self.columns.append(c)
            self.columns.append(f"c_{c}")
            self.columns.append(f'"{c}"')
            self.columns.append(f"`{c}`")

        # SQL types
        self.types = [
            "INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT",
            "NUMERIC", "DECIMAL", "DECIMAL(10,2)", "REAL", "FLOAT", "DOUBLE",
            "TEXT", "VARCHAR(32)", "CHAR(16)", "BOOLEAN",
            "DATE", "TIME", "TIMESTAMP"
        ]

        # Functions
        self.scalar_funcs = [
            "ABS", "LOWER", "UPPER", "LENGTH", "ROUND", "FLOOR", "CEIL", "SUBSTR",
            "TRIM", "LTRIM", "RTRIM", "REPLACE", "COALESCE", "NULLIF"
        ]
        self.agg_funcs = [
            "COUNT", "SUM", "AVG", "MIN", "MAX"
        ]
        self.other_funcs = [
            "RANDOM", "NOW", "CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP"
        ]
        self.ops_arith = ["+", "-", "*", "/", "%"]
        self.ops_cmp = ["=", "!=", "<>", "<", "<=", ">", ">="]
        self.ops_logic = ["AND", "OR"]
        self.like_escapes = ["\\", "!", "#", "~"]
        self.param_markers = ["?", ":x", ":name", "@p1", "$1"]

        # For mutation
        self.op_replacements = {
            "=": ["!=", "<>", ">=", "<="],
            "!=": ["=", "<>"],
            "<>": ["=", "!="],
            ">": [">=", "<", "<="],
            "<": ["<=", ">", ">="],
            "AND": ["OR"],
            "OR": ["AND"],
        }

    # ---------------------------------
    # Random helpers
    # ---------------------------------

    def maybe(self, p=0.5):
        return self.r.random() < p

    def one_of(self, seq):
        return self.r.choice(seq)

    def rand_ident(self):
        # Mix of base names and random
        if self.maybe(0.6):
            return self.one_of(self.columns + self.tables)
        # random identifier
        letters = string.ascii_letters + "_"
        first = self.one_of(letters)
        length = self.r.randint(1, 10)
        rest_letters = letters + string.digits + "_$"
        rest = "".join(self.one_of(rest_letters) for _ in range(length))
        return first + rest

    def rand_table(self):
        return self.one_of(self.tables)

    def rand_column(self):
        col = self.one_of(self.columns)
        if self.maybe(0.2):
            # qualify with table
            return f"{self.rand_table().split('.')[-1].strip('\"')}.{col}"
        return col

    def rand_string_literal(self):
        # Generate SQL single-quoted string with doubled quotes for escapes
        modes = self.r.randint(0, 8)
        if modes == 0:
            s = ""
        elif modes == 1:
            s = "O'Brien"
        elif modes == 2:
            s = "line1\nline2"
        elif modes == 3:
            s = "🧪 unicode"
        elif modes == 4:
            s = " " * self.r.randint(1, 5)
        elif modes == 5:
            s = "".join(self.one_of(string.printable) for _ in range(self.r.randint(1, 12)))
        elif modes == 6:
            # long string
            s = "".join(self.one_of("abcXYZ0123 '") for _ in range(self.r.randint(20, 60)))
        else:
            s = "path\\to\\file"
        s = s.replace("'", "''")
        return f"'{s}'"

    def rand_blob_literal(self):
        # x'AB12'
        length = self.r.randint(0, 6) * 2 + 2
        hexchars = "0123456789ABCDEF"
        content = "".join(self.one_of(hexchars) for _ in range(length))
        return f"x'{content}'"

    def rand_number_literal(self):
        choices = [
            "0", "1", "-1", "2", "3", "10", "100", "255", "256",
            "32767", "-32768", "2147483647", "-2147483648",
            "9223372036854775807", "-9223372036854775808",
            "0.0", "1.5", "-3.14", ".5", "1.", "1e10", "-2E-3"
        ]
        if self.maybe(0.7):
            return self.one_of(choices)
        # random float/int
        if self.maybe(0.5):
            return str(self.r.randint(-100000, 100000))
        else:
            v = self.r.uniform(-100000, 100000)
            return f"{v:.6f}"

    def rand_bool_null_literal(self):
        return self.one_of(["TRUE", "FALSE", "NULL"])

    def rand_param_marker(self):
        return self.one_of(self.param_markers)

    def rand_type(self):
        t = self.one_of(self.types)
        # occasionally with extras
        if "VARCHAR" not in t and self.maybe(0.2):
            t = f"{t}({self.r.randint(1,64)})"
        return t

    # ---------------------------------
    # Expression generation
    # ---------------------------------

    def gen_expr(self, depth=0):
        if depth > self.max_depth_expr:
            return self.gen_expr_terminal()

        # Weighted selection among expression kinds
        roll = self.r.random()
        if roll < 0.25:
            return self.gen_expr_binary(depth)
        elif roll < 0.40:
            return self.gen_expr_unary(depth)
        elif roll < 0.55:
            return self.gen_func_call(depth)
        elif roll < 0.67:
            return self.gen_case_expr(depth)
        elif roll < 0.75:
            return self.gen_in_expr(depth)
        elif roll < 0.82:
            return self.gen_between_expr(depth)
        elif roll < 0.90 and depth < self.max_depth_expr - 1:
            return f"({self.gen_expr(depth+1)})"
        elif roll < 0.95 and depth < self.max_depth_expr - 1:
            return self.gen_exists_subquery(depth+1)
        else:
            return self.gen_expr_terminal()

    def gen_expr_terminal(self):
        choice = self.r.randint(0, 9)
        if choice == 0:
            return self.rand_number_literal()
        elif choice == 1:
            return self.rand_string_literal()
        elif choice == 2:
            return self.rand_bool_null_literal()
        elif choice == 3:
            return self.rand_blob_literal()
        elif choice == 4:
            # parameter marker
            return self.rand_param_marker()
        elif choice == 5:
            return self.rand_column()
        elif choice == 6:
            return f"CAST({self.gen_expr(1)} AS {self.rand_type()})"
        elif choice == 7:
            return f"({self.rand_number_literal()} {self.one_of(self.ops_arith)} {self.rand_number_literal()})"
        elif choice == 8:
            return self.gen_func_call(1)
        else:
            return "NULL"

    def gen_expr_unary(self, depth=0):
        if self.maybe(0.5):
            return f"NOT {self.gen_expr(depth+1)}"
        else:
            return f"-{self.gen_expr(depth+1)}"

    def gen_expr_binary(self, depth=0):
        # Choose binary operation: arithmetic or comparison or logic
        kind = self.r.random()
        if kind < 0.4:
            op = self.one_of(self.ops_arith)
        elif kind < 0.8:
            op = self.one_of(self.ops_cmp)
        else:
            op = self.one_of(self.ops_logic)
        a = self.gen_expr(depth+1)
        b = self.gen_expr(depth+1)
        return f"{a} {op} {b}"

    def gen_func_call(self, depth=0):
        if self.maybe(0.4):
            fn = self.one_of(self.scalar_funcs)
        elif self.maybe(0.6):
            fn = self.one_of(self.agg_funcs)
        else:
            fn = self.one_of(self.other_funcs)
        # COUNT(*) special case
        if fn == "COUNT" and self.maybe(0.5):
            return "COUNT(*)"
        # args
        nargs = self.r.randint(0, 3)
        if fn in {"NOW", "CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP"}:
            nargs = 0
        args = []
        for _ in range(nargs):
            args.append(self.gen_expr(depth+1))
        return f"{fn}({', '.join(args)})"

    def gen_case_expr(self, depth=0):
        when_count = self.r.randint(1, 2)
        parts = ["CASE"]
        for _ in range(when_count):
            cond = self.gen_expr(depth+1)
            val = self.gen_expr(depth+1)
            parts.append(f"WHEN {cond} THEN {val}")
        if self.maybe(0.7):
            parts.append(f"ELSE {self.gen_expr(depth+1)}")
        parts.append("END")
        return " ".join(parts)

    def gen_in_expr(self, depth=0):
        lhs = self.gen_expr(depth+1)
        if self.maybe(0.6) and depth < self.max_depth_expr - 1:
            sub = self.gen_select(depth+1, minimal=True)
            return f"{lhs} IN ({sub})"
        else:
            n = self.r.randint(1, 4)
            items = [self.gen_expr(depth+1) for _ in range(n)]
            return f"{lhs} IN ({', '.join(items)})"

    def gen_between_expr(self, depth=0):
        a = self.gen_expr(depth+1)
        b = self.gen_expr(depth+1)
        c = self.gen_expr(depth+1)
        if self.maybe(0.5):
            return f"{a} BETWEEN {b} AND {c}"
        else:
            return f"{a} NOT BETWEEN {b} AND {c}"

    def gen_like_expr(self, depth=0):
        a = self.gen_expr(depth+1)
        b = self.rand_string_literal()
        s = f"{a} LIKE {b}"
        if self.maybe(0.3):
            s += f" ESCAPE '{self.one_of(self.like_escapes)}'"
        return s

    def gen_exists_subquery(self, depth=0):
        sub = self.gen_select(depth+1, minimal=True)
        return f"EXISTS ({sub})"

    # ---------------------------------
    # Statement generation
    # ---------------------------------

    def gen_select(self, depth=0, minimal=False):
        # WITH clause
        parts = []
        if self.maybe(0.12) and depth == 0:
            parts.append(self.gen_with_clause(depth))

        parts.append("SELECT")

        if self.maybe(0.2):
            parts.append(self.one_of(["DISTINCT", "ALL"]))

        # select list
        if minimal and self.maybe(0.3):
            select_list = ["*"]
        else:
            ncols = self.r.randint(1, 4)
            select_cols = []
            for _ in range(ncols):
                if self.maybe(0.15):
                    select_cols.append("*")
                else:
                    e = self.gen_expr(depth+1)
                    if self.maybe(0.3):
                        alias = self.rand_ident()
                        if self.maybe(0.5):
                            select_cols.append(f"{e} AS {alias}")
                        else:
                            select_cols.append(f"{e} {alias}")
                    else:
                        select_cols.append(e)
            select_list = select_cols
        parts.append(", ".join(select_list))

        # FROM clause
        if self.maybe(0.85) or minimal is False:
            parts.append("FROM")
            parts.append(self.gen_from_clause(depth))

        # WHERE
        if self.maybe(0.6):
            cond = self.gen_expr(depth+1)
            # Occasionally produce LIKE specifically
            if self.maybe(0.2):
                cond = self.gen_like_expr(depth+1)
            parts.append("WHERE")
            parts.append(cond)

        # GROUP BY
        if self.maybe(0.35):
            n = self.r.randint(1, 3)
            cols = [self.rand_column() for _ in range(n)]
            parts.append("GROUP BY")
            parts.append(", ".join(cols))
            if self.maybe(0.4):
                parts.append("HAVING")
                parts.append(self.gen_expr(depth+1))

        # ORDER BY
        if self.maybe(0.55):
            n = self.r.randint(1, 3)
            items = []
            for _ in range(n):
                ex = self.gen_expr(depth+1)
                direction = self.one_of(["ASC", "DESC"]) if self.maybe(0.6) else ""
                nulls = ""
                if self.maybe(0.2):
                    nulls = self.one_of(["NULLS FIRST", "NULLS LAST"])
                items.append(" ".join(x for x in [ex, direction, nulls] if x))
            parts.append("ORDER BY")
            parts.append(", ".join(items))

        # LIMIT/OFFSET
        if self.maybe(0.5):
            parts.append("LIMIT")
            parts.append(self.rand_number_literal())
            if self.maybe(0.5):
                parts.append("OFFSET")
                parts.append(self.rand_number_literal())

        stmt = " ".join(parts)

        # Set operations
        if depth < self.max_depth_select and self.maybe(0.25):
            op = self.one_of(["UNION", "UNION ALL", "INTERSECT", "EXCEPT"])
            right = self.gen_select(depth+1, minimal=True)
            stmt = f"{stmt} {op} {right}"

        # Wrap as subquery rarely
        if self.maybe(0.05):
            stmt = f"({stmt})"

        # Add semicolon sometimes
        if self.maybe(0.5):
            stmt += ";"
        return stmt

    def gen_with_clause(self, depth=0):
        s = "WITH"
        if self.maybe(0.2):
            s += " RECURSIVE"
        n = self.r.randint(1, 2)
        ctes = []
        for _ in range(n):
            name = self.rand_ident()
            cols_decl = ""
            if self.maybe(0.4):
                cols = [self.rand_ident() for _ in range(self.r.randint(1, 3))]
                cols_decl = f"({', '.join(cols)})"
            sub = self.gen_select(depth+1, minimal=True)
            ctes.append(f"{name}{cols_decl} AS ({sub})")
        return f"{s} " + ", ".join(ctes)

    def gen_table_factor(self, depth=0):
        if self.maybe(0.7):
            name = self.rand_table()
            alias = ""
            if self.maybe(0.4):
                if self.maybe(0.5):
                    alias = f" AS {self.rand_ident()}"
                else:
                    alias = f" {self.rand_ident()}"
            return f"{name}{alias}"
        else:
            # subquery
            sub = self.gen_select(depth+1, minimal=True)
            alias = self.rand_ident()
            as_kw = " AS " if self.maybe(0.5) else " "
            return f"({sub}){as_kw}{alias}"

    def gen_join(self, depth=0):
        left = self.gen_table_factor(depth)
        join_count = self.r.randint(0, 2)
        for _ in range(join_count):
            jt = self.one_of(
                ["JOIN", "INNER JOIN", "LEFT JOIN", "LEFT OUTER JOIN", "RIGHT JOIN",
                 "FULL JOIN", "FULL OUTER JOIN", "CROSS JOIN"]
            )
            right = self.gen_table_factor(depth)
            if "CROSS" in jt and self.maybe(0.7):
                left = f"{left} {jt} {right}"
            else:
                cond = ""
                if self.maybe(0.7):
                    cond = f" ON {self.gen_expr(depth+1)}"
                else:
                    cols = [self.rand_ident() for _ in range(self.r.randint(1, 2))]
                    cond = f" USING ({', '.join(cols)})"
                left = f"{left} {jt} {right}{cond}"
        return left

    def gen_from_clause(self, depth=0):
        items = [self.gen_join(depth)]
        if self.maybe(0.25):
            # multiple table refs separated by comma
            n = self.r.randint(1, 2)
            for _ in range(n):
                items.append(self.gen_table_factor(depth))
        return ", ".join(items)

    def gen_insert(self):
        table = self.rand_table()
        parts = ["INSERT"]
        if self.maybe(0.2):
            parts.append("OR")
            parts.append(self.one_of(["ABORT", "FAIL", "IGNORE", "REPLACE", "ROLLBACK"]))
        parts.append("INTO")
        parts.append(table)
        if self.maybe(0.6):
            # columns list
            n = self.r.randint(1, 4)
            cols = [self.rand_ident() for _ in range(n)]
            parts.append(f"({', '.join(cols)})")
        if self.maybe(0.5):
            # VALUES with multi-rows
            parts.append("VALUES")
            row_count = self.r.randint(1, 3)
            rows = []
            for _ in range(row_count):
                n = self.r.randint(1, 4)
                vals = [self.gen_expr(1) for _ in range(n)]
                rows.append(f"({', '.join(vals)})")
            parts.append(", ".join(rows))
        else:
            # INSERT INTO ... SELECT
            parts.append(self.gen_select(1, minimal=True))
        if self.maybe(0.5):
            parts.append(";")
        return " ".join(parts)

    def gen_update(self):
        table = self.rand_table()
        parts = ["UPDATE", table, "SET"]
        n = self.r.randint(1, 4)
        assigns = []
        for _ in range(n):
            col = self.rand_ident()
            val = self.gen_expr(1)
            assigns.append(f"{col} = {val}")
        parts.append(", ".join(assigns))
        if self.maybe(0.7):
            parts.append("WHERE")
            parts.append(self.gen_expr(1))
        if self.maybe(0.3):
            parts.append("RETURNING")
            cols = [self.rand_ident() for _ in range(self.r.randint(1, 3))]
            parts.append(", ".join(cols))
        if self.maybe(0.5):
            parts.append(";")
        return " ".join(parts)

    def gen_delete(self):
        table = self.rand_table()
        parts = ["DELETE FROM", table]
        if self.maybe(0.7):
            parts.append("WHERE")
            parts.append(self.gen_expr(1))
        if self.maybe(0.3):
            parts.append("RETURNING *")
        if self.maybe(0.5):
            parts.append(";")
        return " ".join(parts)

    def gen_create_table(self):
        name = self.rand_table()
        parts = ["CREATE"]
        if self.maybe(0.25):
            parts.append("TEMP")
        parts.append("TABLE")
        if self.maybe(0.3):
            parts.append("IF NOT EXISTS")
        parts.append(name)
        ncols = self.r.randint(1, 6)
        cols = []
        for _ in range(ncols):
            colname = self.rand_ident()
            typ = self.rand_type()
            constraints = []
            if self.maybe(0.4):
                constraints.append("NOT NULL")
            if self.maybe(0.2):
                constraints.append("UNIQUE")
            if self.maybe(0.2):
                constraints.append("PRIMARY KEY")
            if self.maybe(0.2):
                constraints.append(f"DEFAULT {self.gen_expr(1)}")
            if self.maybe(0.15):
                constraints.append(f"CHECK ({self.gen_expr(1)})")
            cols.append(" ".join([colname, typ] + constraints))
        tbl_constraints = []
        if self.maybe(0.3):
            pk_cols = [self.rand_ident() for _ in range(self.r.randint(1, 3))]
            tbl_constraints.append(f"PRIMARY KEY ({', '.join(pk_cols)})")
        if self.maybe(0.25):
            uq_cols = [self.rand_ident() for _ in range(self.r.randint(1, 3))]
            tbl_constraints.append(f"UNIQUE ({', '.join(uq_cols)})")
        if self.maybe(0.20):
            fk_cols = [self.rand_ident() for _ in range(self.r.randint(1, 2))]
            ref_tbl = self.rand_table()
            ref_cols = [self.rand_ident() for _ in range(self.r.randint(1, 2))]
            tbl_constraints.append(f"FOREIGN KEY ({', '.join(fk_cols)}) REFERENCES {ref_tbl} ({', '.join(ref_cols)})")
        body = ", ".join(cols + tbl_constraints)
        stmt = f"CREATE TABLE {('IF NOT EXISTS ' if 'IF NOT EXISTS' in parts else '')}{name} ({body})"
        # The above line re-adds; refine:
        stmt = " ".join(parts[:-1]) + f" {name} ({body})"
        if self.maybe(0.5):
            stmt += ";"
        return stmt

    def gen_create_index(self):
        name = self.rand_ident()
        table = self.rand_table()
        parts = ["CREATE"]
        if self.maybe(0.3):
            parts.append("UNIQUE")
        parts.append("INDEX")
        if self.maybe(0.3):
            parts.append("IF NOT EXISTS")
        parts.append(name)
        parts.append("ON")
        parts.append(table)
        # columns
        n = self.r.randint(1, 3)
        cols = []
        for _ in range(n):
            c = self.rand_ident()
            direction = self.one_of(["ASC", "DESC"]) if self.maybe(0.5) else ""
            cols.append(" ".join(x for x in [c, direction] if x))
        parts.append(f"({', '.join(cols)})")
        if self.maybe(0.2):
            parts.append("WHERE")
            parts.append(self.gen_expr(1))
        if self.maybe(0.5):
            parts.append(";")
        return " ".join(parts)

    def gen_drop(self):
        if self.maybe(0.5):
            kind = self.one_of(["TABLE", "INDEX", "VIEW"])
        else:
            kind = "TABLE"
        parts = ["DROP", kind]
        if self.maybe(0.3):
            parts.append("IF EXISTS")
        name = self.rand_table() if kind in {"TABLE", "VIEW"} else self.rand_ident()
        parts.append(name)
        if self.maybe(0.5):
            parts.append(";")
        return " ".join(parts)

    def gen_alter(self):
        table = self.rand_table()
        parts = ["ALTER TABLE", table]
        action = self.r.randint(0, 3)
        if action == 0:
            parts.append("ADD COLUMN")
            parts.append(self.rand_ident())
            parts.append(self.rand_type())
        elif action == 1:
            parts.append("DROP COLUMN")
            parts.append(self.rand_ident())
        elif action == 2:
            parts.append("RENAME COLUMN")
            parts.append(self.rand_ident())
            parts.append("TO")
            parts.append(self.rand_ident())
        else:
            parts.append("RENAME TO")
            parts.append(self.rand_ident())
        if self.maybe(0.5):
            parts.append(";")
        return " ".join(parts)

    def gen_create_view(self):
        name = self.rand_ident()
        parts = ["CREATE"]
        if self.maybe(0.25):
            parts.append("TEMP")
        parts.append("VIEW")
        if self.maybe(0.3):
            parts.append("IF NOT EXISTS")
        parts.append(name)
        parts.append("AS")
        parts.append(self.gen_select(1, minimal=False))
        if self.maybe(0.5):
            parts.append(";")
        return " ".join(parts)

    def gen_misc_token_soup(self):
        # Generate intentionally invalid/edge-case sequences to hit tokenizer and error paths
        pieces = []
        # random keywords, identifiers, operators, comments, weird tokens
        token_pool = (
            self.keywords
            + [self.rand_ident() for _ in range(10)]
            + ["@", "$$", "$1", ":param", "--", "/*", "*/", "(", ")", ",", ";", ".", "*", "+", "-", "||", "|", "&", "^", "~", "!!"]
        )
        length = self.r.randint(3, 20)
        for _ in range(length):
            t = self.one_of(token_pool)
            if t == "--":
                # line comment
                pieces.append("--" + "comment " * self.r.randint(0, 2))
                pieces.append("\n")
            elif t == "/*":
                pieces.append("/*")
                pieces.append("cmt " * self.r.randint(0, 2))
                pieces.append("*/")
            elif t in {"(", ")", ",", ";", ".", "*", "+", "-", "||", "|", "&", "^", "~"}:
                pieces.append(t)
            else:
                # sometimes a literal
                if self.maybe(0.1):
                    pieces.append(self.rand_string_literal())
                elif self.maybe(0.1):
                    pieces.append(self.rand_number_literal())
                else:
                    pieces.append(t)
            # Add spaces randomly
            if self.maybe(0.7):
                pieces.append(" ")
        s = "".join(pieces).strip()
        if not s:
            s = "/* empty */"
        return s

    def gen_invalid_stmt(self):
        templates = [
            "SELECT FROM {table}",
            "SELECT {col} {op} FROM",
            "INSERT INTO {table} ({col}, {col2})",
            "UPDATE SET {col} = {val}",
            "DELETE {table} WHERE {expr}",
            "CREATE TABLE {table} ( {col} {type}, {col2} )",  # missing type for col2
            "ALTER TABLE {table} RENAME",
            "DROP {kw} IF {kw2} {name}",
            "SELECT ( {expr} + {expr} ",
            "SELECT 'unterminated FROM {table}",
            "SELECT * FROM {table} WHERE {col} LIKE '{wild}",
            "WITH c AS ({sel}) SELECT * FRM {table}",
        ]
        t = self.one_of(templates)
        s = t.format(
            table=self.rand_table(),
            col=self.rand_ident(),
            col2=self.rand_ident(),
            type=self.rand_type(),
            val=self.gen_expr(1),
            expr=self.gen_expr(1),
            op=self.one_of(self.ops_cmp),
            kw=self.one_of(self.keywords),
            kw2=self.one_of(self.keywords),
            name=self.rand_ident(),
            sel=self.gen_select(1, minimal=True),
            wild="%" * self.r.randint(1, 3),
        )
        # Randomly append garbage
        if self.maybe(0.3):
            s += " " + self.gen_misc_token_soup()
        return s

    # ---------------------------------
    # Mutation
    # ---------------------------------

    def mutate_stmt(self, s):
        # Apply one random mutation
        choice = self.r.randint(0, 7)
        if choice == 0:
            # randomly change case of keywords
            def flip(word):
                if self.maybe(0.5):
                    return word.lower()
                return word.upper()
            tokens = s.split()
            tokens = [flip(t) if t.upper() in self.keywords else t for t in tokens]
            return " ".join(tokens)
        elif choice == 1:
            # Insert random comments
            insertion = self.one_of([
                "/*c*/", "/* comment */", "--x\n", "-- long comment here\n"
            ])
            pos = self.r.randint(0, len(s))
            return s[:pos] + " " + insertion + " " + s[pos:]
        elif choice == 2:
            # Replace operators
            for k, repls in self.op_replacements.items():
                if k in s and self.maybe(0.5):
                    s = s.replace(k, self.one_of(repls), 1)
                    break
            return s
        elif choice == 3:
            # Add parentheses
            return f"({s})"
        elif choice == 4:
            # Append semicolon or remove
            if s.endswith(";"):
                return s[:-1]
            else:
                return s + ";"
        elif choice == 5:
            # Replace a number with extreme
            return s.replace(" 1 ", " 2147483647 ", 1)
        elif choice == 6:
            # Random whitespace normalization
            return " ".join(s.split())
        else:
            # Append token soup
            return s + " " + self.gen_misc_token_soup()

    # ---------------------------------
    # Batch generation per wave
    # ---------------------------------

    def gen_wave_statements(self, wave):
        # Return list of statements for the given wave
        stmts = []
        emit = lambda x: self._emit_unique(stmts, x)
        r = self.r

        def bulk_generate(gen_fn, count):
            for _ in range(count):
                try:
                    s = gen_fn()
                except Exception:
                    # Fail-safe: token soup
                    s = self.gen_misc_token_soup()
                emit(s)

        if wave == 0:
            # Wave 1: tokenizer stress + simple selects
            count_invalid = 700
            count_simple_select = 2300
            count_mut = 0

            bulk_generate(self.gen_invalid_stmt, count_invalid)
            bulk_generate(lambda: self.gen_select(0, minimal=True), count_simple_select)

            # Some token soups
            for _ in range(200):
                emit(self.gen_misc_token_soup())

            # Create a small seed corpus for later mutation
            seeds = []
            for _ in range(100):
                seeds.append(self.gen_select(0, minimal=False))
            self.corpus.extend(seeds)
            for s in seeds:
                emit(s)

        elif wave == 1:
            # Wave 2: richer SELECTs with joins, subqueries, set ops
            count_select = 2500
            bulk_generate(lambda: self.gen_select(0, minimal=False), count_select)

            # Some explicit LIKE/IN conditions
            for _ in range(300):
                s = f"SELECT {self.gen_expr(1)} FROM {self.gen_from_clause(0)} WHERE {self.gen_like_expr(1)}"
                if self.maybe(0.5):
                    s += ";"
                emit(s)

            # Mutate some from corpus
            for _ in range(500):
                if not self.corpus:
                    break
                base = self.one_of(self.corpus)
                emit(self.mutate_stmt(base))

        elif wave == 2:
            # Wave 3: DML focus
            bulk_generate(self.gen_insert, 1000)
            bulk_generate(self.gen_update, 700)
            bulk_generate(self.gen_delete, 600)

            # Combine with SELECT subqueries in DML
            for _ in range(400):
                table = self.rand_table()
                s = f"INSERT INTO {table} SELECT * FROM ({self.gen_select(1, minimal=False)})"
                if self.maybe(0.5):
                    s += ";"
                emit(s)

            # Mutate prior ones
            for _ in range(400):
                if not self.corpus:
                    break
                base = self.one_of(self.corpus)
                emit(self.mutate_stmt(base))

        else:
            # Wave 4: DDL focus + views + alter/drop + mutations
            bulk_generate(self.gen_create_table, 700)
            bulk_generate(self.gen_create_index, 700)
            bulk_generate(self.gen_create_view, 400)
            bulk_generate(self.gen_alter, 400)
            bulk_generate(self.gen_drop, 600)

            # Mixed: WITH + SELECT
            for _ in range(300):
                emit(self.gen_select(0, minimal=False))

            # Mutations
            for _ in range(600):
                if not self.corpus:
                    break
                base = self.one_of(self.corpus)
                emit(self.mutate_stmt(base))

        # Remember some for future mutations
        if self.maybe(0.8):
            sample = stmts[: min(500, len(stmts))]
            self.corpus.extend(sample)

        return stmts

    def _emit_unique(self, dest, s):
        # Deduplicate cheaply
        key = hash(s)
        if key not in self.seen:
            self.seen.add(key)
            dest.append(s)


_STATE = _FuzzerState()


def fuzz(parse_sql):
    # Up to 4 waves; one parse_sql call per fuzz() to maximize efficiency bonus
    wave = _STATE.call_idx
    if wave >= 4:
        return False

    # Generate batch for current wave
    stmts = _STATE.gen_wave_statements(wave)

    # Shuffle to diversify early coverage in parsing loop
    _STATE.r.shuffle(stmts)

    # Execute
    parse_sql(stmts)

    _STATE.call_idx += 1
    # Stop after last wave
    return _STATE.call_idx < 4


# -----------------------------
# Solution wrapper
# -----------------------------

class Solution:
    def solve(self, resources_path: str) -> dict:
        # Return code string of this file (the evaluator will import and call fuzz)
        # Since this method is executed in a different context, reconstruct the
        # module code by reading __file__ if possible, else compose string manually.
        # To avoid file IO complexity, we return a concatenation of the functions defined above.
        # We'll embed the current module's source by reconstructing from globals.
        import inspect

        # Collect source of current module
        src_parts = []
        # Ensure deterministic order
        targets = [
            _FastRandom,
            _FuzzerState,
            " _STATE_PLACEHOLDER ",  # placeholder to inject state init
            fuzz,
            Solution,  # include minimal wrapper; but avoid recursion
        ]
        # We will directly return the entire source of this module by inspecting globals
        # However, inspect.getsource(Solution) would include this solve method which would then include itself
        # To avoid complexity, just rebuild code as a string identical to top-level definitions

        # Build the code string manually using the already defined functions' sources
        imports_code = "import os\nimport random\nimport string\nimport time\n\n\n"
        fr_code = inspect.getsource(_FastRandom) + "\n\n"
        fuzzer_code = inspect.getsource(_FuzzerState) + "\n\n"
        state_code = "_STATE = _FuzzerState()\n\n\n"
        fuzz_code = inspect.getsource(fuzz) + "\n\n"
        sol_class_code = "class Solution:\n" \
                         "    def solve(self, resources_path: str) -> dict:\n" \
                         "        return {'code': open(__file__, 'r', encoding='utf-8').read()}\n"

        # But reading __file__ may not work in the evaluator; Instead, include code directly
        full_code = imports_code + fr_code + fuzzer_code + state_code + fuzz_code
        # Provide a minimal Solution class to satisfy API in the returned code
        full_code += "class Solution:\n" \
                     "    def solve(self, resources_path: str) -> dict:\n" \
                     "        return {'code': open(__file__, 'r', encoding='utf-8').read()}\n"

        return {"code": full_code}