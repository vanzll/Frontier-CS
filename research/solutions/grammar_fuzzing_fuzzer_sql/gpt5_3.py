import os
import sys
import random
import string
import re
import time


class Solution:
    def solve(self, resources_path: str) -> dict:
        code = f'''import os
import sys
import random
import string
import re
import time

RESOURCES_PATH = {repr(resources_path)}

# Global state holder
_STATE = None

def _safe_int(n, default):
    try:
        return int(n)
    except Exception:
        return default

class _SQLFuzzerState:
    def __init__(self, resources_path):
        self.resources_path = resources_path
        self.rng = random.Random(time.time() ^ (os.getpid() << 16))
        self.start_time = None
        self.time_budget = 58.0  # seconds (slightly under 60 to avoid timeout)
        self.round = 0
        self.max_rounds = 12  # keep parse_sql calls low to get efficiency bonus
        self.corpus = []
        self.corpus_limit = 4000  # store a subset for mutation
        self.tables = ["users", "orders", "products", "t", "logs", "x", "y"]
        self.table_counter = 0
        self.index_counter = 0
        self.functions_numeric = ["ABS", "CEIL", "FLOOR", "ROUND", "EXP", "LN", "LOG", "POWER", "SQRT", "SIN", "COS", "TAN", "RANDOM"]
        self.functions_string = ["LOWER", "UPPER", "SUBSTR", "SUBSTRING", "TRIM", "LTRIM", "RTRIM", "REPLACE", "CONCAT", "LENGTH"]
        self.functions_datetime = ["CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP", "DATE", "TIME"]
        self.operators = ["+", "-", "*", "/", "%", "^", "||", "AND", "OR", "NOT", "=", "<>", "!=", "<", ">", "<=", ">=", "LIKE", "IN", "IS", "IS NOT", "BETWEEN"]
        self.punct = [",", "(", ")", ".", ":", "::", "[", "]"]
        self.datatypes = [
            "INT", "INTEGER", "SMALLINT", "BIGINT",
            "NUMERIC", "NUMERIC(10,2)", "DECIMAL(18,6)", "REAL", "DOUBLE", "FLOAT",
            "BOOLEAN", "BOOL",
            "CHAR(10)", "NCHAR(8)", "VARCHAR(20)", "VARCHAR(255)", "NVARCHAR(50)", "TEXT",
            "DATE", "TIME", "TIMESTAMP", "DATETIME", "BLOB"
        ]
        self.bool_literals = ["TRUE", "FALSE"]
        self.null_literals = ["NULL"]
        self.keywords = set()
        self.reserved = set()
        self._load_keywords()
        self.kw_list = list(sorted(self.keywords)) or [
            "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER",
            "TABLE", "INDEX", "VIEW", "TRIGGER", "WHERE", "FROM", "JOIN", "LEFT", "RIGHT", "FULL",
            "OUTER", "INNER", "ON", "USING", "GROUP", "BY", "HAVING", "ORDER",
            "LIMIT", "OFFSET", "VALUES", "INTO", "DISTINCT", "ALL", "UNION",
            "INTERSECT", "EXCEPT", "WITH", "AS", "CASE", "WHEN", "THEN", "ELSE", "END",
            "PRIMARY", "KEY", "FOREIGN", "REFERENCES", "CHECK", "DEFAULT", "NOT", "NULL", "UNIQUE",
            "ASC", "DESC", "IS", "BETWEEN", "LIKE", "EXISTS", "CAST"
        ]
        # Characters for fuzzing whitespaces/comments
        self.ws_chars = [" ", "\\t", "\\n", "\\r", "\\v", "\\f"]
        # Prebuild an identifier pool for quicker generation
        self.ident_pool = self._build_ident_pool()
        # Prebuild some literal pools
        self.string_pool = self._build_string_pool()
        self.number_pool = self._build_number_pool()

    def _append_sys_path(self):
        try:
            if self.resources_path and os.path.isdir(self.resources_path):
                if self.resources_path not in sys.path:
                    sys.path.insert(0, self.resources_path)
        except Exception:
            pass

    def _load_keywords(self):
        # Defaults + try to import tokenizer or parse grammar to discover keywords recognized by parser
        default_keywords = {
            "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER",
            "TABLE", "INDEX", "VIEW", "TRIGGER", "WHERE", "FROM", "JOIN", "LEFT", "RIGHT", "FULL",
            "OUTER", "INNER", "ON", "USING", "GROUP", "BY", "HAVING", "ORDER",
            "LIMIT", "OFFSET", "VALUES", "INTO", "DISTINCT", "ALL", "UNION",
            "INTERSECT", "EXCEPT", "WITH", "AS", "CASE", "WHEN", "THEN", "ELSE", "END",
            "PRIMARY", "KEY", "FOREIGN", "REFERENCES", "CHECK", "DEFAULT", "NOT", "NULL", "UNIQUE",
            "ASC", "DESC", "IS", "BETWEEN", "LIKE", "EXISTS", "CAST", "AND", "OR"
        }
        self.keywords.update(default_keywords)
        # Try to import tokenizer from target
        try:
            self._append_sys_path()
            import sql_engine.tokenizer as tok  # type: ignore
            # Try to find sets/dicts of keywords
            for name in dir(tok):
                obj = getattr(tok, name)
                if isinstance(obj, (set, frozenset)):
                    for k in obj:
                        if isinstance(k, str) and k.isupper():
                            self.keywords.add(k)
                elif isinstance(obj, dict):
                    for k in obj.keys():
                        if isinstance(k, str) and k.isupper():
                            self.keywords.add(k)
        except Exception:
            pass
        # Try to parse grammar file for uppercase tokens
        try:
            gpath = os.path.join(self.resources_path, "sql_grammar.txt")
            if os.path.isfile(gpath):
                with open(gpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                tokens = set(re.findall(r"\\b[A-Z][A-Z_0-9]*\\b", content))
                # Filter out obvious BNF markers or rules
                # Keep tokens that look like SQL keywords or types
                for t in tokens:
                    if len(t) >= 2:
                        self.keywords.add(t)
        except Exception:
            pass

    def _build_ident_pool(self):
        base = [
            "a", "b", "c", "t", "x", "y", "z",
            "id", "name", "value", "flag", "price", "qty", "count",
            "created_at", "updated_at", "ts", "dt"
        ]
        # Add keyword-like but quoted identifiers and mixcases
        pool = set(base)
        for k in list(self.keywords)[:60]:
            pool.add(k.lower())
            pool.add(k.title())
            pool.add(k + "_col")
        # Add numeric suffixes
        for b in list(pool):
            for i in range(3):
                pool.add(f"{b}{i}")
        # Add random generated identifiers
        letters = string.ascii_lowercase + "_"
        for _ in range(200):
            l = self.rng.randint(1, 10)
            s = self.rng.choice(letters)
            for _j in range(l - 1):
                s += self.rng.choice(letters + string.digits)
            pool.add(s)
        return list(pool)

    def _build_string_pool(self):
        pool = set()
        base = ["hello", "world", "foo", "bar", "baz", "lorem ipsum", "O'Reilly", "line\\nfeed", "tab\\tbed", "percent%sign", "underscore_", "space space", "unicode_µ", "quote\\\""]
        for s in base:
            pool.add(s)
            pool.add(s.upper())
        # Long and edge cases
        pool.add("")
        pool.add("a" * 64)
        pool.add("b" * 255)
        pool.add("'" * 3)
        pool.add("\\" * 4 + "n")
        pool.add(";" * 2)
        return list(pool)

    def _build_number_pool(self):
        nums = [
            "0", "1", "-1", "2", "10", "-10", "1234567890", "-999999999",
            "3.14159", "-2.71828", ".5", "0.", "1e10", "-1e-5", "1.0E+309", "-inf", "NaN",
            "0xFF", "0xdeadBEEF"
        ]
        return nums

    def now(self):
        return time.time()

    def maybe(self, p):
        return self.rng.random() < p

    def choose(self, seq):
        if not seq:
            return ""
        return self.rng.choice(seq)

    def weighted_choice(self, items):
        # items: list of (elem, weight)
        total = sum(w for _, w in items)
        r = self.rng.random() * (total if total > 0 else 1.0)
        upto = 0.0
        for elem, w in items:
            upto += w
            if r <= upto:
                return elem
        return items[-1][0]

    def rand_ident(self, allow_quoted=True):
        ident = self.choose(self.ident_pool)
        style = self.rng.randint(0, 7)
        if not allow_quoted:
            return ident
        if style == 0:
            return ident
        elif style == 1:
            return f'"{ident}"'
        elif style == 2:
            return f'`{ident}`'
        elif style == 3:
            return f'[{ident}]'
        elif style == 4:
            # dot qualified
            return f"{self.choose(self.ident_pool)}.{ident}"
        elif style == 5:
            # schema.table.column
            return f"{self.choose(self.ident_pool)}.{self.choose(self.ident_pool)}.{ident}"
        else:
            return ident

    def rand_string_literal(self):
        s = self.choose(self.string_pool)
        style = self.rng.randint(0, 7)
        if style == 0:
            # single quotes with escaping
            s2 = s.replace("'", "''")
            return f"'{s2}'"
        elif style == 1:
            return f'"{s}"'
        elif style == 2:
            s2 = s.replace("'", "''")
            return f"E'{s2}'"
        elif style == 3:
            return "''"  # empty
        elif style == 4:
            # unterminated sometimes for tokenizer exploration
            return "'" + s if self.maybe(0.1) else f"'{s}'"
        elif style == 5:
            return f"N'{s}'"
        elif style == 6:
            return "q'[weird ] bracket ] quote]'"
        else:
            s2 = s.replace("'", "''")
            return f"'{s2}'"

    def rand_number_literal(self):
        return self.choose(self.number_pool)

    def rand_bool_or_null(self):
        return self.choose(self.bool_literals + self.null_literals)

    def rand_literal(self):
        t = self.rng.randint(0, 5)
        if t in (0, 1):
            return self.rand_number_literal()
        elif t in (2, 3):
            return self.rand_string_literal()
        elif t == 4:
            return self.rand_bool_or_null()
        else:
            # special tokens to exercise tokenizer and parser
            return self.choose(["CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP", "DEFAULT", "NULL"])

    def rand_comment(self):
        style = self.rng.randint(0, 4)
        if style == 0:
            return "-- " + self.choose(self.string_pool)
        elif style == 1:
            return "/* " + self.choose(self.string_pool) + " */"
        elif style == 2:
            return "/* nested /* comment */ end */"
        elif style == 3:
            # sometimes unclosed comment
            return "/* " + self.choose(self.string_pool)
        else:
            return "--"  # minimal

    def rand_ws(self):
        # Combine spaces, tabs, newlines randomly
        n = self.rng.randint(1, 5)
        s = ""
        for _ in range(n):
            s += self.rng.choice([" ", "\\t", "\\n", "\\r"])
        # Convert escape sequences to actual whitespace
        s = s.replace("\\t", "\\t").replace("\\n", "\\n").replace("\\r", "\\r")
        return s

    def expr(self, depth=0):
        if depth > 3:
            return self.expr_leaf()
        choice = self.rng.randint(0, 11)
        if choice <= 3:
            # binary operation
            left = self.expr(depth + 1)
            right = self.expr(depth + 1)
            op = self.choose(["+", "-", "*", "/", "%", "|", "&", "^", "||", "=", "<>", "!=", "<", ">", "<=", ">="])
            return f"({left} {op} {right})"
        elif choice == 4:
            return f"({self.expr(depth + 1)})"
        elif choice == 5:
            return f"-{self.expr(depth + 1)}"
        elif choice == 6:
            # function call
            fname = self.choose(self.functions_numeric + self.functions_string + ["COALESCE", "NULLIF", "GREATEST", "LEAST", "IFNULL", "ISNULL"])
            argc = self.rng.randint(1, 3)
            args = ", ".join(self.expr(depth + 1) for _ in range(argc))
            return f"{fname}({args})"
        elif choice == 7:
            # CASE WHEN
            n = self.rng.randint(1, 3)
            parts = []
            for _ in range(n):
                cond = self.expr(depth + 1)
                val = self.expr(depth + 1)
                parts.append(f"WHEN {cond} THEN {val}")
            else_val = self.expr(depth + 1)
            return f"(CASE { ' '.join(parts) } ELSE {else_val} END)"
        elif choice == 8:
            # IN list
            n = self.rng.randint(1, 5)
            inside = ", ".join(self.expr_leaf() for _ in range(n))
            return f"({self.expr(depth + 1)} IN ({inside}))"
        elif choice == 9:
            # BETWEEN
            a = self.expr(depth + 1)
            b = self.expr_leaf()
            c = self.expr_leaf()
            return f"({a} BETWEEN {b} AND {c})"
        elif choice == 10:
            # EXISTS subquery
            sub = self.select_stmt(depth + 1, minimal=True)
            return f"EXISTS ({sub})"
        else:
            return self.expr_leaf()

    def expr_leaf(self):
        t = self.rng.randint(0, 7)
        if t in (0, 1):
            return self.rand_literal()
        elif t in (2, 3):
            return self.rand_ident()
        elif t == 4:
            # qualified
            return f"{self.rand_ident(False)}.{self.rand_ident(False)}"
        elif t == 5:
            # CAST
            return f"CAST({self.expr(3)} AS {self.choose(self.datatypes)})"
        elif t == 6:
            return "NULL"
        else:
            return self.rand_number_literal()

    def column_list(self, minc=1, maxc=5):
        n = self.rng.randint(minc, maxc)
        cols = []
        for _ in range(n):
            cols.append(self.rand_ident(False))
        return ", ".join(cols)

    def table_source(self, allow_sub=True, depth=0):
        if allow_sub and depth < 2 and self.maybe(0.2):
            subsel = self.select_stmt(depth + 1, minimal=True)
            alias = self.rand_ident(False)
            return f"({subsel}) AS {alias}"
        else:
            tname = self.rand_table_name()
            alias = self.rand_ident(False) if self.maybe(0.4) else ""
            return f"{tname} {('AS ' + alias) if alias else ''}".strip()

    def join_clause(self, depth=0):
        # Build a chain of joins
        left = self.table_source(True, depth + 1)
        m = self.rng.randint(0, 2)
        for _ in range(m):
            jt = self.choose(["JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN", "INNER JOIN", "LEFT OUTER JOIN", "RIGHT OUTER JOIN"])
            right = self.table_source(True, depth + 1)
            cond = ""
            if self.maybe(0.6):
                cond = "ON " + self.expr(depth + 1)
            else:
                cond = "USING (" + self.column_list(1, 3) + ")"
            left = f"{left} {jt} {right} {cond}"
        return left

    def order_item(self, depth=0):
        e = self.expr(depth + 1)
        direction = self.choose(["ASC", "DESC", ""])
        return f"{e} {direction}".strip()

    def select_list(self, depth=0):
        n = self.rng.randint(1, 6)
        items = []
        for _ in range(n):
            if self.maybe(0.1):
                items.append("*")
                continue
            e = self.expr(depth + 1)
            if self.maybe(0.3):
                alias = self.rand_ident(False)
                items.append(f"{e} AS {alias}")
            else:
                items.append(e)
        return ", ".join(items)

    def select_stmt(self, depth=0, minimal=False):
        distinct = "DISTINCT " if self.maybe(0.2) else ""
        lst = self.select_list(depth + 1)
        s = f"SELECT {distinct}{lst}"
        if self.maybe(0.9):  # usually have FROM
            frm = self.join_clause(depth + 1)
            s += f" FROM {frm}"
        if not minimal and self.maybe(0.7):
            s += f" WHERE {self.expr(depth + 1)}"
        if not minimal and self.maybe(0.5):
            n = self.rng.randint(1, 3)
            group_items = ", ".join(self.expr(depth + 1) for _ in range(n))
            s += f" GROUP BY {group_items}"
            if self.maybe(0.5):
                s += f" HAVING {self.expr(depth + 1)}"
        if not minimal and self.maybe(0.6):
            n = self.rng.randint(1, 3)
            order_items = ", ".join(self.order_item(depth + 1) for _ in range(n))
            s += f" ORDER BY {order_items}"
        if not minimal and self.maybe(0.5):
            s += f" LIMIT {_safe_int(self.rng.randint(0, 1000), 10)}"
            if self.maybe(0.5):
                s += f" OFFSET {_safe_int(self.rng.randint(0, 1000), 0)}"
        # set operations
        if not minimal and self.maybe(0.3):
            op = self.choose(["UNION", "UNION ALL", "INTERSECT", "EXCEPT"])
            right = self.select_stmt(depth + 1, minimal=True)
            s = f"({s}) {op} ({right})"
        # CTE
        if not minimal and self.maybe(0.25):
            cte_name = self.rand_ident(False)
            cte_body = self.select_stmt(depth + 1, minimal=True)
            s = f"WITH {cte_name} AS ({cte_body}) {s}"
        # EXPLAIN
        if self.maybe(0.05):
            s = "EXPLAIN " + s
        return s

    def value_list(self, depth=0, minc=1, maxc=6):
        n = self.rng.randint(minc, maxc)
        vals = []
        for _ in range(n):
            vals.append(self.expr(depth + 1))
        return "(" + ", ".join(vals) + ")"

    def insert_stmt(self, depth=0):
        tbl = self.rand_table_name()
        col_list = ""
        if self.maybe(0.7):
            col_list = f"({self.column_list(1, self.rng.randint(2, 6))})"
        if self.maybe(0.6):
            # VALUES clause
            rows = self.rng.randint(1, 5)
            values = ", ".join(self.value_list(depth + 1) for _ in range(rows))
            stmt = f"INSERT INTO {tbl} {col_list} VALUES {values}"
        else:
            # INSERT from subquery
            subq = self.select_stmt(depth + 1, minimal=True)
            stmt = f"INSERT INTO {tbl} {col_list} {subq}"
        return stmt

    def update_stmt(self, depth=0):
        tbl = self.rand_table_name()
        assigns = []
        n = self.rng.randint(1, 5)
        for _ in range(n):
            col = self.rand_ident(False)
            val = self.expr(depth + 1)
            assigns.append(f"{col} = {val}")
        s = f"UPDATE {tbl} SET " + ", ".join(assigns)
        if self.maybe(0.7):
            if self.maybe(0.4):
                s += f" FROM {self.join_clause(depth + 1)}"
            s += f" WHERE {self.expr(depth + 1)}"
        return s

    def delete_stmt(self, depth=0):
        tbl = self.rand_table_name()
        s = f"DELETE FROM {tbl}"
        if self.maybe(0.7):
            if self.maybe(0.3):
                s += f" USING {self.join_clause(depth + 1)}"
            s += f" WHERE {self.expr(depth + 1)}"
        return s

    def create_table_stmt(self):
        name = self.new_table_name()
        ncols = self.rng.randint(1, 8)
        cols = []
        colnames = []
        for _ in range(ncols):
            cname = self.rand_ident(False)
            colnames.append(cname)
            ctype = self.choose(self.datatypes)
            constraints = []
            if self.maybe(0.3):
                constraints.append("NOT NULL")
            if self.maybe(0.2):
                constraints.append("UNIQUE")
            if self.maybe(0.2):
                constraints.append(f"DEFAULT {self.expr(2)}")
            coldef = f"{cname} {ctype} {' '.join(constraints)}".strip()
            cols.append(coldef)
        # Table constraints
        tcon = []
        if self.maybe(0.4):
            key_cols = ", ".join(self.rng.sample(colnames, k=max(1, self.rng.randint(1, min(3, len(colnames))))))
            tcon.append(f"PRIMARY KEY ({key_cols})")
        if self.maybe(0.2) and len(colnames) >= 2:
            uq_cols = ", ".join(self.rng.sample(colnames, k=2))
            tcon.append(f"UNIQUE ({uq_cols})")
        if self.maybe(0.2):
            tcon.append(f"CHECK ({self.expr(2)})")
        defs = cols + tcon
        stmt = f"CREATE TABLE {name} (" + ", ".join(defs) + ")"
        return stmt

    def alter_table_stmt(self):
        tbl = self.rand_table_name()
        action_type = self.rng.randint(0, 4)
        if action_type == 0:
            # ADD COLUMN
            cname = self.rand_ident(False)
            ctype = self.choose(self.datatypes)
            return f"ALTER TABLE {tbl} ADD COLUMN {cname} {ctype}"
        elif action_type == 1:
            # DROP COLUMN
            cname = self.rand_ident(False)
            return f"ALTER TABLE {tbl} DROP COLUMN {cname}"
        elif action_type == 2:
            # RENAME COLUMN
            old = self.rand_ident(False)
            new = self.rand_ident(False)
            return f"ALTER TABLE {tbl} RENAME COLUMN {old} TO {new}"
        elif action_type == 3:
            # RENAME TABLE
            newtbl = self.rand_ident(False)
            return f"ALTER TABLE {tbl} RENAME TO {newtbl}"
        else:
            # ADD CONSTRAINT
            cname = self.rand_ident(False)
            return f"ALTER TABLE {tbl} ADD CONSTRAINT {cname} CHECK ({self.expr(2)})"

    def drop_stmt(self):
        typ = self.choose(["TABLE", "INDEX", "VIEW"])
        name = self.rand_ident(False)
        opt = " IF EXISTS" if self.maybe(0.4) else ""
        return f"DROP {typ}{opt} {name}"

    def create_index_stmt(self):
        idxname = self.new_index_name()
        tbl = self.rand_table_name()
        cols = self.column_list(1, 4)
        unique = "UNIQUE " if self.maybe(0.4) else ""
        where = f" WHERE {self.expr(2)}" if self.maybe(0.3) else ""
        return f"CREATE {unique}INDEX {idxname} ON {tbl} ({cols}){where}"

    def begin_commit_stmt(self):
        return self.choose(["BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT sp", "RELEASE sp"])

    def rand_table_name(self):
        # Use known tables often, sometimes random
        if self.maybe(0.8) and self.tables:
            return self.choose(self.tables)
        return self.rand_ident(False)

    def new_table_name(self):
        self.table_counter += 1
        name = f"t{self.table_counter}"
        self.tables.append(name)
        return name

    def new_index_name(self):
        self.index_counter += 1
        return f"idx{self.index_counter}"

    def tokenizer_fuzz_stmt(self):
        # Build a noisy token stream prefixed with a keyword to pass to parser
        prefix = self.choose(["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"])
        tokens = []
        length = self.rng.randint(5, 30)
        for _ in range(length):
            t_choice = self.rng.randint(0, 12)
            if t_choice in (0, 1):
                tokens.append(self.choose(self.kw_list))
            elif t_choice in (2, 3):
                tokens.append(self.rand_ident())
            elif t_choice in (4, 5):
                tokens.append(self.rand_string_literal())
            elif t_choice in (6, 7):
                tokens.append(self.rand_number_literal())
            elif t_choice == 8:
                tokens.append(self.choose(self.operators))
            elif t_choice == 9:
                tokens.append(self.choose(self.punct))
            elif t_choice == 10:
                tokens.append(self.rand_comment())
            else:
                tokens.append(self.rand_bool_or_null())
        junk = " ".join(tokens)
        # Sprinkle whitespace/comments
        if self.maybe(0.4):
            junk = self.rand_comment() + " " + junk
        return f"{prefix} {junk}"

    def generate_statement(self):
        # Choose statement type with weights
        kind = self.weighted_choice([
            ("SELECT", 40),
            ("INSERT", 12),
            ("UPDATE", 10),
            ("DELETE", 8),
            ("CREATE_TABLE", 8),
            ("ALTER_TABLE", 5),
            ("CREATE_INDEX", 4),
            ("DROP", 4),
            ("TXN", 2),
            ("TOKEN", 7)
        ])
        try:
            if kind == "SELECT":
                return self.select_stmt(0, minimal=False)
            elif kind == "INSERT":
                return self.insert_stmt(0)
            elif kind == "UPDATE":
                return self.update_stmt(0)
            elif kind == "DELETE":
                return self.delete_stmt(0)
            elif kind == "CREATE_TABLE":
                return self.create_table_stmt()
            elif kind == "ALTER_TABLE":
                return self.alter_table_stmt()
            elif kind == "CREATE_INDEX":
                return self.create_index_stmt()
            elif kind == "DROP":
                return self.drop_stmt()
            elif kind == "TXN":
                return self.begin_commit_stmt()
            elif kind == "TOKEN":
                return self.tokenizer_fuzz_stmt()
        except Exception:
            # Fallback to a simple select on error to avoid generation crashes
            return "SELECT 1"
        return "SELECT 1"

    def mutate(self, sql):
        s = sql
        # Randomly change case
        if self.maybe(0.6):
            s = "".join(ch.upper() if self.maybe(0.5) else ch.lower() for ch in s)
        # Insert random whitespace/comments
        if self.maybe(0.7):
            insertions = self.rng.randint(1, 3)
            for _ in range(insertions):
                pos = self.rng.randint(0, max(0, len(s) - 1))
                s = s[:pos] + " " + self.rand_comment() + " " + s[pos:]
        # Replace some keywords with neighbors or duplicates
        if self.maybe(0.5):
            replacements = {
                "SELECT": self.choose(["SELECT", "SELECT DISTINCT", "SELECT ALL", "sElEcT"]),
                "JOIN": self.choose(["JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN", "INNER JOIN"]),
                "WHERE": self.choose(["WHERE", "QUALIFY", "FILTER", "WHERE NOT"]),
                "GROUP BY": self.choose(["GROUP BY", "GROUP", "GROUP  BY"]),
                "ORDER BY": self.choose(["ORDER BY", "ORDER  BY"]),
                "INSERT": self.choose(["INSERT", "UPSERT", "REPLACE"]),
                "UPDATE": self.choose(["UPDATE", "UPDATE OR REPLACE"]),
                "DELETE": self.choose(["DELETE", "DELETE FROM"]),
                "CREATE": self.choose(["CREATE", "CREATE OR REPLACE", "CREATE TEMP"]),
            }
            for k, v in replacements.items():
                if k in s and self.maybe(0.3):
                    s = s.replace(k, v)
        # Randomly wrap in parentheses or add extraneous parentheses
        if self.maybe(0.4):
            s = "(" + s + ")"
        # Randomly truncate or add dangling tokens
        if self.maybe(0.15):
            cut = self.rng.randint(0, len(s))
            s = s[:cut]
        if self.maybe(0.2):
            s += " " + self.choose(["; ;", "???", "/*", "-- end", "::json", "::text"])
        return s

    def build_batch(self, round_idx, time_left):
        batch = []
        # Dynamically size batch to balance time and efficiency
        if round_idx == 0:
            base = 1900
        else:
            base = 1500
        if time_left < 15:
            base = max(600, int(base * 0.5))
        elif time_left < 30:
            base = int(base * 0.8)
        # Phase-specific composition
        if round_idx == 0:
            # tokenizer-heavy plus simple selects
            for _ in range(int(base * 0.6)):
                batch.append(self.tokenizer_fuzz_stmt())
            for _ in range(int(base * 0.4)):
                batch.append(self.select_stmt(0, minimal=False))
        else:
            # Mixed DDL/DML/Queries
            ddl_cnt = int(base * 0.10)
            dml_cnt = int(base * 0.20)
            sel_cnt = int(base * 0.55)
            other_cnt = base - (ddl_cnt + dml_cnt + sel_cnt)
            # DDL
            for _ in range(ddl_cnt):
                # Mostly CREATE TABLE with some ALTER
                if self.maybe(0.7):
                    batch.append(self.create_table_stmt())
                else:
                    batch.append(self.alter_table_stmt())
            # DML INSERT/UPDATE/DELETE
            for _ in range(dml_cnt):
                t = self.rng.random()
                if t < 0.5:
                    batch.append(self.insert_stmt(0))
                elif t < 0.75:
                    batch.append(self.update_stmt(0))
                else:
                    batch.append(self.delete_stmt(0))
            # SELECTs
            for _ in range(sel_cnt):
                batch.append(self.select_stmt(0, minimal=False))
            # Others
            for _ in range(other_cnt):
                if self.maybe(0.5):
                    batch.append(self.create_index_stmt())
                else:
                    batch.append(self.drop_stmt())
            # Mutations from corpus and this batch
            to_mutate = []
            if self.corpus:
                sample = self.rng.sample(self.corpus, k=min(len(self.corpus), max(20, int(base * 0.1))))
                to_mutate.extend(sample)
            sample2 = self.rng.sample(batch, k=min(len(batch), max(20, int(base * 0.1))))
            to_mutate.extend(sample2)
            for s in to_mutate:
                if self.maybe(0.8):
                    batch.append(self.mutate(s))
        # Optional sprinkle: standalone transaction and explain statements
        extras = self.rng.randint(5, 20)
        for _ in range(extras):
            if self.maybe(0.5):
                batch.append(self.begin_commit_stmt())
            else:
                batch.append("EXPLAIN " + self.select_stmt(0, minimal=True))
        # Cap batch size to avoid extremes
        max_batch = 4000
        if len(batch) > max_batch:
            batch = self.rng.sample(batch, max_batch)
        return batch

    def update_corpus(self, batch):
        # Keep a subset of statements for future mutations
        if not batch:
            return
        keep_n = min(len(batch), 300)
        samples = self.rng.sample(batch, k=keep_n)
        self.corpus.extend(samples)
        if len(self.corpus) > self.corpus_limit:
            # Randomly drop some to keep size
            self.rng.shuffle(self.corpus)
            self.corpus = self.corpus[: self.corpus_limit]

    def fuzz_once(self, parse_sql):
        if self.start_time is None:
            self.start_time = self.now()
        elapsed = self.now() - self.start_time
        time_left = self.time_budget - elapsed
        if time_left <= 0:
            return False
        # Build batch for this round
        batch = self.build_batch(self.round, time_left)
        # Execute through parser once (efficiency bonus)
        try:
            parse_sql(batch)
        except Exception:
            # parse_sql is supposed to catch internally, but be safe
            pass
        # Update corpus with sample of batch
        self.update_corpus(batch)
        self.round += 1
        # Continue while time left and rounds under max
        elapsed = self.now() - self.start_time
        time_left = self.time_budget - elapsed
        if self.round >= self.max_rounds or time_left <= 0:
            return False
        return True


def fuzz(parse_sql):
    global _STATE
    if _STATE is None:
        _STATE = _SQLFuzzerState(RESOURCES_PATH)
    return _STATE.fuzz_once(parse_sql)
'''
        return {"code": code}