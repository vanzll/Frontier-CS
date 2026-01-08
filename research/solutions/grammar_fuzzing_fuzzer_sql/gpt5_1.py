import os
import random
import re
import string
import time
from typing import List, Optional, Tuple, Dict, Any


class Solution:
    def solve(self, resources_path: str) -> dict:
        code = r'''
import random
import re
import string
import time
import os
from typing import List, Optional

# Global fuzzer instance placeholder
_fuzzer_instance = None

def fuzz(parse_sql):
    global _fuzzer_instance
    if _fuzzer_instance is None:
        _fuzzer_instance = _Fuzzer()
    return _fuzzer_instance.fuzz(parse_sql)


class _Fuzzer:
    def __init__(self):
        self.rnd = random.Random()
        # Seed with high entropy, but reproducible run-level if env is set
        seed_env = os.environ.get("SQL_FUZZ_SEED")
        if seed_env:
            try:
                self.rnd.seed(int(seed_env))
            except Exception:
                self.rnd.seed(seed_env)
        else:
            self.rnd.seed(int(time.time() * 1_000_000) ^ os.getpid())
        self.call_count = 0
        self.max_calls = 3  # Limit number of fuzz() calls to boost efficiency bonus
        self.start_time = None
        self.time_budget = 55.0  # seconds; leave some margin
        self.corpus: List[str] = []
        self.keyword_set = set()
        self.operator_set = set()
        self.punctuations = [",", ";", "(", ")", ".", "*", "+", "-", "/", "%", "=", "<>", "!=", "<", "<=", ">", ">=", "||"]
        self._init_keywords_from_tokenizer()
        self._init_keywords_from_grammar()
        self._init_defaults()
        self._prepare_seed_corpus()

    def _init_keywords_from_tokenizer(self):
        # Try to introspect tokenizer to discover keywords/operators if available
        try:
            import resources.sql_engine.tokenizer as tok
            # Different implementations might expose different names
            kw = set()
            for name in ("KEYWORDS", "RESERVED", "keywords", "SQL_KEYWORDS"):
                if hasattr(tok, name):
                    obj = getattr(tok, name)
                    if isinstance(obj, (set, list, tuple, dict)):
                        if isinstance(obj, dict):
                            kw.update(map(str, obj.keys()))
                        else:
                            kw.update(map(str, obj))
            # Merge found keywords, keep uppercase versions
            if kw:
                for k in kw:
                    k2 = str(k)
                    if k2:
                        self.keyword_set.add(k2.upper())
            # Operators and punctuation
            ops = set()
            for name in ("OPERATORS", "OPERATORS_SET", "SYMBOLS", "PUNCTUATION", "PUNCTUATIONS"):
                if hasattr(tok, name):
                    obj = getattr(tok, name)
                    if isinstance(obj, (set, list, tuple)):
                        for x in obj:
                            ops.add(str(x))
            if ops:
                self.operator_set.update(ops)
        except Exception:
            # Ignore if module is not importable
            pass

    def _init_keywords_from_grammar(self):
        # Attempt to read grammar file for additional keywords
        paths = [
            os.path.join("resources", "sql_grammar.txt"),
            os.path.join(os.path.dirname(__file__), "resources", "sql_grammar.txt")
        ]
        grammar_text = ""
        for p in paths:
            try:
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        grammar_text = f.read()
                        break
            except Exception:
                continue
        if grammar_text:
            # Very rough extraction: uppercase words likely to be keywords
            # Avoid capturing non-keyword uppercase like <EXPR> by ignoring <...>
            cleaned = re.sub(r"<[^>]+>", " ", grammar_text)
            for token in re.findall(r"\b[A-Z][A-Z_]*\b", cleaned):
                if token not in {"BNF", "SQL", "AND", "OR", "NOT"}:
                    # We'll add AND/OR/NOT separately anyway
                    self.keyword_set.add(token.upper())
            # Add these even if not in grammar
        # Always ensure basic SQL keyword coverage
        base_keywords = [
            "SELECT","FROM","WHERE","GROUP","BY","HAVING","ORDER","LIMIT","OFFSET",
            "INSERT","INTO","VALUES","UPDATE","SET","DELETE",
            "CREATE","TABLE","DROP","ALTER","ADD","COLUMN","RENAME","TO",
            "AS","DISTINCT","ALL",
            "INNER","LEFT","RIGHT","FULL","OUTER","JOIN","CROSS","NATURAL","ON","USING",
            "NULL","NOT","AND","OR","IS","LIKE","IN","BETWEEN","EXISTS",
            "UNION","INTERSECT","EXCEPT","ALL",
            "PRIMARY","KEY","FOREIGN","REFERENCES","UNIQUE","CHECK","DEFAULT",
            "TRUE","FALSE",
            "WITH","RECURSIVE","CASE","WHEN","THEN","ELSE","END",
            "ASC","DESC","NULLS","FIRST","LAST",
            "INDEX","IF","EXISTS","VIEW","BEGIN","COMMIT","ROLLBACK","TRANSACTION",
            "CAST"
        ]
        for k in base_keywords:
            self.keyword_set.add(k)
        # Operators
        base_ops = ["+", "-", "*", "/", "%", "=", "<>", "!=", "<", "<=", ">", ">=", "||", "^"]
        self.operator_set.update(base_ops)

    def _init_defaults(self):
        # Identifier pools and types
        self.base_table_names = [
            "t", "t0", "t1", "t2", "t3", "users", "orders", "products", "customers", "accounts", "logs",
            "events", "items", "inventory", "warehouse", "employees", "salaries", "departments",
            "sessions", "roles", "permissions", "audit", "transactions", "payments", "invoices",
            "nodes", "edges", "graph", "tree", "paths", "messages", "posts", "comments"
        ]
        self.base_column_names = [
            "id", "name", "value", "price", "qty", "quantity", "count", "amount", "total",
            "created_at", "updated_at", "timestamp", "date", "time", "active", "status", "code", "flag",
            "data", "payload", "meta", "description", "text", "title", "email", "address",
            "phone", "lat", "lng", "score", "rank", "level", "age", "height", "weight",
            "category", "type", "tag", "hash", "uid", "gid", "pid", "rid", "parent_id", "child_id"
        ]
        # Add reserved words used as identifiers to test quoting
        reserved_as_id = [
            "select", "table", "from", "where", "order", "group", "user", "index", "key", "primary", "insert",
            "update", "delete", "drop", "create", "values", "and", "or", "not", "inner", "left", "right", "join"
        ]
        self.base_column_names += reserved_as_id
        self.base_table_names += [f"table{i}" for i in range(20)]
        self.base_column_names += [f"col{i}" for i in range(50)]
        # Data types
        self.datatypes = [
            "INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT",
            "BOOLEAN", "BOOL",
            "REAL", "DOUBLE", "DOUBLE PRECISION", "FLOAT", "FLOAT4", "FLOAT8",
            "NUMERIC", "DECIMAL", "DECIMAL(10,2)", "NUMERIC(18,6)",
            "CHAR", "CHAR(10)", "NCHAR(12)",
            "VARCHAR(10)", "VARCHAR(100)", "VARCHAR(255)", "NVARCHAR(100)",
            "TEXT", "CLOB",
            "DATE", "TIME", "TIMESTAMP",
            "BLOB", "BYTEA"
        ]
        # Functions: scalar and aggregate
        self.scalar_functions = [
            "ABS", "ROUND", "FLOOR", "CEIL", "CEILING",
            "LENGTH", "LOWER", "UPPER", "TRIM", "LTRIM", "RTRIM", "SUBSTR", "SUBSTRING",
            "COALESCE", "NULLIF", "IFNULL",
            "RANDOM", "RANDOMBLOB", "HEX", "UNICODE",
            "DATE", "TIME", "DATETIME", "JULIANDAY", "STRFTIME",
            "CAST"
        ]
        self.aggregate_functions = ["COUNT", "SUM", "AVG", "MIN", "MAX", "GROUP_CONCAT"]
        self.collations = ["BINARY", "NOCASE", "RTRIM", "C"]
        self.boolean_literals = ["TRUE", "FALSE"]
        self.null_literals = ["NULL"]
        self.special_literals = ["CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP"]
        # Interesting numeric strings
        self.interesting_numbers = [
            "0", "1", "-1", "42", "-42", "2147483647", "-2147483648", "9223372036854775807", "-9223372036854775808",
            "3.14159", "-0.0", "1e10", "-1e-10", "0.00000001", "999999999999999999999999"
        ]
        # String samples
        self.interesting_strings = [
            "", "a", "abc", "A'B", "O''Reilly", "line\nbreak", "tab\ttab", "quote\"double", "back\\slash",
            "semi;colon", "percent%percent", "under_score", "dash-dash", "中文", "emoji🙂", "NULL", "TRUE", "FALSE"
        ]

    def _prepare_seed_corpus(self):
        # Predetermined statements to kickoff coverage across parser functions
        seeds = [
            "SELECT 1",
            "SELECT * FROM t",
            "SELECT a, b, c FROM users",
            "SELECT DISTINCT name FROM users WHERE age > 18",
            "SELECT u.id, o.id FROM users u INNER JOIN orders o ON u.id = o.user_id",
            "SELECT COUNT(*) FROM orders WHERE price BETWEEN 10 AND 100",
            "SELECT name FROM products WHERE name LIKE 'A%' ESCAPE '\\\\'",
            "SELECT CASE WHEN price > 100 THEN 'high' ELSE 'low' END AS category FROM products",
            "SELECT * FROM t WHERE a IN (1, 2, 3) AND b IS NOT NULL",
            "SELECT * FROM t ORDER BY a DESC, b ASC NULLS FIRST LIMIT 10 OFFSET 5",
            "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)",
            "INSERT INTO orders VALUES (DEFAULT, 1, 99.99, '2020-01-01')",
            "UPDATE users SET name = 'Bob', age = age + 1 WHERE id = 1",
            "DELETE FROM users WHERE id = 2",
            "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL, price REAL DEFAULT 0.0)",
            "CREATE TABLE users (id INT, name VARCHAR(100), email TEXT, created_at TIMESTAMP)",
            "CREATE INDEX idx_users_name ON users(name)",
            "DROP TABLE IF EXISTS temp_table",
            "ALTER TABLE users ADD COLUMN active BOOLEAN DEFAULT TRUE",
            "WITH cte AS (SELECT * FROM users) SELECT * FROM cte",
            "SELECT (SELECT COUNT(*) FROM orders) AS cnt",
            "SELECT * FROM users u LEFT JOIN orders o USING (id)",
            "SELECT a.* FROM (SELECT * FROM t) a",
            "SELECT CAST(price AS INTEGER) FROM products"
        ]
        # Also add some weird formatting and comments
        seeds += [
            "-- comment only\nSELECT 1",
            "/* block comment */ SELECT /* mid */ * FROM /* x */ t /* end */",
            "SELECT 'semi;inside' AS s",
            "SELECT 'O''Reilly' AS name",
            "SELECT -1 + +2 AS calc",
            "SELECT NULL, TRUE, FALSE",
            "SELECT 1e10 AS x, -1e-10 AS y",
            "SELECT name FROM users WHERE name LIKE '100\\%%' ESCAPE '\\\\'",
        ]
        self.corpus.extend(seeds)

    # Utility functions
    def _maybe(self, p: float) -> bool:
        return self.rnd.random() < p

    def _random_case(self, s: str) -> str:
        # Randomize the case of characters in s
        out = []
        for ch in s:
            if ch.isalpha():
                if self._maybe(0.6):
                    out.append(ch.upper() if self._maybe(0.5) else ch.lower())
                else:
                    out.append(ch)
            else:
                out.append(ch)
        return "".join(out)

    def _kw(self, s: str) -> str:
        # Return keyword possibly with randomized case
        if self._maybe(0.85):
            # either consistent case or mixed
            choice = self.rnd.randint(0, 3)
            if choice == 0:
                return s.upper()
            elif choice == 1:
                return s.lower()
            elif choice == 2:
                return s.capitalize()
            else:
                return self._random_case(s)
        else:
            return s

    def _join_tokens(self, tokens: List[str]) -> str:
        # Join tokens with random whitespace and occasional comments
        parts = []
        for i, tok in enumerate(tokens):
            parts.append(tok)
            if i < len(tokens) - 1:
                sep_choice = self.rnd.random()
                if sep_choice < 0.10:
                    # line comment
                    comment = "-- " + self._random_identifier() + "\n"
                    parts.append(comment)
                elif sep_choice < 0.18:
                    # block comment
                    comment = "/* " + self._random_identifier() + " */"
                    parts.append(" " + comment + " ")
                elif sep_choice < 0.30:
                    parts.append("\n")
                else:
                    # random spaces
                    spaces = " " * self.rnd.randint(1, 3)
                    parts.append(spaces)
        return "".join(parts)

    def _quote_identifier(self, name: str) -> str:
        # Randomly quote an identifier with different styles
        style = self.rnd.random()
        if style < 0.70:
            return name
        elif style < 0.88:
            # double quotes
            n = name.replace('"', '""')
            return f'"{n}"'
        elif style < 0.96:
            # backticks
            n = name.replace('`', '``')
            return f'`{n}`'
        else:
            # square brackets
            n = name.replace(']', ']]')
            return f'[{n}]'

    def _random_identifier_base(self) -> str:
        # Choose from a pool and mix with digits/underscores
        pool = self.base_column_names + self.base_table_names
        base = self.rnd.choice(pool)
        # Maybe append digits/underscore
        if self._maybe(0.5):
            base += self.rnd.choice(["", "_", "__"]) + str(self.rnd.randint(0, 9999))
        # Maybe insert a dash or dollar (some dialects not allow; tokenizer still handles)
        if self._maybe(0.05):
            base += self.rnd.choice(["-x", "$y"])
        return base

    def _random_identifier(self) -> str:
        return self._quote_identifier(self._random_identifier_base())

    def _random_table_name(self) -> str:
        return self._quote_identifier(self.rnd.choice(self.base_table_names) + (str(self.rnd.randint(0, 50)) if self._maybe(0.3) else ""))

    def _random_column_name(self) -> str:
        return self._quote_identifier(self.rnd.choice(self.base_column_names) + (str(self.rnd.randint(0, 100)) if self._maybe(0.3) else ""))

    def _random_number_literal(self) -> str:
        if self._maybe(0.4):
            return self.rnd.choice(self.interesting_numbers)
        # Compose random numeric
        sign = "-" if self._maybe(0.3) else ""
        if self._maybe(0.4):
            # integer
            return sign + str(self.rnd.randint(0, 10**9))
        else:
            # float
            m = str(self.rnd.randint(0, 9999))
            f = "." + str(self.rnd.randint(0, 9999)).rjust(2, "0")
            if self._maybe(0.2):
                exp = "e" + ("-" if self._maybe(0.5) else "") + str(self.rnd.randint(0, 308))
            else:
                exp = ""
            return sign + m + f + exp

    def _random_string_literal(self) -> str:
        if self._maybe(0.4):
            s = self.rnd.choice(self.interesting_strings)
        else:
            length = self.rnd.randint(0, 20)
            chars = []
            for _ in range(length):
                # mix ascii letters, digits, punctuation
                c = self.rnd.choice(string.ascii_letters + string.digits + " _-.,:/\\|!@#$%^&*()[]{}<>?~`")
                # include newline occasionally
                if c == " " and self._maybe(0.1):
                    c = "\n"
                chars.append(c)
            s = "".join(chars)
        s = s.replace("'", "''")
        return "'" + s + "'"

    def _random_literal(self) -> str:
        choice = self.rnd.random()
        if choice < 0.45:
            return self._random_number_literal()
        elif choice < 0.85:
            return self._random_string_literal()
        elif choice < 0.93:
            return self.rnd.choice(self.boolean_literals)
        elif choice < 0.98:
            return self.rnd.choice(self.null_literals)
        else:
            return self.rnd.choice(self.special_literals)

    def _random_datatype(self) -> str:
        base = self.rnd.choice(self.datatypes)
        # If datatype supports size/precision variation
        if "CHAR(" in base or "VARCHAR(" in base or "DECIMAL(" in base or "NUMERIC(" in base:
            return base
        if base in ("CHAR", "NCHAR"):
            return f"{base}({self.rnd.randint(1, 255)})"
        if base in ("VARCHAR", "NVARCHAR"):
            return f"{base}({self.rnd.randint(1, 255)})"
        if base in ("DECIMAL", "NUMERIC"):
            p = self.rnd.randint(1, 18)
            s = self.rnd.randint(0, min(6, p))
            return f"{base}({p},{s})"
        return base

    def _random_func_call(self, depth: int = 2) -> str:
        if self._maybe(0.3):
            fn = self.rnd.choice(self.aggregate_functions)
        else:
            fn = self.rnd.choice(self.scalar_functions)
        fn_kw = self._kw(fn)
        if fn.upper() == "COUNT" and self._maybe(0.3):
            arg = "*"
        else:
            argc = self.rnd.randint(0, 3) if self._maybe(0.3) else self.rnd.randint(1, 3)
            args = [self._gen_expr(depth - 1) for _ in range(argc)]
            # SUBSTRING special syntax sometimes
            arg = ", ".join(args)
        if fn.upper() == "CAST":
            # CAST(expr AS type)
            return f"{self._kw('CAST')}({self._gen_expr(depth - 1)} {self._kw('AS')} {self._random_datatype()})"
        # DISTINCT in aggregate
        if fn.upper() in self.aggregate_functions and self._maybe(0.3):
            return f"{fn_kw}({self._kw('DISTINCT')} {arg})"
        return f"{fn_kw}({arg})"

    def _random_cmp_op(self) -> str:
        ops = ["=", "!=", "<>", "<", "<=", ">", ">="]
        return self.rnd.choice(ops)

    def _random_arith_op(self) -> str:
        ops = ["+", "-", "*", "/", "%", "^", "||"]
        return self.rnd.choice(ops)

    def _gen_expr(self, depth: int = 2) -> str:
        if depth <= 0:
            base_choice = self.rnd.random()
            if base_choice < 0.35:
                return self._random_literal()
            elif base_choice < 0.65:
                # column or table.column
                if self._maybe(0.3):
                    return f"{self._random_identifier()}.{self._random_column_name()}"
                else:
                    return self._random_column_name()
            elif base_choice < 0.85:
                return self._random_func_call(0)
            else:
                return "(" + self._random_literal() + ")"
        # Non-leaf
        c = self.rnd.random()
        if c < 0.25:
            # binary arithmetic
            left = self._gen_expr(depth - 1)
            right = self._gen_expr(depth - 1)
            return f"({left} {self._random_arith_op()} {right})"
        elif c < 0.45:
            # unary
            op = self.rnd.choice(["-", "+", self._kw("NOT")])
            expr = self._gen_expr(depth - 1)
            if op.upper() == "NOT":
                return f"{op} {expr}"
            return f"({op}{expr})"
        elif c < 0.65:
            # function
            return self._random_func_call(depth - 1)
        elif c < 0.8:
            # CASE
            n_when = self.rnd.randint(1, 3)
            parts = [self._kw("CASE")]
            if self._maybe(0.3):
                parts.append(self._gen_expr(depth - 1))
            for _ in range(n_when):
                parts.append(self._kw("WHEN"))
                parts.append(self._gen_predicate(depth - 1))
                parts.append(self._kw("THEN"))
                parts.append(self._gen_expr(depth - 1))
            if self._maybe(0.6):
                parts.append(self._kw("ELSE"))
                parts.append(self._gen_expr(depth - 1))
            parts.append(self._kw("END"))
            return self._join_tokens(parts)
        elif c < 0.9:
            # CAST
            return f"{self._kw('CAST')}({self._gen_expr(depth - 1)} {self._kw('AS')} {self._random_datatype()})"
        else:
            # Parenthesized
            return f"({self._gen_expr(depth - 1)})"

    def _gen_predicate(self, depth: int = 2) -> str:
        if depth <= 0:
            # base predicate: comparison or literal boolean
            if self._maybe(0.3):
                return self._kw(self.rnd.choice(["TRUE", "FALSE"]))
            left = self._gen_expr(1)
            op = self._random_cmp_op()
            right = self._gen_expr(1)
            return f"({left} {op} {right})"
        c = self.rnd.random()
        if c < 0.25:
            left = self._gen_expr(depth - 1)
            right = self._gen_expr(depth - 1)
            return f"({left} {self._random_cmp_op()} {right})"
        elif c < 0.45:
            # BETWEEN
            e = self._gen_expr(depth - 1)
            lo = self._gen_expr(depth - 1)
            hi = self._gen_expr(depth - 1)
            not_kw = f"{self._kw('NOT')} " if self._maybe(0.5) else ""
            return f"({e} {not_kw}{self._kw('BETWEEN')} {lo} {self._kw('AND')} {hi})"
        elif c < 0.65:
            # IN
            e = self._gen_expr(depth - 1)
            if self._maybe(0.5):
                items = ", ".join(self._gen_expr(depth - 2) for _ in range(self.rnd.randint(1, 5)))
                rhs = f"({items})"
            else:
                rhs = f"({self._gen_simple_select(1)})"
            not_kw = f"{self._kw('NOT')} " if self._maybe(0.5) else ""
            return f"({e} {not_kw}{self._kw('IN')} {rhs})"
        elif c < 0.75:
            # IS NULL
            e = self._gen_expr(depth - 1)
            not_kw = f" {self._kw('NOT')}" if self._maybe(0.5) else ""
            return f"({e} {self._kw('IS')}{not_kw} {self._kw('NULL')})"
        elif c < 0.85:
            # LIKE
            e = self._gen_expr(depth - 1)
            pat = self._random_string_literal()
            not_kw = f"{self._kw('NOT')} " if self._maybe(0.4) else ""
            expr = f"({e} {not_kw}{self._kw('LIKE')} {pat})"
            if self._maybe(0.3):
                expr += f" {self._kw('ESCAPE')} '\\\\'"
            return expr
        else:
            # boolean combo
            a = self._gen_predicate(depth - 1)
            b = self._gen_predicate(depth - 1)
            op = self._kw(self.rnd.choice(["AND", "OR"]))
            if self._maybe(0.3):
                return f"({self._kw('NOT')} ({a} {op} {b}))"
            return f"({a} {op} {b})"

    def _gen_select_item(self, depth: int = 2) -> str:
        c = self.rnd.random()
        if c < 0.15:
            if self._maybe(0.5):
                return "*"
            else:
                return f"{self._random_identifier()}.*"
        expr = self._gen_expr(depth)
        if self._maybe(0.5):
            # alias
            alias = self._random_identifier()
            if self._maybe(0.5):
                return f"{expr} {self._kw('AS')} {alias}"
            else:
                return f"{expr} {alias}"
        return expr

    def _gen_table_ref(self, depth: int = 2) -> str:
        if depth > 0 and self._maybe(0.2):
            # subquery
            sub = self._gen_simple_select(depth - 1)
            alias = self._random_identifier()
            return f"({sub}) {self._kw('AS')} {alias}"
        # regular table with optional alias
        name = self._random_table_name()
        if self._maybe(0.6):
            alias = self._random_identifier()
            if self._maybe(0.5):
                return f"{name} {self._kw('AS')} {alias}"
            else:
                return f"{name} {alias}"
        return name

    def _gen_join_chain(self, depth: int = 2) -> str:
        # Start with a base table
        s = self._gen_table_ref(depth)
        n_joins = self.rnd.randint(0, 3)
        for _ in range(n_joins):
            jt = self.rnd.choice([
                "JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN",
                "LEFT OUTER JOIN", "RIGHT OUTER JOIN", "FULL OUTER JOIN", "CROSS JOIN", "NATURAL JOIN"
            ])
            jt_kw = " ".join(self._kw(x) for x in jt.split())
            right = self._gen_table_ref(depth - 1)
            if "NATURAL" in jt:
                s = f"{s} {jt_kw} {right}"
            elif "CROSS" in jt:
                s = f"{s} {jt_kw} {right}"
            else:
                if self._maybe(0.5):
                    cond = self._gen_predicate(depth - 1)
                    s = f"{s} {jt_kw} {right} {self._kw('ON')} {cond}"
                else:
                    col = self._random_column_name()
                    s = f"{s} {jt_kw} {right} {self._kw('USING')} ({col})"
        return s

    def _gen_simple_select(self, depth: int = 2) -> str:
        # A smaller select used in subqueries
        items = ", ".join(self._gen_select_item(max(1, depth)) for _ in range(self.rnd.randint(1, 3)))
        parts = [self._kw("SELECT")]
        if self._maybe(0.3):
            parts.append(self._kw("DISTINCT"))
        parts.append(items)
        if self._maybe(0.8):
            parts.append(self._kw("FROM"))
            parts.append(self._gen_table_ref(depth))
        if self._maybe(0.5):
            parts.append(self._kw("WHERE"))
            parts.append(self._gen_predicate(depth))
        if self._maybe(0.4):
            parts.append(self._kw("ORDER"))
            parts.append(self._kw("BY"))
            order_items = []
            for _ in range(self.rnd.randint(1, 2)):
                itm = self._gen_expr(depth)
                if self._maybe(0.5):
                    itm += " " + self._kw(self.rnd.choice(["ASC", "DESC"]))
                if self._maybe(0.3):
                    itm += " " + self._kw("NULLS") + " " + self._kw(self.rnd.choice(["FIRST", "LAST"]))
                order_items.append(itm)
            parts.append(", ".join(order_items))
        if self._maybe(0.3):
            parts.append(self._kw("LIMIT"))
            parts.append(self._random_number_literal())
            if self._maybe(0.5):
                parts.append(self._kw("OFFSET"))
                parts.append(self._random_number_literal())
        return self._join_tokens(parts)

    def _gen_select(self, depth: int = 3) -> str:
        parts = []
        # WITH clause
        if self._maybe(0.25):
            parts.append(self._kw("WITH"))
            if self._maybe(0.2):
                parts.append(self._kw("RECURSIVE"))
            ctes = []
            n_cte = self.rnd.randint(1, 3)
            for _ in range(n_cte):
                cte_name = self._random_identifier()
                if self._maybe(0.5):
                    # column list
                    cols = ", ".join(self._random_column_name() for __ in range(self.rnd.randint(1, 3)))
                    cte_head = f"{cte_name}({cols})"
                else:
                    cte_head = cte_name
                cte_body = self._gen_simple_select(depth - 1)
                ctes.append(f"{cte_head} {self._kw('AS')} ({cte_body})")
            parts.append(", ".join(ctes))
        # Main SELECT
        parts.append(self._kw("SELECT"))
        if self._maybe(0.3):
            parts.append(self._kw(self.rnd.choice(["DISTINCT", "ALL"])))
        n_items = self.rnd.randint(1, 6)
        items = ", ".join(self._gen_select_item(depth) for _ in range(n_items))
        parts.append(items)
        # FROM
        if self._maybe(0.85):
            parts.append(self._kw("FROM"))
            parts.append(self._gen_join_chain(depth))
        # WHERE
        if self._maybe(0.65):
            parts.append(self._kw("WHERE"))
            parts.append(self._gen_predicate(depth))
        # GROUP BY / HAVING
        if self._maybe(0.5):
            parts.append(self._kw("GROUP"))
            parts.append(self._kw("BY"))
            gb_items = ", ".join(self._gen_expr(depth - 1) for _ in range(self.rnd.randint(1, 3)))
            parts.append(gb_items)
            if self._maybe(0.5):
                parts.append(self._kw("HAVING"))
                parts.append(self._gen_predicate(depth - 1))
        # ORDER BY
        if self._maybe(0.5):
            parts.append(self._kw("ORDER"))
            parts.append(self._kw("BY"))
            order_items = []
            for _ in range(self.rnd.randint(1, 3)):
                itm = self._gen_expr(depth - 1)
                if self._maybe(0.7):
                    itm += " " + self._kw(self.rnd.choice(["ASC", "DESC"]))
                if self._maybe(0.4):
                    itm += " " + self._kw("NULLS") + " " + self._kw(self.rnd.choice(["FIRST", "LAST"]))
                order_items.append(itm)
            parts.append(", ".join(order_items))
        # LIMIT OFFSET
        if self._maybe(0.4):
            parts.append(self._kw("LIMIT"))
            parts.append(self._random_number_literal())
            if self._maybe(0.7):
                parts.append(self._kw("OFFSET"))
                parts.append(self._random_number_literal())
        stmt = self._join_tokens(parts)
        # Set operations
        if depth > 1 and self._maybe(0.3):
            op = self._kw(self.rnd.choice(["UNION", "UNION ALL", "INTERSECT", "EXCEPT"]))
            right = self._gen_simple_select(depth - 1)
            stmt = f"({stmt}) {op} ({right})"
        return stmt

    def _gen_insert(self) -> str:
        tbl = self._random_table_name()
        cols = []
        if self._maybe(0.7):
            n = self.rnd.randint(1, 6)
            cols = [self._random_column_name() for _ in range(n)]
            col_list = "(" + ", ".join(cols) + ")"
        else:
            col_list = ""
        parts = [self._kw("INSERT")]
        if self._maybe(0.2):
            parts.append(self._kw("OR"))
            parts.append(self._kw(self.rnd.choice(["ABORT", "REPLACE", "ROLLBACK", "FAIL", "IGNORE"])))
        parts.append(self._kw("INTO"))
        parts.append(tbl)
        if col_list:
            parts.append(col_list)
        if self._maybe(0.25):
            # INSERT ... SELECT
            parts.append(self._gen_simple_select(2))
        else:
            parts.append(self._kw("VALUES"))
            n_rows = self.rnd.randint(1, 5)
            tuples = []
            for _ in range(n_rows):
                if cols:
                    n_vals = len(cols)
                else:
                    n_vals = self.rnd.randint(1, 6)
                values = []
                for __ in range(n_vals):
                    if self._maybe(0.05):
                        values.append(self._kw("DEFAULT"))
                    else:
                        values.append(self._gen_expr(2))
                tuples.append("(" + ", ".join(values) + ")")
            parts.append(", ".join(tuples))
        return self._join_tokens(parts)

    def _gen_update(self) -> str:
        tbl = self._random_table_name()
        parts = [self._kw("UPDATE"), tbl, self._kw("SET")]
        n = self.rnd.randint(1, 5)
        assigns = []
        for _ in range(n):
            col = self._random_column_name()
            expr = self._gen_expr(2)
            assigns.append(f"{col} = {expr}")
        parts.append(", ".join(assigns))
        if self._maybe(0.7):
            parts.append(self._kw("WHERE"))
            parts.append(self._gen_predicate(2))
        return self._join_tokens(parts)

    def _gen_delete(self) -> str:
        tbl = self._random_table_name()
        parts = [self._kw("DELETE"), self._kw("FROM"), tbl]
        if self._maybe(0.8):
            parts.append(self._kw("WHERE"))
            parts.append(self._gen_predicate(2))
        return self._join_tokens(parts)

    def _gen_create_table(self) -> str:
        tbl = self._random_table_name()
        parts = [self._kw("CREATE"), self._kw("TABLE")]
        if self._maybe(0.4):
            parts.append(self._kw("IF"))
            parts.append(self._kw("NOT"))
            parts.append(self._kw("EXISTS"))
        parts.append(tbl)
        # Columns
        n_cols = self.rnd.randint(1, 8)
        col_defs = []
        pk_columns = []
        for _ in range(n_cols):
            cname = self._random_column_name()
            ctype = self._random_datatype()
            constraints = []
            if self._maybe(0.2):
                constraints.append(self._kw("PRIMARY") + " " + self._kw("KEY"))
                pk_columns.append(cname)
            if self._maybe(0.3):
                constraints.append(self._kw("NOT") + " " + self._kw("NULL"))
            if self._maybe(0.2):
                constraints.append(self._kw("UNIQUE"))
            if self._maybe(0.25):
                constraints.append(self._kw("DEFAULT") + " " + self._gen_expr(1))
            if self._maybe(0.15):
                constraints.append(self._kw("CHECK") + " (" + self._gen_predicate(1) + ")")
            col_defs.append(" ".join([cname, ctype] + constraints))
        # Table constraints
        t_constraints = []
        if self._maybe(0.3) and len(col_defs) >= 2:
            # Primary key multi-col
            cols = ", ".join(self._random_column_name() for _ in range(self.rnd.randint(1, min(3, n_cols))))
            t_constraints.append(self._kw("PRIMARY") + " " + self._kw("KEY") + f" ({cols})")
        if self._maybe(0.2) and len(col_defs) >= 2:
            cols = ", ".join(self._random_column_name() for _ in range(self.rnd.randint(1, min(3, n_cols))))
            t_constraints.append(self._kw("UNIQUE") + f" ({cols})")
        if self._maybe(0.2) and len(col_defs) >= 1:
            # Foreign key
            cols = ", ".join(self._random_column_name() for _ in range(self.rnd.randint(1, min(2, n_cols))))
            ref_tbl = self._random_table_name()
            ref_col = self._random_column_name()
            fk = f"{self._kw('FOREIGN')} {self._kw('KEY')} ({cols}) {self._kw('REFERENCES')} {ref_tbl} ({ref_col})"
            if self._maybe(0.5):
                fk += " " + self._kw("ON") + " " + self._kw("DELETE") + " " + self._kw(self.rnd.choice(["CASCADE","SET NULL","RESTRICT","NO ACTION"]))
            t_constraints.append(fk)
        all_defs = col_defs + t_constraints
        parts.append("(" + ", ".join(all_defs) + ")")
        if self._maybe(0.1):
            parts.append(self._kw("WITHOUT") + " " + self._kw("ROWID"))
        return self._join_tokens(parts)

    def _gen_create_index(self) -> str:
        idx_name = self._random_identifier()
        tbl = self._random_table_name()
        cols = ", ".join(self._random_column_name() for _ in range(self.rnd.randint(1, 4)))
        parts = [self._kw("CREATE")]
        if self._maybe(0.4):
            parts.append(self._kw("UNIQUE"))
        parts.append(self._kw("INDEX"))
        if self._maybe(0.4):
            parts.append(self._kw("IF"))
            parts.append(self._kw("NOT"))
            parts.append(self._kw("EXISTS"))
        parts.append(idx_name)
        parts.append(self._kw("ON"))
        parts.append(tbl)
        parts.append("(" + cols + ")")
        return self._join_tokens(parts)

    def _gen_drop(self) -> str:
        # Drop table or index or view
        kind = self.rnd.choice(["TABLE", "INDEX", "VIEW"])
        parts = [self._kw("DROP"), self._kw(kind)]
        if self._maybe(0.6):
            parts.append(self._kw("IF"))
            parts.append(self._kw("EXISTS"))
        parts.append(self._random_identifier())
        return self._join_tokens(parts)

    def _gen_alter_table(self) -> str:
        tbl = self._random_table_name()
        parts = [self._kw("ALTER"), self._kw("TABLE"), tbl]
        action = self.rnd.random()
        if action < 0.33:
            # ADD COLUMN
            parts.append(self._kw("ADD"))
            parts.append(self._kw("COLUMN"))
            cname = self._random_column_name()
            ctype = self._random_datatype()
            extra = []
            if self._maybe(0.3):
                extra.append(self._kw("DEFAULT") + " " + self._gen_expr(1))
            if self._maybe(0.3):
                extra.append(self._kw("NOT") + " " + self._kw("NULL"))
            parts.append(" ".join([cname, ctype] + extra))
        elif action < 0.66:
            # RENAME COLUMN
            parts.append(self._kw("RENAME"))
            parts.append(self._kw("COLUMN"))
            old = self._random_column_name()
            parts.append(old)
            parts.append(self._kw("TO"))
            parts.append(self._random_column_name())
        else:
            # RENAME TABLE
            parts.append(self._kw("RENAME"))
            parts.append(self._kw("TO"))
            parts.append(self._random_table_name())
        return self._join_tokens(parts)

    def _gen_statement(self) -> str:
        p = self.rnd.random()
        if p < 0.50:
            return self._gen_select(3)
        elif p < 0.62:
            return self._gen_insert()
        elif p < 0.74:
            return self._gen_update()
        elif p < 0.84:
            return self._gen_delete()
        elif p < 0.92:
            if self._maybe(0.6):
                return self._gen_create_table()
            else:
                return self._gen_create_index()
        elif p < 0.97:
            return self._gen_alter_table()
        else:
            return self._gen_drop()

    def _mutate(self, sql: str) -> str:
        # Apply random mutation strategies to increase diversity
        strategies = []
        strategies.append(self._mut_replace_keywords_case)
        strategies.append(self._mut_insert_comments)
        strategies.append(self._mut_replace_operators)
        strategies.append(self._mut_shuffle_whitespace)
        strategies.append(self._mut_duplicate_substring)
        strategies.append(self._mut_truncate_random)
        strategies.append(self._mut_insert_random_token)
        # Choose 1-3 strategies
        n = self.rnd.randint(1, 3)
        for strat in self.rnd.sample(strategies, n):
            try:
                sql = strat(sql)
            except Exception:
                pass
        return sql

    def _mut_replace_keywords_case(self, s: str) -> str:
        # Randomly change case of known keywords
        # Replace occurrences by random-case variant
        words = set(["SELECT","FROM","WHERE","GROUP","BY","HAVING","ORDER","LIMIT","OFFSET","INSERT","INTO","VALUES","UPDATE",
                     "SET","DELETE","CREATE","TABLE","DROP","ALTER","ADD","COLUMN","RENAME","TO","AS","DISTINCT","ALL","JOIN",
                     "INNER","LEFT","RIGHT","FULL","OUTER","CROSS","NATURAL","ON","USING","NULL","NOT","AND","OR","IS","LIKE",
                     "IN","BETWEEN","EXISTS","UNION","INTERSECT","EXCEPT","PRIMARY","KEY","FOREIGN","REFERENCES","UNIQUE",
                     "CHECK","DEFAULT","TRUE","FALSE","WITH","RECURSIVE","CASE","WHEN","THEN","ELSE","END","ASC","DESC","NULLS",
                     "FIRST","LAST","INDEX","IF","EXISTS","VIEW","CAST"])
        def repl(m):
            w = m.group(0)
            return self._kw(w)
        pattern = r"\b(" + "|".join(sorted(words)) + r")\b"
        try:
            return re.sub(pattern, repl, s, flags=re.IGNORECASE)
        except Exception:
            return s

    def _mut_insert_comments(self, s: str) -> str:
        if len(s) < 4:
            return s
        # Insert 1-3 comments at random positions
        n = self.rnd.randint(1, 3)
        for _ in range(n):
            pos = self.rnd.randint(0, len(s))
            if self._maybe(0.5):
                comment = "/*" + self._random_identifier_base() + "*/"
            else:
                comment = "--" + self._random_identifier_base() + "\n"
            s = s[:pos] + " " + comment + " " + s[pos:]
        return s

    def _mut_replace_operators(self, s: str) -> str:
        # Replace some operators with others
        ops = ["=", "!=", "<>", "<=", ">=", "<", ">", "+", "-", "*", "/", "%", "||"]
        new_ops = ["=", "!=", "<>", "<=", ">=", "<", ">", "+", "-", "*", "/", "%", "||"]
        for _ in range(self.rnd.randint(1, 3)):
            old = self.rnd.choice(ops)
            new = self.rnd.choice(new_ops)
            if old != new and old in s:
                s = s.replace(old, new)
        return s

    def _mut_shuffle_whitespace(self, s: str) -> str:
        # Replace spaces with random whitespace including newlines
        if self._maybe(0.5):
            return re.sub(r"[ \t]+", lambda m: " " * self.rnd.randint(1, 3), s)
        else:
            return re.sub(r"[ \t]+", lambda m: self.rnd.choice([" ", "  ", "\n", "\n "]), s)

    def _mut_duplicate_substring(self, s: str) -> str:
        if len(s) < 8:
            return s
        i = self.rnd.randint(0, len(s) - 4)
        j = self.rnd.randint(i + 1, min(len(s), i + 20))
        frag = s[i:j]
        insert_pos = self.rnd.randint(0, len(s))
        return s[:insert_pos] + frag + s[insert_pos:]

    def _mut_truncate_random(self, s: str) -> str:
        if len(s) < 10 or not self._maybe(0.2):
            return s
        pos = self.rnd.randint(5, len(s))
        return s[:pos]

    def _mut_insert_random_token(self, s: str) -> str:
        tok_choices = list(self.keyword_set) + list(self.operator_set) + self.punctuations
        tok = self._kw(self.rnd.choice(tok_choices)) if self._maybe(0.6) else self.rnd.choice(tok_choices)
        pos = self.rnd.randint(0, len(s))
        return s[:pos] + " " + str(tok) + " " + s[pos:]

    def _batch_generate(self, target_count: int) -> List[str]:
        stmts: List[str] = []
        # Generate fresh statements
        for _ in range(target_count):
            s = self._gen_statement()
            if self._maybe(0.3):
                s = self._mutate(s)
            stmts.append(s)
        return stmts

    def _corpus_mutations(self, base: List[str], target_count: int) -> List[str]:
        out: List[str] = []
        if not base:
            return out
        for _ in range(target_count):
            s = self.rnd.choice(base)
            # Sometimes add semicolons or combine statements
            if self._maybe(0.2):
                s2 = self.rnd.choice(base)
                combined = s + ("" if s.endswith(";") else ";") + "\n" + s2
                s = combined
            s = self._mutate(s)
            out.append(s)
        return out

    def _long_complex_statements(self, n: int) -> List[str]:
        out = []
        for _ in range(n):
            # Build a complex statement: WITH + JOINs + set ops
            s = self._gen_select(4)
            # wrap with comments and mutations
            if self._maybe(0.7):
                s = self._mut_insert_comments(s)
            if self._maybe(0.5):
                s = self._mut_replace_keywords_case(s)
            if self._maybe(0.5):
                s = self._mut_replace_operators(s)
            out.append(s)
        return out

    def fuzz(self, parse_sql):
        if self.start_time is None:
            self.start_time = time.time()
        # Determine dynamic batch sizes
        # Heuristic: big first batch, then smaller ones; aim for fewer calls
        if self.call_count == 0:
            gen_count = 2000
            mut_count = 1000
            long_count = 200
        elif self.call_count == 1:
            gen_count = 1200
            mut_count = 800
            long_count = 120
        else:
            gen_count = 800
            mut_count = 600
            long_count = 80
        # Build the batch
        batch: List[str] = []
        # Mutations of seed corpus and previously generated
        batch.extend(self._corpus_mutations(self.corpus, mut_count))
        # Freshly generated statements
        batch.extend(self._batch_generate(gen_count))
        # Some long and complex statements
        batch.extend(self._long_complex_statements(long_count))
        # Some pure seeds to ensure deterministic coverage
        if self.call_count == 0:
            batch.extend(self.corpus[:200])

        # Shuffle batch for variety
        self.rnd.shuffle(batch)
        # Update corpus with a sample of the generated ones
        self.corpus.extend(batch[:500])

        # Execute
        try:
            parse_sql(batch)
        except Exception:
            # The evaluator's parse_sql is supposed to catch exceptions, but guard anyway
            pass

        self.call_count += 1

        # Stop early to maximize efficiency bonus or if time nearly up
        elapsed = time.time() - self.start_time
        if self.call_count >= self.max_calls or elapsed > self.time_budget:
            return False
        return True
'''
        return {"code": code}