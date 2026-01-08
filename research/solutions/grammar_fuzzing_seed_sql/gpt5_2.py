import os
import sys
import importlib
import re
import random
from typing import List, Optional, Set, Tuple

class Solution:
    def _import_parse_sql(self, resources_path: str):
        parse_sql = None
        try:
            if resources_path and resources_path not in sys.path:
                sys.path.insert(0, resources_path)
            try:
                mod = importlib.import_module("sql_engine.parser")
                parse_sql = getattr(mod, "parse_sql", None)
            except Exception:
                pass
            if parse_sql is None:
                mod = importlib.import_module("sql_engine")
                parse_sql = getattr(mod, "parse_sql", None)
        except Exception:
            parse_sql = None
        return parse_sql

    def _get_keywords_from_module(self) -> Set[str]:
        kw = set()
        try:
            tok = importlib.import_module("sql_engine.tokenizer")
            if hasattr(tok, "KEYWORDS"):
                K = getattr(tok, "KEYWORDS")
                if isinstance(K, dict):
                    for k in K.keys():
                        if isinstance(k, str):
                            kw.add(k.upper())
                elif isinstance(K, (set, list, tuple)):
                    for x in K:
                        if isinstance(x, str):
                            kw.add(x.upper())
            # Sometimes tokenizer keeps regex for operators/identifiers as well
            for name in dir(tok):
                if name.isupper():
                    val = getattr(tok, name)
                    if isinstance(val, (list, tuple, set)):
                        for x in val:
                            if isinstance(x, str) and x.isalpha():
                                kw.add(x.upper())
        except Exception:
            pass
        return kw

    def _get_keywords_from_file(self, resources_path: str) -> Set[str]:
        kw = set()
        try:
            tpath = os.path.join(resources_path, "sql_engine", "tokenizer.py")
            if os.path.exists(tpath):
                text = open(tpath, "r", encoding="utf-8").read()
                # Find uppercase words in quotes likely representing keywords
                for m in re.finditer(r"['\"]([A-Z_][A-Z0-9_]*)['\"]", text):
                    w = m.group(1)
                    if len(w) >= 2 and w.upper() == w:
                        kw.add(w)
        except Exception:
            pass
        return kw

    def _get_keywords(self, resources_path: str) -> Set[str]:
        kw = set()
        try:
            kw |= self._get_keywords_from_module()
        except Exception:
            pass
        try:
            kw |= self._get_keywords_from_file(resources_path)
        except Exception:
            pass
        # Add common SQL keywords fallback
        if not kw:
            kw = {
                "SELECT", "FROM", "WHERE", "GROUP", "BY", "HAVING", "ORDER", "LIMIT", "OFFSET",
                "INSERT", "INTO", "VALUES", "UPDATE", "SET", "DELETE",
                "CREATE", "TABLE", "IF", "NOT", "EXISTS", "DROP", "ALTER", "ADD", "COLUMN",
                "PRIMARY", "KEY", "UNIQUE", "DEFAULT", "NULL", "NOT", "REFERENCES",
                "JOIN", "LEFT", "RIGHT", "FULL", "OUTER", "INNER", "CROSS", "ON", "USING",
                "AND", "OR", "IS", "IN", "BETWEEN", "LIKE", "AS", "DISTINCT", "ALL",
                "UNION", "INTERSECT", "EXCEPT", "CASE", "WHEN", "THEN", "ELSE", "END",
                "EXISTS", "VIEW", "INDEX", "RENAME", "TO", "TRUE", "FALSE"
            }
        return kw

    def _try_parse_variants(self, parse_sql, stmt: str) -> Optional[str]:
        variants = []
        # Basic variants: as-is, with semicolon, upper, lower
        base = stmt.strip()
        if not base:
            return None
        variants.append(base)
        if not base.endswith(";"):
            variants.append(base + ";")
        upper = base.upper()
        variants.append(upper)
        if not upper.endswith(";"):
            variants.append(upper + ";")
        lower = base.lower()
        variants.append(lower)
        if not lower.endswith(";"):
            variants.append(lower + ";")
        # With some spacing/newline variants
        spaced = re.sub(r"\s+", " ", base)
        variants.append(spaced)
        if not spaced.endswith(";"):
            variants.append(spaced + ";")
        # Try injecting a simple comment after first keyword (if any)
        parts = base.split(None, 1)
        if len(parts) == 2:
            commented = parts[0] + " /*c*/ " + parts[1]
            variants.append(commented)
            if not commented.endswith(";"):
                variants.append(commented + ";")
        # Deduplicate while preserving order
        seen = set()
        unique_variants = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                unique_variants.append(v)
        for v in unique_variants:
            try:
                parse_sql(v)
                return v
            except Exception:
                continue
        return None

    def _maybe_add(self, parse_sql, out: List[str], seen: Set[str], stmt: str) -> bool:
        try:
            good = self._try_parse_variants(parse_sql, stmt)
            if good and good not in seen:
                seen.add(good)
                out.append(good)
                return True
        except Exception:
            pass
        return False

    def _candidate_statements_curated(self) -> List[str]:
        # A broad set of SQL statements designed to cover common parser branches
        stmts = []

        # Comments and whitespace variants
        stmts.append("/* block comment */ SELECT 1 -- inline comment")
        stmts.append("-- only comment line\nSELECT 'hello world'")
        stmts.append("SELECT /* mid */ 2 + 3 * 4 AS result")

        # Simple selects
        stmts.append("SELECT 1")
        stmts.append("SELECT -1, 3.1415, 1e10")
        stmts.append("SELECT NULL, TRUE, FALSE")
        stmts.append("SELECT 'it''s ok' AS txt")

        # Expressions and aliases
        stmts.append("SELECT (1 + 2) * (3 - 4) AS calc, 5 / 2 AS half")
        stmts.append("SELECT CASE WHEN 1=1 THEN 42 ELSE 0 END AS c")
        stmts.append("SELECT COALESCE(NULL, 1, 2) AS c")
        stmts.append("SELECT NULLIF(1, 1) AS n, NULLIF(1, 2) AS m")

        # Identifiers quoted and backticks
        stmts.append('SELECT "col", "table"."col2" FROM "table"')
        stmts.append("SELECT `col`, `t`.`c` FROM `t`")

        # FROM, WHERE conditions
        stmts.append("SELECT a FROM t")
        stmts.append("SELECT * FROM t WHERE a > 10 AND b < 5 OR c = 'x'")
        stmts.append("SELECT a FROM t WHERE NOT a IN (1,2,3)")
        stmts.append("SELECT a FROM t WHERE a BETWEEN 1 AND 10")
        stmts.append("SELECT a FROM t WHERE a IS NOT NULL")
        stmts.append("SELECT a FROM t WHERE name LIKE 'A%'")
        stmts.append(r"SELECT a FROM t WHERE name LIKE 'A\_%' ESCAPE '\'")

        # ORDER BY, LIMIT, OFFSET
        stmts.append("SELECT a FROM t ORDER BY a DESC, b ASC")
        stmts.append("SELECT a FROM t ORDER BY 1")
        stmts.append("SELECT a FROM t LIMIT 10")
        stmts.append("SELECT a FROM t LIMIT 10 OFFSET 5")
        stmts.append("SELECT a FROM t OFFSET 5")

        # DISTINCT and ALL
        stmts.append("SELECT DISTINCT a, b FROM t")
        stmts.append("SELECT ALL a FROM t")

        # GROUP BY / HAVING
        stmts.append("SELECT a, COUNT(*) AS cnt FROM t GROUP BY a HAVING COUNT(*) > 1")
        stmts.append("SELECT a FROM t GROUP BY 1")

        # Joins
        stmts.append("SELECT * FROM t1 JOIN t2 ON t1.id = t2.id")
        stmts.append("SELECT * FROM t1 INNER JOIN t2 ON t1.id = t2.id")
        stmts.append("SELECT * FROM t1 LEFT JOIN t2 USING (id)")
        stmts.append("SELECT * FROM t1 RIGHT JOIN t2 ON t1.id = t2.id")
        stmts.append("SELECT * FROM t1 FULL OUTER JOIN t2 ON t1.id = t2.id")
        stmts.append("SELECT * FROM t1 CROSS JOIN t2")
        stmts.append("SELECT * FROM t1, t2")
        stmts.append("SELECT * FROM t1 LEFT OUTER JOIN t2 ON t1.k = t2.k")

        # Subqueries
        stmts.append("SELECT * FROM (SELECT 1 AS x) sub")
        stmts.append("SELECT (SELECT 1) AS s")
        stmts.append("SELECT a FROM t WHERE a IN (SELECT a FROM t2)")
        stmts.append("SELECT EXISTS (SELECT 1 FROM t2) AS e")
        stmts.append("SELECT * FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.id = t1.id)")

        # Set operations
        stmts.append("SELECT 1 UNION SELECT 2")
        stmts.append("SELECT 1 UNION ALL SELECT 2")
        stmts.append("SELECT 1 INTERSECT SELECT 1")
        stmts.append("SELECT 1 EXCEPT SELECT 1")

        # CTEs
        stmts.append("WITH cte AS (SELECT 1 AS x) SELECT x FROM cte")
        stmts.append("WITH a AS (SELECT 1 AS x), b AS (SELECT 2 AS y) SELECT x, y FROM a, b")
        stmts.append("WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r WHERE n < 3) SELECT * FROM r")

        # Functions and aggregates
        stmts.append("SELECT COUNT(*), SUM(val), AVG(val), MAX(val), MIN(val) FROM t")
        stmts.append("SELECT ABS(-5), ROUND(3.1415), LENGTH('abc')")

        # Window functions (if supported)
        stmts.append("SELECT a, ROW_NUMBER() OVER (PARTITION BY b ORDER BY c) AS rn FROM t")
        stmts.append("SELECT SUM(a) OVER (ORDER BY b ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t")

        # DML & DDL
        stmts.append("CREATE TABLE t (id INT PRIMARY KEY, name VARCHAR(100) NOT NULL, price DECIMAL(10,2) DEFAULT 0.0, active BOOLEAN, created_at TIMESTAMP)")
        stmts.append("CREATE TABLE IF NOT EXISTS t2 (id INTEGER, ref_id INT REFERENCES t(id), UNIQUE(id, ref_id))")
        stmts.append("ALTER TABLE t ADD COLUMN quantity INT DEFAULT 0")
        stmts.append("ALTER TABLE t DROP COLUMN active")
        stmts.append("ALTER TABLE t RENAME TO t_renamed")
        stmts.append("ALTER TABLE t_renamed RENAME COLUMN name TO full_name")
        stmts.append("CREATE INDEX idx_t_name ON t_renamed (full_name, price DESC)")
        stmts.append("DROP INDEX idx_t_name")
        stmts.append("CREATE VIEW v AS SELECT a, b FROM t")
        stmts.append("DROP VIEW v")
        stmts.append("INSERT INTO t_renamed (id, full_name, price, quantity) VALUES (1, 'a', 1.23, 10), (2, 'b', 0.0, NULL)")
        stmts.append("INSERT INTO t_renamed VALUES (3, 'c', 3.14, 2)")
        stmts.append("INSERT INTO t (id, name) SELECT id, full_name FROM t_renamed WHERE price > 0")
        stmts.append("UPDATE t_renamed SET price = price * 1.1, full_name = 'updated' WHERE id <> 0")
        stmts.append("DELETE FROM t_renamed WHERE price IS NULL OR price <= 0")
        stmts.append("TRUNCATE TABLE t_renamed")
        stmts.append("DROP TABLE t_renamed")
        stmts.append("DROP TABLE IF EXISTS t")

        # Casting and collate-like
        stmts.append("SELECT CAST(1 AS INTEGER), CAST('3.14' AS FLOAT)")
        stmts.append("SELECT a || b AS concat FROM t")

        # Boolean expressions / IS TRUE/FALSE
        stmts.append("SELECT (1=1) IS TRUE AS t, (1=0) IS FALSE AS f")

        # Parentheses and precedence
        stmts.append("SELECT ((a + b) * (c - d)) / (e + 1) FROM t")

        return stmts

    def _generate_select_variations(self) -> List[str]:
        # Programmatically combine features to produce a diverse set of SELECT statements
        cols_options = [
            "*",
            "1 AS one",
            "a",
            "a, b",
            "t1.a, t2.b",
            "COUNT(*) AS cnt",
            "SUM(val) AS total, AVG(val) AS avg",
            "CASE WHEN a>1 THEN 'x' ELSE 'y' END AS c"
        ]
        from_options = [
            "",
            "FROM t",
            "FROM t1",
            "FROM t1 AS x",
            "FROM t1 JOIN t2 ON t1.id = t2.id",
            "FROM t1 LEFT JOIN t2 USING (id)",
            "FROM (SELECT 1 AS x) s",
            "FROM t1, t2"
        ]
        where_options = [
            "",
            "WHERE a > 1",
            "WHERE a BETWEEN 1 AND 10",
            "WHERE name LIKE 'A%'",
            "WHERE a IN (1,2,3)",
            "WHERE a IN (SELECT a FROM t2)",
            "WHERE (a>1) AND NOT (b<2 OR c=3)",
            "WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.id = t1.id)"
        ]
        group_options = [
            "",
            "GROUP BY a",
            "GROUP BY 1, 2"
        ]
        having_options = [
            "",
            "HAVING COUNT(*) > 1"
        ]
        order_options = [
            "",
            "ORDER BY a DESC, b",
            "ORDER BY 1 DESC"
        ]
        limit_options = [
            "",
            "LIMIT 10",
            "LIMIT 10 OFFSET 5",
            "OFFSET 3"
        ]
        stmts = []
        # Create a curated small cross-product, but respecting that HAVING needs GROUP BY
        combos = [
            (0, 1, 0, 0, 0, 0),
            (1, 1, 1, 0, 0, 1),
            (2, 2, 2, 1, 1, 2),
            (3, 3, 3, 2, 1, 3),
            (4, 4, 4, 1, 0, 1),
            (5, 5, 5, 1, 0, 2),
            (6, 6, 6, 0, 0, 0),
            (7, 7, 0, 2, 0, 1),
        ]
        # Expand combos to generate statements
        for ci, fi, wi, gi, hi, oi in combos:
            cols = cols_options[ci % len(cols_options)]
            frm = from_options[fi % len(from_options)]
            where = where_options[wi % len(where_options)]
            group = group_options[gi % len(group_options)]
            having = having_options[hi % len(having_options)]
            order = order_options[oi % len(order_options)]
            # Only include HAVING if group is not empty
            parts = [f"SELECT {cols}"]
            if frm:
                parts.append(frm)
            if where:
                parts.append(where)
            if group:
                parts.append(group)
            if having and group:
                parts.append(having)
            if order:
                parts.append(order)
            # Choose a limit option to append for variation
            limit = random.choice(limit_options)
            if limit:
                parts.append(limit)
            stmt = " ".join(parts)
            stmts.append(stmt)

        # Some set operations combos
        stmts.append("SELECT a FROM t UNION SELECT b FROM t2")
        stmts.append("SELECT a FROM t UNION ALL SELECT a FROM t")
        stmts.append("SELECT a FROM t INTERSECT SELECT a FROM t2")
        stmts.append("SELECT a FROM t EXCEPT SELECT a FROM t2")

        # Subquery scalar select in target list
        stmts.append("SELECT (SELECT MAX(a) FROM t2) AS maxa FROM t")

        # Mixed parentheses and precedence
        stmts.append("SELECT a + b * c AS x, (a + b) * c AS y FROM t ORDER BY x, y DESC LIMIT 5 OFFSET 2")

        return stmts

    def _generate_ddl_dml_variations(self) -> List[str]:
        stmts = []
        # Various CREATE TABLE syntaxes
        stmts.append("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INT, email VARCHAR(255) UNIQUE, active BOOLEAN DEFAULT TRUE)")
        stmts.append("CREATE TABLE IF NOT EXISTS orders (id INT, user_id INT REFERENCES users(id), total DECIMAL(10,2), created_at TIMESTAMP)")
        stmts.append("CREATE TEMP TABLE tmp (k INT, v TEXT)")
        stmts.append("ALTER TABLE users ADD COLUMN last_login DATE")
        stmts.append("ALTER TABLE users DROP COLUMN active")
        stmts.append("ALTER TABLE users RENAME COLUMN email TO email_addr")
        stmts.append("ALTER TABLE users RENAME TO people")
        stmts.append("CREATE INDEX idx_people_name ON people (name)")
        stmts.append("CREATE UNIQUE INDEX idx_orders_user ON orders (user_id)")
        stmts.append("DROP INDEX idx_people_name")

        # INSERT variations
        stmts.append("INSERT INTO people (id, name, age, email_addr, last_login) VALUES (1, 'Alice', 30, 'a@example.com', DATE '2020-01-01')")
        stmts.append("INSERT INTO people VALUES (2, 'Bob', 25, 'b@example.com', NULL)")
        stmts.append("INSERT INTO orders (id, user_id, total, created_at) VALUES (10, 1, 123.45, TIMESTAMP '2020-01-02 12:30:00'), (11, 2, 0.00, TIMESTAMP '2020-01-03 09:00:00')")
        stmts.append("INSERT INTO tmp SELECT id, name FROM people")

        # UPDATE variations
        stmts.append("UPDATE people SET age = age + 1 WHERE id = 1")
        stmts.append("UPDATE people SET name = 'Robert', email_addr = 'bob@example.com'")
        stmts.append("UPDATE orders SET total = total * 1.05 WHERE total > 100")

        # DELETE variations
        stmts.append("DELETE FROM tmp WHERE k IN (SELECT id FROM people WHERE age >= 30)")
        stmts.append("DELETE FROM tmp")

        # VIEW
        stmts.append("CREATE VIEW top_customers AS SELECT user_id, SUM(total) AS total_spent FROM orders GROUP BY user_id HAVING SUM(total) > 1000")
        stmts.append("DROP VIEW top_customers")

        # TRUNCATE and DROP TABLE
        stmts.append("TRUNCATE TABLE tmp")
        stmts.append("DROP TABLE IF EXISTS tmp")
        stmts.append("DROP TABLE people")
        stmts.append("DROP TABLE IF EXISTS orders")

        return stmts

    def _ensure_minimum_valid(self, parse_sql, out: List[str], seen: Set[str]) -> None:
        # Ensure at least a minimal set of simple statements are available
        basics = [
            "SELECT 1",
            "SELECT 'a'",
            "SELECT 1 + 2",
            "SELECT * FROM t",
            "SELECT a FROM t WHERE a = 1",
            "SELECT a FROM t ORDER BY a",
            "SELECT a FROM t LIMIT 1",
        ]
        for s in basics:
            if len(out) >= 8:
                break
            self._maybe_add(parse_sql, out, seen, s)

    def solve(self, resources_path: str) -> List[str]:
        parse_sql = self._import_parse_sql(resources_path)
        keywords = self._get_keywords(resources_path)

        curated = self._candidate_statements_curated()
        select_variants = self._generate_select_variations()
        ddl_dml_variants = self._generate_ddl_dml_variations()

        # Optionally filter candidate families based on keyword hints to reduce invalid attempts
        families: List[Tuple[str, List[str]]] = []
        families.append(("curated", curated))

        # Heuristic gating
        if {"SELECT", "FROM"}.issubset(keywords):
            families.append(("select_variants", select_variants))
        if {"CREATE", "TABLE"}.issubset(keywords) or {"INSERT", "UPDATE", "DELETE"}.intersection(keywords):
            families.append(("ddl_dml", ddl_dml_variants))

        out: List[str] = []
        seen: Set[str] = set()

        if parse_sql is None:
            # Fallback: return a conservative set without validation
            conservative = [
                "SELECT 1",
                "SELECT 'hello'",
                "SELECT 1 + 2 AS sum",
                "SELECT a FROM t",
                "SELECT a, COUNT(*) FROM t GROUP BY a",
                "SELECT a FROM t WHERE a BETWEEN 1 AND 10",
                "SELECT a FROM t WHERE a IN (1,2,3)",
                "SELECT a FROM t ORDER BY a DESC LIMIT 5 OFFSET 1",
                "SELECT * FROM t1 JOIN t2 ON t1.id = t2.id",
                "SELECT * FROM (SELECT 1 AS x) s",
                "SELECT 1 UNION SELECT 2",
            ]
            # Add some DML/DDL that many parsers accept
            conservative += [
                "CREATE TABLE t (id INT, name TEXT)",
                "INSERT INTO t (id, name) VALUES (1, 'a'), (2, 'b')",
                "UPDATE t SET name = 'z' WHERE id = 2",
                "DELETE FROM t WHERE id = 1",
                "DROP TABLE t",
            ]
            return conservative

        # Try curated first for broad coverage
        for _, fam in families:
            for stmt in fam:
                self._maybe_add(parse_sql, out, seen, stmt)

        # Ensure minimum valid statements exist, in case many advanced ones fail
        self._ensure_minimum_valid(parse_sql, out, seen)

        # Target an efficient number of statements: cap to a reasonable max, but try to keep diversity
        # If we have too few valid statements (< 15), attempt a few more small syntactic variations
        if len(out) < 15:
            extra_small = [
                "SELECT 2",
                "SELECT 0",
                "SELECT -42",
                "SELECT 'X'",
                "SELECT NULL",
                "SELECT TRUE",
                "SELECT FALSE",
                "SELECT 1, 2, 3",
                "SELECT 1 AS a, 2 AS b, 3 AS c",
                "SELECT a FROM t WHERE a <> 0",
                "SELECT a FROM t WHERE a != 0",
                "SELECT a FROM t WHERE a IS NULL",
                "SELECT a FROM t WHERE a IS NOT NULL",
                "SELECT a FROM t LIMIT 2",
                "SELECT a FROM t OFFSET 1",
            ]
            for s in extra_small:
                self._maybe_add(parse_sql, out, seen, s)

        # Final trimming: prefer at most ~60 statements for efficiency while preserving diversity
        if len(out) > 60:
            # Keep: ensure we retain at least one from each major family type via matching patterns
            patterns = [
                r"\bCREATE\b", r"\bALTER\b", r"\bDROP\b",
                r"\bINSERT\b", r"\bUPDATE\b", r"\bDELETE\b",
                r"\bJOIN\b", r"\bUNION\b|\bINTERSECT\b|\bEXCEPT\b",
                r"\bGROUP BY\b", r"\bHAVING\b", r"\bORDER BY\b",
                r"\bLIMIT\b|\bOFFSET\b", r"\bWITH\b", r"\bEXISTS\b", r"\bCASE\b",
            ]
            retained = []
            seen_pat = set()
            # First pass: ensure coverage for each pattern
            for pat in patterns:
                rx = re.compile(pat)
                for s in out:
                    if rx.search(s) and s not in retained:
                        retained.append(s)
                        break
            # Second pass: fill up to 60 with remaining
            for s in out:
                if len(retained) >= 60:
                    break
                if s not in retained:
                    retained.append(s)
            out = retained

        return out