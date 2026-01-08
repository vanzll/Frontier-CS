import os
import sys
import re
import importlib
from typing import Callable, List, Optional, Tuple, Set


class Solution:
    def solve(self, resources_path: str) -> list[str]:
        parse_func = self._load_parse_function(resources_path)
        candidates = self._generate_candidates()
        valid = self._filter_valid(parse_func, candidates)
        if not valid:
            # Fallback: return a compact, broadly compatible set
            return self._fallback_statements()
        final = self._select_diverse_subset(valid, max_count=50)
        if not final:
            return self._fallback_statements()
        return final

    def _load_parse_function(self, resources_path: str) -> Optional[Callable[[str], object]]:
        try:
            if resources_path and resources_path not in sys.path:
                sys.path.insert(0, resources_path)
            # Try direct import of the package and parser
            parse_func = None
            try:
                parser_mod = importlib.import_module("sql_engine.parser")
            except Exception:
                parser_mod = None
            if parser_mod is not None:
                for name in [
                    "parse_sql",
                    "parse",
                    "parse_statement",
                    "parse_query",
                    "parse_sql_statement",
                    "parse_sqls",
                ]:
                    if hasattr(parser_mod, name):
                        cand = getattr(parser_mod, name)
                        if callable(cand):
                            parse_func = cand
                            break
                if parse_func is None and hasattr(parser_mod, "Parser"):
                    Parser = getattr(parser_mod, "Parser")
                    try:
                        parser_instance = Parser()
                        for name in ["parse_sql", "parse", "parse_statement", "parse_query"]:
                            if hasattr(parser_instance, name):
                                meth = getattr(parser_instance, name)
                                if callable(meth):
                                    def wrapper(s: str, _meth=meth):
                                        return _meth(s)
                                    parse_func = wrapper
                                    break
                    except Exception:
                        pass
            if parse_func is None:
                # Try top-level package
                try:
                    pkg = importlib.import_module("sql_engine")
                    for name in [
                        "parse_sql",
                        "parse",
                        "parse_statement",
                        "parse_query",
                        "parse_sql_statement",
                        "parse_sqls",
                    ]:
                        if hasattr(pkg, name):
                            cand = getattr(pkg, name)
                            if callable(cand):
                                parse_func = cand
                                break
                except Exception:
                    pass
            return parse_func
        except Exception:
            return None

    def _generate_candidates(self) -> List[str]:
        stmts: List[str] = []

        # Core literals and expressions
        stmts += [
            "SELECT 1;",
            "SELECT 1 AS one, 2.5 AS two, 3e2 AS three;",
            "SELECT 'hello' AS s, 'It''s' AS escaped;",
            "/*block*/ SELECT 1; -- inline",
            "-- only comment then select\nSELECT 1;",
            "SELECT NULL;",
        ]

        # Basic SELECT with FROM
        stmts += [
            "SELECT * FROM t;",
            "SELECT t.* FROM t;",
            "SELECT a, b FROM t;",
            "SELECT a FROM t WHERE a = 1;",
            "SELECT a FROM t WHERE a <> 2;",
            "SELECT a FROM t WHERE a != 3;",
            "SELECT a FROM t WHERE a > 0 AND b < 10 OR NOT c;",
            "SELECT a FROM t WHERE a BETWEEN 1 AND 5;",
            "SELECT a FROM t WHERE a NOT BETWEEN 1 AND 5;",
            "SELECT a FROM t WHERE a IN (1,2,3);",
            "SELECT a FROM t WHERE a NOT IN (1,2,3);",
            "SELECT a FROM t WHERE a LIKE 'a%';",
            "SELECT a FROM t WHERE a IS NULL;",
            "SELECT a FROM t WHERE a IS NOT NULL;",
            "SELECT a, COUNT(*) AS c FROM t GROUP BY a;",
            "SELECT a, COUNT(*) AS c FROM t GROUP BY a HAVING COUNT(*) > 1;",
            "SELECT DISTINCT a FROM t;",
            "SELECT DISTINCT a FROM t ORDER BY a DESC;",
            "SELECT a, b FROM t ORDER BY a DESC, b ASC;",
            "SELECT a FROM t ORDER BY 1;",
            "SELECT a FROM t ORDER BY a NULLS LAST;",
            "SELECT a FROM t LIMIT 10;",
            "SELECT a FROM t LIMIT 10 OFFSET 5;",
        ]

        # Arithmetic and boolean ops
        stmts += [
            "SELECT (1 + 2) * 3 AS v;",
            "SELECT a + b - c FROM t;",
            "SELECT a / (b + 1) FROM t;",
            "SELECT a % 2 FROM t;",
            "SELECT -a, +b FROM t;",
            "SELECT a AND b OR NOT c FROM t;",
            "SELECT 'a' || 'b' AS cat;",
        ]

        # Functions, aggregates, case
        stmts += [
            "SELECT ABS(-1), ROUND(1.23, 1), LENGTH('abc');",
            "SELECT COALESCE(NULL, 1), NULLIF(1, 1);",
            "SELECT SUM(a), AVG(b), MIN(c), MAX(c), COUNT(*) FROM t;",
            "SELECT COUNT(DISTINCT a) FROM t;",
            "SELECT CASE WHEN a > 10 THEN 'big' WHEN a = 0 THEN 'zero' ELSE 'small' END FROM t;",
        ]

        # Joins
        joins = [
            "SELECT t1.a, t2.b FROM t1 JOIN t2 ON t1.id = t2.id;",
            "SELECT * FROM t1 INNER JOIN t2 ON t1.id = t2.id;",
            "SELECT * FROM t1 LEFT JOIN t2 ON t1.id = t2.id;",
            "SELECT * FROM t1 LEFT OUTER JOIN t2 ON t1.id = t2.id;",
            "SELECT * FROM t1 RIGHT JOIN t2 ON t1.id = t2.id;",
            "SELECT * FROM t1 FULL JOIN t2 ON t1.id = t2.id;",
            "SELECT * FROM t1 CROSS JOIN t2;",
            "SELECT * FROM t1 NATURAL JOIN t2;",
        ]
        stmts += joins

        # USING join (some dialects)
        stmts += [
            "SELECT * FROM t1 JOIN t2 USING(id);",
            "SELECT * FROM t1 LEFT JOIN t2 USING(id);",
        ]

        # Subqueries and EXISTS
        stmts += [
            "SELECT (SELECT MAX(x) FROM t2) AS m FROM t1;",
            "SELECT a FROM (SELECT 1 AS a) sub;",
            "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.id = t.id);",
            "SELECT a FROM t WHERE NOT EXISTS (SELECT 1 FROM u);",
            "SELECT a FROM t WHERE a IN (SELECT a FROM u);",
        ]

        # Set operations
        stmts += [
            "SELECT a FROM t UNION SELECT a FROM u;",
            "SELECT a FROM t UNION ALL SELECT a FROM u;",
            "SELECT a FROM t INTERSECT SELECT a FROM u;",
            "SELECT a FROM t EXCEPT SELECT a FROM u;",
        ]

        # Windows (may be unsupported; will be filtered)
        stmts += [
            "SELECT ROW_NUMBER() OVER (ORDER BY a) FROM t;",
            "SELECT a, SUM(b) OVER (PARTITION BY a ORDER BY c ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t;",
        ]

        # INSERT
        stmts += [
            "INSERT INTO t (a, b) VALUES (1, 'x');",
            "INSERT INTO t VALUES (DEFAULT, NULL), (2, 'y');",
            "INSERT INTO t SELECT a, b FROM u;",
            "INSERT OR REPLACE INTO t (id, name) VALUES (1, 'a');",
        ]

        # UPDATE
        stmts += [
            "UPDATE t SET a = 1, b = b + 1 WHERE id = 5;",
            "UPDATE t SET a = NULL;",
            "UPDATE t SET a = CASE WHEN a IS NULL THEN 0 ELSE a END;",
        ]

        # DELETE
        stmts += [
            "DELETE FROM t WHERE id IN (SELECT id FROM u WHERE u.flag = 1);",
            "DELETE FROM t;",
        ]

        # CREATE TABLE variations
        stmts += [
            "CREATE TABLE t (id INT PRIMARY KEY, name TEXT);",
            "CREATE TABLE IF NOT EXISTS u (id INTEGER, val VARCHAR(255) DEFAULT 'd', flag BOOLEAN NOT NULL, created DATE, amount DECIMAL(10,2), data BLOB);",
            "CREATE TABLE q (a INT, b INT, CONSTRAINT pk PRIMARY KEY (a, b));",
            "CREATE TABLE fk (a INT REFERENCES q(a) ON DELETE CASCADE ON UPDATE SET NULL);",
            "CREATE TABLE \"Case\" (\"Col\" INTEGER);",
        ]

        # CREATE INDEX
        stmts += [
            "CREATE INDEX idx_t_a ON t (a);",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_t_ab ON t (a, b DESC);",
        ]

        # ALTER TABLE
        stmts += [
            "ALTER TABLE t ADD COLUMN c INT;",
            "ALTER TABLE t DROP COLUMN c;",
            "ALTER TABLE t RENAME COLUMN a TO a1;",
            "ALTER TABLE t RENAME TO t_renamed;",
        ]

        # DROP statements
        stmts += [
            "DROP TABLE t;",
            "DROP TABLE IF EXISTS t;",
        ]

        # Transactions and explain (may be unsupported; will be filtered)
        stmts += [
            "BEGIN;",
            "COMMIT;",
            "ROLLBACK;",
            "EXPLAIN SELECT * FROM t;",
        ]

        # WITH/CTE
        stmts += [
            "WITH x AS (SELECT 1 AS a), y AS (SELECT a+1 AS b FROM x) SELECT b FROM y WHERE b > 1;",
            "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r WHERE n < 3) SELECT * FROM r;",
        ]

        # Casting and misc
        stmts += [
            "SELECT CAST('1' AS INT);",
            "SELECT CAST(a AS TEXT) FROM t;",
        ]

        # Ensure semicolon variants for parsers that dislike semicolons
        expanded: List[str] = []
        for s in stmts:
            expanded.append(s)
            s_strip = s.strip()
            if s_strip.endswith(";"):
                expanded.append(s_strip[:-1])
            else:
                expanded.append(s_strip + ";")
        # Deduplicate while preserving order
        seen: Set[str] = set()
        unique: List[str] = []
        for s in expanded:
            if s not in seen:
                unique.append(s)
                seen.add(s)
        return unique

    def _filter_valid(self, parse_func: Optional[Callable[[str], object]], candidates: List[str]) -> List[str]:
        if parse_func is None:
            # Without a parser, we cannot filter; return a moderated candidate list
            # Prefer a diverse but smaller subset
            return self._select_diverse_subset(candidates, max_count=50)
        valid: List[str] = []
        for s in candidates:
            ok = self._try_parse(parse_func, s)
            if not ok:
                # Try slight whitespace normalization
                s2 = re.sub(r"\s+", " ", s).strip()
                ok = self._try_parse(parse_func, s2)
                if ok:
                    s = s2
            if ok:
                if s not in valid:
                    valid.append(s)
        # If we validated too many, reduce while keeping diversity
        if len(valid) > 100:
            valid = self._select_diverse_subset(valid, max_count=70)
        return valid

    def _try_parse(self, parse_func: Callable[[str], object], s: str) -> bool:
        try:
            res = parse_func(s)
            # Some parsers return (ast, rest) or list; we don't care
            _ = res
            return True
        except Exception:
            # Try toggling semicolon
            s_strip = s.strip()
            alt = None
            if s_strip.endswith(";"):
                alt = s_strip[:-1]
            else:
                alt = s_strip + ";"
            try:
                res2 = parse_func(alt)
                _ = res2
                return True
            except Exception:
                return False

    def _fallback_statements(self) -> List[str]:
        # A compact set of broadly valid SQL statements
        return [
            "SELECT 1;",
            "SELECT 'x', 2.5, 3e2;",
            "SELECT * FROM t;",
            "SELECT a, b FROM t WHERE a = 1;",
            "SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(*) > 1;",
            "SELECT a FROM t ORDER BY a DESC LIMIT 10 OFFSET 5;",
            "SELECT t1.a, t2.b FROM t1 JOIN t2 ON t1.id = t2.id;",
            "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.id = t.id);",
            "SELECT a FROM t UNION SELECT a FROM u;",
            "INSERT INTO t (a, b) VALUES (1, 'x');",
            "INSERT INTO t SELECT a, b FROM u;",
            "UPDATE t SET a = 1 WHERE id = 5;",
            "DELETE FROM t WHERE id IN (SELECT id FROM u);",
            "CREATE TABLE t (id INT PRIMARY KEY, name TEXT);",
            "CREATE INDEX idx_t_a ON t (a);",
            "ALTER TABLE t ADD COLUMN c INT;",
            "DROP TABLE t;",
        ]

    def _select_diverse_subset(self, statements: List[str], max_count: int) -> List[str]:
        # Greedy set cover on features to keep diversity
        feats_per_stmt: List[Tuple[Set[str], str]] = []
        for s in statements:
            feats = self._features(s)
            feats_per_stmt.append((feats, s))

        # Ensure we cover distinct categories (by first keyword)
        categories = {}
        for feats, s in feats_per_stmt:
            cat = self._category(s)
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((feats, s))

        selected: List[str] = []
        covered: Set[str] = set()

        # First pick one from each category if possible
        primary_cats = [
            "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE TABLE", "CREATE INDEX",
            "ALTER TABLE", "DROP TABLE", "WITH", "EXPLAIN", "BEGIN", "COMMIT", "ROLLBACK"
        ]
        for cat in primary_cats:
            if cat in categories:
                best = None
                best_gain = -1
                for feats, s in categories[cat]:
                    gain = len(feats - covered)
                    if gain > best_gain:
                        best_gain = gain
                        best = s
                if best and best not in selected:
                    selected.append(best)
                    covered |= self._features(best)
                    if len(selected) >= max_count:
                        return selected

        # Then greedy selection for remaining
        # Sort candidates by decreasing new features, tie-break by shorter length
        remaining = [s for s in statements if s not in selected]
        while remaining and len(selected) < max_count:
            best_stmt = None
            best_gain = -1
            best_len = 10**9
            for s in remaining:
                feats = self._features(s)
                gain = len(feats - covered)
                if gain > best_gain or (gain == best_gain and len(s) < best_len):
                    best_gain = gain
                    best_len = len(s)
                    best_stmt = s
            if best_stmt is None or best_gain <= 0:
                # No further gains; optionally add a couple for variety
                for s in remaining:
                    if len(selected) >= max_count:
                        break
                    selected.append(s)
                break
            selected.append(best_stmt)
            covered |= self._features(best_stmt)
            remaining = [s for s in remaining if s != best_stmt]

        return selected[:max_count]

    def _features(self, s: str) -> Set[str]:
        feats: Set[str] = set()
        text = s.strip()
        # Punctuation tokens
        for ch in "(),.*=<>!+-/%;":
            if ch in text:
                feats.add(f"PUNC:{ch}")
        if "!=" in text:
            feats.add("OP:NE_BANG")
        if "<>" in text:
            feats.add("OP:NE_ANGLE")
        if "||" in text:
            feats.add("OP:CONCAT")
        if " IS NOT NULL" in text.upper():
            feats.add("PRED:IS_NOT_NULL")
        if " IS NULL" in text.upper():
            feats.add("PRED:IS_NULL")
        if " BETWEEN " in text.upper():
            feats.add("PRED:BETWEEN")
        if " IN (" in text.upper():
            feats.add("PRED:IN")
        if " EXISTS " in text.upper():
            feats.add("PRED:EXISTS")
        if " LIKE " in text.upper():
            feats.add("PRED:LIKE")
        # Numeric formats
        if re.search(r"\b\d+\.\d+\b", text):
            feats.add("NUM:DECIMAL")
        if re.search(r"\b\d+e[+-]?\d+\b", text, flags=re.IGNORECASE):
            feats.add("NUM:EXP")
        # String literal detection with escape
        if "''" in text:
            feats.add("STR:ESCAPE_SINGLE_QUOTE")
        if "/*" in text and "*/" in text:
            feats.add("COMMENT:BLOCK")
        if re.search(r"--", text):
            feats.add("COMMENT:LINE")
        # Keywords/features
        kw = re.findall(r"[A-Za-z_]+", text.upper())
        for w in kw:
            if w in {
                "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "WITH",
                "UNION", "INTERSECT", "EXCEPT", "ALL", "DISTINCT", "FROM", "WHERE",
                "GROUP", "BY", "HAVING", "ORDER", "LIMIT", "OFFSET", "ASC", "DESC",
                "JOIN", "LEFT", "RIGHT", "FULL", "INNER", "OUTER", "CROSS", "NATURAL", "USING", "ON",
                "AS", "INTO", "VALUES", "SET", "TABLE", "INDEX", "IF", "EXISTS", "NOT", "NULL",
                "PRIMARY", "KEY", "FOREIGN", "REFERENCES", "CONSTRAINT", "UNIQUE", "DEFAULT",
                "AND", "OR", "CASE", "WHEN", "THEN", "ELSE", "END", "IS", "TRUE", "FALSE",
                "OVER", "PARTITION", "ROWS", "PRECEDING", "CURRENT", "ROW", "WINDOW",
                "RECURSIVE", "EXPLAIN", "BEGIN", "COMMIT", "ROLLBACK", "ON", "DELETE", "UPDATE",
            }:
                feats.add(f"KW:{w}")
        # Category
        feats.add(f"CAT:{self._category(text)}")
        return feats

    def _category(self, s: str) -> str:
        t = s.strip().lstrip("/*- ").upper()
        # First keyword or composite for CREATE/ALTER/DROP
        tokens = re.findall(r"[A-Z_]+", t)
        if not tokens:
            return "UNKNOWN"
        if tokens[0] in {"CREATE", "ALTER", "DROP"} and len(tokens) > 1:
            return f"{tokens[0]} {tokens[1]}"
        return tokens[0]