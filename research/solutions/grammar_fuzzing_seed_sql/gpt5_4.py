import os
import re
from pathlib import Path
from typing import List, Set


def _read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _extract_upper_keywords(text: str) -> Set[str]:
    # Remove obvious string literals to avoid picking up keywords inside them
    text = re.sub(r"'.*?'", " ", text, flags=re.S)
    text = re.sub(r'".*?"', " ", text, flags=re.S)
    # Remove comments in Python and SQL-like
    text = re.sub(r"#.*", " ", text)  # Python comments
    text = re.sub(r"--.*", " ", text)  # SQL single-line comments
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)  # SQL block comments
    # Extract uppercase tokens
    kws = set(re.findall(r"\b[A-Z_][A-Z0-9_]*\b", text))
    return kws


def _discover_keywords(resources_path: str) -> Set[str]:
    base = Path(resources_path)
    kws: Set[str] = set()

    grammar = base / "sql_grammar.txt"
    if grammar.exists():
        kws |= _extract_upper_keywords(_read_text_safe(grammar))

    engine_dir = base / "sql_engine"
    for fname in ("parser.py", "tokenizer.py", "ast_nodes.py", "__init__.py"):
        f = engine_dir / fname
        if f.exists():
            kws |= _extract_upper_keywords(_read_text_safe(f))

    # Common SQL keywords fallback to improve robustness if extraction fails
    common = {
        "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "EXISTS", "LIKE", "BETWEEN",
        "IS", "NULL", "DISTINCT", "GROUP", "BY", "HAVING", "ORDER", "ASC", "DESC",
        "LIMIT", "OFFSET", "UNION", "ALL", "INTERSECT", "EXCEPT",
        "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "OUTER", "CROSS", "USING", "ON",
        "AS", "CASE", "WHEN", "THEN", "ELSE", "END",
        "INSERT", "INTO", "VALUES", "UPDATE", "SET", "DELETE",
        "CREATE", "TABLE", "PRIMARY", "KEY", "UNIQUE", "CHECK", "DEFAULT", "REFERENCES",
        "INDEX", "VIEW", "DROP", "IF", "EXISTS",
        "ALTER", "ADD", "COLUMN", "RENAME", "TO",
        "BEGIN", "COMMIT", "ROLLBACK",
        "WITH", "CAST",
    }
    if not kws:
        kws = common
    else:
        # Ensure basics exist
        kws |= {"SELECT", "FROM", "WHERE", "AS", "ON"}
    return kws


class Solution:
    def solve(self, resources_path: str) -> List[str]:
        kws = _discover_keywords(resources_path)

        def has_all(*words: str) -> bool:
            return all(w in kws for w in words if w)

        def has_any(*words: str) -> bool:
            return any(w in kws for w in words if w)

        stmts: List[str] = []

        def add(stmt: str, required: Set[str] = None, any_of: Set[str] = None) -> None:
            if required and not all(w in kws for w in required):
                return
            if any_of and not any(w in kws for w in any_of):
                return
            stmts.append(stmt)

        # Base simple selects
        add("SELECT 1")
        add("SELECT 1 FROM t", {"SELECT", "FROM"})
        add("SELECT * FROM t", {"SELECT", "FROM"})
        add("SELECT DISTINCT a, b FROM t", {"SELECT", "FROM", "DISTINCT"})
        add("SELECT a AS aa, b + 1 AS bp, -c AS cn FROM t AS tt", {"SELECT", "FROM", "AS"})

        # WHERE predicates
        add("SELECT a FROM t WHERE a = 1", {"SELECT", "FROM", "WHERE"})
        add("SELECT a FROM t WHERE a <> 1 AND b >= 2 OR c < 3", {"SELECT", "FROM", "WHERE", "AND", "OR"})
        add("SELECT a FROM t WHERE NOT (a BETWEEN 1 AND 10)", {"SELECT", "FROM", "WHERE", "NOT", "BETWEEN"})
        if has_all("LIKE"):
            if has_all("ESCAPE"):
                add("SELECT a FROM t WHERE name LIKE 'A%' ESCAPE '\\'", {"SELECT", "FROM", "WHERE", "LIKE", "ESCAPE"})
            add("SELECT a FROM t WHERE name NOT LIKE '%a%'", {"SELECT", "FROM", "WHERE", "LIKE", "NOT"})
        add("SELECT a FROM t WHERE a IN (1, 2, 3) AND b NOT IN ('x', 'y')", {"SELECT", "FROM", "WHERE", "IN", "AND", "NOT"})
        if has_all("EXISTS"):
            add("SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.id = t.id)", {"SELECT", "FROM", "WHERE", "EXISTS"})
            add("SELECT a FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.id = t.id)", {"SELECT", "FROM", "WHERE", "EXISTS", "NOT"})
        add("SELECT a FROM t WHERE a IS NULL OR a IS NOT NULL", {"SELECT", "FROM", "WHERE", "IS", "NULL", "OR", "NOT"})

        # GROUP BY / HAVING
        if has_all("GROUP", "BY"):
            add("SELECT b, COUNT(*) AS c, SUM(a) AS s FROM t GROUP BY b", {"SELECT", "FROM", "GROUP", "BY"})
            if has_all("HAVING"):
                add("SELECT b, COUNT(*) AS c FROM t GROUP BY b HAVING COUNT(*) > 1", {"SELECT", "FROM", "GROUP", "BY", "HAVING"})
        add("SELECT COUNT(DISTINCT a) FROM t", {"SELECT", "FROM", "DISTINCT"})

        # ORDER BY / LIMIT / OFFSET
        if has_all("ORDER", "BY"):
            add("SELECT a, b FROM t ORDER BY 2 DESC, a ASC", {"SELECT", "FROM", "ORDER", "BY"})
        if has_all("LIMIT"):
            add("SELECT a FROM t ORDER BY a LIMIT 10" if has_all("ORDER", "BY") else "SELECT a FROM t LIMIT 10", {"SELECT", "FROM", "LIMIT"})
        if has_all("LIMIT", "OFFSET"):
            add("SELECT a FROM t WHERE a > 0 LIMIT 5 OFFSET 2", {"SELECT", "FROM", "WHERE", "LIMIT", "OFFSET"})

        # Joins
        if has_all("JOIN", "ON"):
            add("SELECT t.a, u.b FROM t JOIN u ON t.id = u.id", {"SELECT", "FROM", "JOIN", "ON"})
            if has_all("INNER"):
                add("SELECT t.a FROM t INNER JOIN u ON t.id = u.id", {"SELECT", "FROM", "INNER", "JOIN", "ON"})
            if has_all("LEFT"):
                add("SELECT t.a FROM t LEFT JOIN u ON t.id = u.id", {"SELECT", "FROM", "LEFT", "JOIN", "ON"})
            if has_all("RIGHT"):
                add("SELECT t.a FROM t RIGHT JOIN u ON t.id = u.id", {"SELECT", "FROM", "RIGHT", "JOIN", "ON"})
            if has_all("FULL"):
                add("SELECT t.a FROM t FULL JOIN u ON t.id = u.id", {"SELECT", "FROM", "FULL", "JOIN", "ON"})
            if has_all("CROSS"):
                add("SELECT * FROM t CROSS JOIN u", {"SELECT", "FROM", "CROSS", "JOIN"})
            if has_all("USING"):
                add("SELECT * FROM t INNER JOIN u USING (id)" if has_all("INNER") else "SELECT * FROM t JOIN u USING (id)", {"SELECT", "FROM", "JOIN", "USING"})
            add("SELECT * FROM a JOIN b ON (a.id = b.id AND a.x = b.x)", {"SELECT", "FROM", "JOIN", "ON", "AND"})
        if has_all("NATURAL", "JOIN"):
            add("SELECT * FROM t NATURAL JOIN u", {"SELECT", "FROM", "NATURAL", "JOIN"})

        # Subqueries
        add("SELECT x.cnt FROM (SELECT COUNT(*) AS cnt FROM t) AS x", {"SELECT", "FROM", "AS"})
        add("SELECT s.d1, s.d2 FROM (SELECT a AS d1, b AS d2 FROM t) AS s WHERE s.d1 > 1", {"SELECT", "FROM", "AS", "WHERE"})

        # Set operations
        if has_all("UNION"):
            add("SELECT a FROM t UNION SELECT a FROM u", {"SELECT", "FROM", "UNION"})
            if has_all("ALL"):
                add("SELECT a FROM t UNION ALL SELECT a FROM u", {"SELECT", "FROM", "UNION", "ALL"})
        if has_all("INTERSECT"):
            add("SELECT a FROM t INTERSECT SELECT a FROM u", {"SELECT", "FROM", "INTERSECT"})
        if has_all("EXCEPT"):
            add("SELECT a FROM t EXCEPT SELECT a FROM u", {"SELECT", "FROM", "EXCEPT"})

        # CASE expressions
        if has_all("CASE", "WHEN", "THEN", "END"):
            add("SELECT CASE WHEN a > 10 THEN 'big' WHEN a = 10 THEN 'ten' ELSE 'small' END AS size FROM t", {"SELECT", "FROM", "CASE", "WHEN", "THEN", "ELSE", "END", "AS"})
            add("SELECT CASE a WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END FROM t", {"SELECT", "FROM", "CASE", "WHEN", "THEN", "ELSE", "END"})

        # Functions and arithmetic
        add("SELECT COALESCE(name, 'n/a') AS n FROM t", {"SELECT", "FROM", "AS"})
        add("SELECT NULLIF(a, 0), ABS(-a), ROUND(1.234, 2) FROM t", {"SELECT", "FROM"})

        # CAST
        if has_all("CAST", "AS"):
            add("SELECT CAST(a AS INTEGER) FROM t", {"SELECT", "FROM", "CAST", "AS"})
            add("SELECT CAST('2020-01-01' AS DATE)", {"SELECT", "CAST", "AS"})

        # WITH (CTE)
        if has_all("WITH", "AS"):
            add("WITH cte AS (SELECT id FROM t) SELECT * FROM cte", {"WITH", "AS", "SELECT", "FROM"})

        # Strings and identifiers
        add("SELECT 'It''s fine' AS msg", {"SELECT", "AS"})

        # DML
        if has_all("INSERT", "INTO", "VALUES"):
            add("INSERT INTO t (id, name, price) VALUES (1, 'a', 1.23), (2, 'b', NULL)", {"INSERT", "INTO", "VALUES"})
            add("INSERT INTO t SELECT id, name, price FROM u WHERE price > 0", {"INSERT", "INTO", "SELECT", "FROM", "WHERE"})
        if has_all("UPDATE", "SET"):
            add("UPDATE t SET name = 'x', price = price + 1 WHERE id IN (SELECT id FROM u)", {"UPDATE", "SET", "WHERE", "IN", "SELECT", "FROM"})
            add("UPDATE t SET active = NOT active", {"UPDATE", "SET", "NOT"})
        if has_all("DELETE", "FROM"):
            add("DELETE FROM t WHERE id = 1", {"DELETE", "FROM", "WHERE"})

        # DDL
        if has_all("CREATE", "TABLE"):
            # Keep types generic; parser typically accepts identifiers for types
            add("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL, price DECIMAL(10,2) DEFAULT 0.0, created DATE, active BOOLEAN DEFAULT TRUE)", {"CREATE", "TABLE"})
            req_u = {"CREATE", "TABLE"}
            add("CREATE TABLE u (id INTEGER PRIMARY KEY, t_id INTEGER REFERENCES t(id), name VARCHAR(50) UNIQUE, flag BOOLEAN, CHECK (t_id > 0))", req_u)
        if has_all("CREATE", "INDEX"):
            add("CREATE INDEX idx_t_name ON t(name)", {"CREATE", "INDEX"})
            if has_all("UNIQUE"):
                add("CREATE UNIQUE INDEX idx_u_name ON u(name)", {"CREATE", "INDEX", "UNIQUE"})
        if has_all("CREATE", "VIEW", "AS"):
            add("CREATE VIEW v AS SELECT a FROM t WHERE a > 0", {"CREATE", "VIEW", "AS", "SELECT", "FROM", "WHERE"})
        if has_all("DROP", "VIEW"):
            add("DROP VIEW v", {"DROP", "VIEW"})
        if has_all("DROP", "TABLE"):
            if has_all("IF", "EXISTS"):
                add("DROP TABLE IF EXISTS t", {"DROP", "TABLE", "IF", "EXISTS"})
            else:
                add("DROP TABLE t", {"DROP", "TABLE"})
        if has_all("DROP", "INDEX"):
            add("DROP INDEX idx_t_name", {"DROP", "INDEX"})
        if has_all("ALTER", "TABLE"):
            add("ALTER TABLE t ADD COLUMN extra INTEGER DEFAULT 0", {"ALTER", "TABLE", "ADD", "COLUMN"})
            add("ALTER TABLE t DROP COLUMN extra", {"ALTER", "TABLE", "DROP", "COLUMN"} if "DROP" in kws else {"ALTER", "TABLE", "COLUMN"})
            add("ALTER TABLE t RENAME COLUMN name TO title", {"ALTER", "TABLE", "RENAME", "COLUMN", "TO"} if has_all("RENAME", "COLUMN", "TO") else {"ALTER", "TABLE"})
            add("ALTER TABLE t RENAME TO t2", {"ALTER", "TABLE", "RENAME", "TO"} if has_all("RENAME", "TO") else {"ALTER", "TABLE"})
            add("ALTER TABLE t ALTER COLUMN price SET DEFAULT 0.0", {"ALTER", "TABLE", "ALTER", "COLUMN", "SET", "DEFAULT"} if has_all("ALTER", "SET", "DEFAULT") else {"ALTER", "TABLE"})

        # Transactions
        if has_all("BEGIN"):
            add("BEGIN", {"BEGIN"})
        if has_all("COMMIT"):
            add("COMMIT", {"COMMIT"})
        if has_all("ROLLBACK"):
            add("ROLLBACK", {"ROLLBACK"})

        # Deduplicate keeping order
        seen = set()
        final: List[str] = []
        for s in stmts:
            if s not in seen:
                seen.add(s)
                final.append(s)

        # Keep up to 50 to balance efficiency and coverage
        return final[:50]