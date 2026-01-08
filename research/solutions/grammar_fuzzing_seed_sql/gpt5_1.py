import os

class Solution:
    def solve(self, resources_path: str) -> list[str]:
        stmts = [
            "/* simple literal */ SELECT 1;",
            "-- simple select\nSELECT * FROM t;",
            "SELECT a, b AS beta FROM foo AS t WHERE a = 1 AND b <> 2 OR c >= 3;",
            "SELECT DISTINCT a, b FROM foo;",
            "SELECT ALL a FROM foo;",
            "SELECT COUNT(*), SUM(a), MIN(b), MAX(c), AVG(d) FROM metrics;",
            "SELECT CASE WHEN a IS NULL THEN 0 WHEN a > 0 THEN 1 ELSE -1 END AS sign FROM vals;",
            "SELECT -a AS neg, +b AS pos FROM nums;",
            "SELECT a + b * 2 - 5 / 3.0 AS expr FROM calc;",
            "SELECT COALESCE(a, 0), NULLIF(a, 0) FROM t;",
            "SELECT a IN (1, 2, 3) AS in_list FROM t;",
            "SELECT a BETWEEN 1 AND 10 AS in_range FROM t;",
            "SELECT a LIKE 'abc%' AS starts, a NOT LIKE '%xyz' AS ends FROM t;",
            "SELECT a IS NULL AS isnull, b IS NOT NULL AS notnull FROM t;",
            "SELECT EXISTS (SELECT 1 FROM t2 WHERE t2.id = t.id) AS ex FROM t;",
            "SELECT (SELECT MAX(x) FROM t2 WHERE t2.fk = t.id) AS m FROM t;",
            "SELECT func(a, 'str', 1.23, TRUE, NULL) FROM t;",
            "SELECT CURRENT_TIMESTAMP, CURRENT_DATE FROM t;",
            'SELECT "col", "Col Name" FROM "Table Name";',
            "SELECT 'O''Reilly' AS s, 0.5 AS n, 1e-3 AS e;",
            "SELECT CAST(a AS INTEGER), CAST('1' AS TEXT) FROM t;",
            "SELECT a FROM t ORDER BY a DESC;",
            "SELECT a FROM t ORDER BY 1 ASC;",
            "SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(*) > 1;",
            "SELECT a FROM t LIMIT 10;",
            "SELECT a FROM t LIMIT 10 OFFSET 5;",
            "SELECT a FROM t WHERE a IN (SELECT a FROM t2 WHERE t2.b = t.a) AND b NOT IN (1, 2, 3);",
            "SELECT a FROM t WHERE a LIKE '%\\_%' ESCAPE '\\';",
            "SELECT a FROM t WHERE NOT (a > 1 AND b < 3) OR c = 4;",
            "SELECT t1.a, t2.b FROM t1 INNER JOIN t2 ON t1.id = t2.id;",
            "SELECT * FROM t1 LEFT JOIN t2 ON t1.id = t2.id;",
            "SELECT * FROM t1 RIGHT JOIN t2 ON t1.id = t2.id;",
            "SELECT * FROM t1 JOIN t2 USING (id);",
            "SELECT * FROM (SELECT * FROM t) AS sub;",
            "SELECT * FROM t1, t2 WHERE t1.id = t2.id;",
            "SELECT a FROM t1 UNION SELECT a FROM t2;",
            "SELECT a FROM t1 UNION ALL SELECT a FROM t2;",
            "(SELECT a FROM t1) INTERSECT (SELECT a FROM t2);",
            "(SELECT a FROM t1) EXCEPT (SELECT a FROM t2);",
            "WITH cte AS (SELECT 1 AS id) SELECT cte.id FROM cte;",
            "SELECT a || '-' || b AS concat FROM t;",
            "INSERT INTO t (a, b, c) VALUES (1, 'x', NULL), (2, 'y', 3.14);",
            "INSERT INTO t DEFAULT VALUES;",
            "INSERT INTO t (a, b) SELECT a, b FROM t2 WHERE a > 0;",
            "UPDATE t SET a = a + 1, b = CASE WHEN b IS NULL THEN 0 ELSE b END WHERE id IN (SELECT id FROM t2);",
            "UPDATE t SET a = 1 FROM u WHERE t.id = u.id;",
            "DELETE FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.id = t.id);",
            "DELETE FROM t USING u WHERE t.id = u.id;",
            "CREATE TABLE IF NOT EXISTS foo (id INTEGER PRIMARY KEY, name VARCHAR(100) NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, amount DECIMAL(10,2), flag BOOLEAN DEFAULT TRUE, ref_id INT REFERENCES other(id), CHECK (amount >= 0), UNIQUE (name, ref_id));",
            "CREATE INDEX IF NOT EXISTS idx_foo_name ON foo(name);",
            "ALTER TABLE foo ADD COLUMN extra TEXT;",
            "DROP INDEX IF EXISTS idx_foo_name;",
            "DROP TABLE IF EXISTS foo;",
            "-- parameters\nSELECT * FROM t WHERE id = :id AND name = $1 AND val = ?;",
        ]

        # Adaptive additions based on available grammar/keywords
        try:
            files = []
            grammar = os.path.join(resources_path, "sql_grammar.txt")
            if os.path.exists(grammar):
                files.append(grammar)
            se_dir = os.path.join(resources_path, "sql_engine")
            if os.path.isdir(se_dir):
                for fn in ("tokenizer.py", "parser.py", "ast_nodes.py", "__init__.py"):
                    fpath = os.path.join(se_dir, fn)
                    if os.path.exists(fpath):
                        files.append(fpath)
            content = ""
            for f in files:
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        content += fh.read() + "\n"
                except Exception:
                    pass
            U = content.upper()

            if "ILIKE" in U:
                stmts.append("SELECT name ILIKE '%AbC%' FROM people;")
            if "SIMILAR TO" in U or ("SIMILAR" in U and " TO " in U):
                stmts.append("SELECT name SIMILAR TO 'A[0-9]+' FROM people;")
            if "REGEXP" in U:
                stmts.append("SELECT name REGEXP '^[a-z]+$' FROM people;")
            if "NULLS FIRST" in U or "NULLS LAST" in U:
                stmts.append("SELECT a FROM t ORDER BY a DESC NULLS LAST;")
            if " OVER " in U:
                stmts.append("SELECT SUM(a) OVER (PARTITION BY b ORDER BY c) FROM t;")
            if "ON CONFLICT" in U:
                stmts.append("INSERT INTO t(id, name) VALUES (1, 'a') ON CONFLICT (id) DO NOTHING;")
            if "RETURNING" in U:
                stmts.append("UPDATE t SET name = 'x' WHERE id = 1 RETURNING id, name;")
            if "TEMPORARY" in U or " TEMP " in U:
                stmts.append("CREATE TEMP TABLE tmp (id INT);")
            if "VIEW" in U:
                stmts.append("CREATE VIEW v AS SELECT a FROM t;")
                stmts.append("DROP VIEW v;")
            if "TRUNCATE" in U:
                stmts.append("TRUNCATE TABLE t;")
            if "RENAME" in U and "ALTER TABLE" in U:
                stmts.append("ALTER TABLE foo RENAME TO foo_renamed;")
            if "EXPLAIN" in U:
                stmts.append("EXPLAIN SELECT * FROM t;")
            if "BEGIN" in U or "TRANSACTION" in U:
                stmts.append("BEGIN TRANSACTION;")
                stmts.append("COMMIT;")
            if "IS DISTINCT FROM" in U:
                stmts.append("SELECT (a IS DISTINCT FROM b) AS diff FROM t;")
            if "WINDOW" in U and "OVER" in U:
                stmts.append("SELECT AVG(a) OVER w FROM t WINDOW w AS (PARTITION BY b ORDER BY c);")
        except Exception:
            pass

        # Cap the total number to avoid excessive inefficiency
        if len(stmts) > 70:
            stmts = stmts[:70]
        return stmts