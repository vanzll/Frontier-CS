import os
import re
import ast

class Solution:
    def solve(self, resources_path: str) -> list[str]:
        """
        Generates SQL test cases to maximize parser coverage.
        Uses a strong baseline of standard SQL queries and augments with 
        grammar-specific features found by scanning the grammar file.
        """
        
        # 1. Base Suite: Comprehensive Standard SQL
        # Covers common recursive descent parser paths for DML, DDL, clauses, expressions
        tests = [
            # Basic Select
            "SELECT * FROM table1",
            "SELECT col1, col2, t1.col3 AS alias3 FROM table1 AS t1",
            "SELECT DISTINCT col1 FROM table1",
            
            # Filtering Logic (WHERE)
            "SELECT * FROM t1 WHERE col1 = 1",
            "SELECT * FROM t1 WHERE col1 > 10 AND col2 < 20",
            "SELECT * FROM t1 WHERE col1 = 1 OR col2 = 2",
            "SELECT * FROM t1 WHERE NOT (col1 = 1)",
            "SELECT * FROM t1 WHERE col1 IS NULL",
            "SELECT * FROM t1 WHERE col1 IS NOT NULL",
            "SELECT * FROM t1 WHERE col1 BETWEEN 1 AND 10",
            "SELECT * FROM t1 WHERE col1 IN (1, 2, 3)",
            "SELECT * FROM t1 WHERE col1 LIKE 'A%'",
            "SELECT * FROM t1 WHERE col1 != 5",
            
            # Joins
            "SELECT * FROM t1 INNER JOIN t2 ON t1.id = t2.id",
            "SELECT * FROM t1 LEFT JOIN t2 ON t1.id = t2.id",
            "SELECT * FROM t1 RIGHT OUTER JOIN t2 ON t1.id = t2.id",
            "SELECT * FROM t1, t2 WHERE t1.id = t2.id", # Implicit
            
            # Aggregation & Grouping
            "SELECT count(*), sum(a), avg(b), min(c), max(d) FROM t1",
            "SELECT col1, count(*) FROM t1 GROUP BY col1",
            "SELECT col1, count(*) FROM t1 GROUP BY col1 HAVING count(*) > 10",
            
            # Ordering & Limits
            "SELECT * FROM t1 ORDER BY col1 ASC, col2 DESC",
            "SELECT * FROM t1 LIMIT 10 OFFSET 5",
            
            # Subqueries
            "SELECT * FROM (SELECT id FROM t1) AS sub",
            "SELECT * FROM t1 WHERE id IN (SELECT id FROM t2 WHERE val > 10)",
            "SELECT (SELECT max(id) FROM t2) FROM t1",
            "SELECT * FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t1.id = t2.id)",
            
            # DML
            "INSERT INTO t1 VALUES (1, 'text', 3.14, TRUE, NULL)",
            "INSERT INTO t1 (c1, c2) VALUES (1, 2), (3, 4)",
            "INSERT INTO t1 SELECT * FROM t2",
            "UPDATE t1 SET c1 = 100, c2 = 'updated' WHERE id = 1",
            "UPDATE t1 SET c1 = c1 + 1",
            "DELETE FROM t1",
            "DELETE FROM t1 WHERE id < 5",
            
            # DDL
            "CREATE TABLE t1 (id INT PRIMARY KEY, name VARCHAR(255) NOT NULL)",
            "CREATE TABLE t2 (id INT, ref_id INT REFERENCES t1(id))",
            "DROP TABLE t1",
            "DROP TABLE IF EXISTS t2",
            "ALTER TABLE t1 ADD COLUMN c3 TEXT",
            "ALTER TABLE t1 DROP COLUMN c3",
            "ALTER TABLE t1 RENAME TO t1_backup",
            "CREATE INDEX idx1 ON t1 (col1)",
            "DROP INDEX idx1",
            "CREATE VIEW v1 AS SELECT * FROM t1",
            "DROP VIEW v1",
            
            # Expressions & Types
            "SELECT 1 + 2 * 3 / 4 % 5 FROM t1",
            "SELECT -1, +2 FROM t1",
            "SELECT CASE WHEN a > 0 THEN 'pos' WHEN a < 0 THEN 'neg' ELSE 'zero' END FROM t1",
            "SELECT CAST(col1 AS VARCHAR) FROM t1",
            "SELECT COALESCE(col1, 0) FROM t1",
            
            # Complex Composite
            "SELECT t1.a, count(t2.b) FROM t1 JOIN t2 ON t1.id = t2.id WHERE t1.x > 10 GROUP BY t1.a HAVING count(t2.b) < 100 ORDER BY 1 DESC"
        ]
        
        # 2. Grammar-Guided Extension
        # Scan grammar file for keywords implying support for specific SQL features 
        # that are not universally supported (to improve coverage without adding invalid queries)
        grammar_file = os.path.join(resources_path, 'sql_grammar.txt')
        if os.path.exists(grammar_file):
            try:
                with open(grammar_file, 'r') as f:
                    content = f.read()
                
                # Identify uppercase tokens which are likely keywords
                keywords = set(re.findall(r'\b[A-Z_]{3,}\b', content))
                
                extras = []
                
                # Transaction Control
                if 'TRANSACTION' in keywords:
                    extras.extend(["BEGIN TRANSACTION", "COMMIT TRANSACTION", "ROLLBACK TRANSACTION"])
                elif 'COMMIT' in keywords:
                    extras.extend(["BEGIN", "COMMIT", "ROLLBACK"])
                
                # Set Operations
                if 'UNION' in keywords: extras.append("SELECT a FROM t1 UNION SELECT a FROM t2")
                if 'INTERSECT' in keywords: extras.append("SELECT a FROM t1 INTERSECT SELECT a FROM t2")
                if 'EXCEPT' in keywords: extras.append("SELECT a FROM t1 EXCEPT SELECT a FROM t2")
                
                # Advanced Joins
                if 'CROSS' in keywords: extras.append("SELECT * FROM t1 CROSS JOIN t2")
                if 'NATURAL' in keywords: extras.append("SELECT * FROM t1 NATURAL JOIN t2")
                
                # CTEs
                if 'WITH' in keywords:
                    extras.append("WITH cte AS (SELECT * FROM t1) SELECT * FROM cte")
                
                # Window Functions
                if 'OVER' in keywords:
                    extras.append("SELECT rank() OVER (PARTITION BY c1 ORDER BY c2) FROM t1")
                
                # Other commands
                if 'EXPLAIN' in keywords: extras.append("EXPLAIN SELECT * FROM t1")
                if 'VACUUM' in keywords: extras.append("VACUUM")
                if 'TRUNCATE' in keywords: extras.append("TRUNCATE TABLE t1")
                if 'REPLACE' in keywords: extras.append("REPLACE INTO t1 VALUES (1, 'x')")
                if 'PRAGMA' in keywords: extras.append("PRAGMA foreign_keys = ON")
                
                tests.extend(extras)
                
            except Exception:
                pass
        
        return list(set(tests))