import sys
import os

class Solution:
    def solve(self, resources_path: str) -> list[str]:
        return [
            # =========================================================================
            # Basic SELECT statements & Tokenizer tests
            # =========================================================================
            
            """SELECT 1;""",
            
            """select 1, -1, 1.0, .5, 1.2e-3, 'string', "string", null, true, false, NuLl, fAlSe from "My_Table";""",

            """SELECT DISTINCT col1, col2 FROM my_table;""",

            """SELECT my_table.*, col3 FROM my_table;""",

            # =========================================================================
            # Comprehensive SELECT Statements
            # =========================================================================
            
            """SELECT t1.col1 AS "alias 1", MAX(t1.col2) FROM table1 AS t1 JOIN table2 t2 ON t1.id = t2.t1_id WHERE t1.col1 > (SELECT MIN(col3) FROM table3) AND t2.col4 LIKE '%pattern' GROUP BY 1 HAVING COUNT(*) > 0 ORDER BY 2 DESC, t1.col1 ASC LIMIT 10 OFFSET 5;""",

            """SELECT * FROM t1 INNER JOIN t2 ON t1.id=t2.id LEFT OUTER JOIN t3 USING(id) RIGHT JOIN t4 ON t1.id=t4.id FULL JOIN t5 ON t1.id=t5.id CROSS JOIN t6, t7 NATURAL JOIN t8;""",

            """SELECT c1 * -c2 + c3 / c4 % c5, c1 > c2 OR c1 >= c2 OR c1 < c2 OR c1 <= c2 OR c1 = c2 AND c1 != c2 AND c1 <> c2 AND NOT (c1 IS NULL OR c2 IS NOT NULL) OR name LIKE 'a%' AND name NOT LIKE 'b%' OR id IN (1, 2, 3.14) AND id NOT IN (4, 5) OR val BETWEEN 1 AND 2 AND val NOT BETWEEN 3 AND 4 FROM op_test;""",

            """SELECT c1, c2 FROM t1 UNION SELECT c1, c2 FROM t2 UNION ALL SELECT c1, c2 FROM t3 INTERSECT SELECT c1, c2 FROM t4 EXCEPT SELECT c1, c2 FROM t5;""",

            """SELECT CASE WHEN c1 > 0 THEN 'pos' WHEN c1 < 0 THEN 'neg' ELSE 'zero' END, CASE c2 WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END FROM t;""",

            """SELECT (SELECT max(c1) FROM t2), t1.* FROM t1, (SELECT c2 FROM t3) AS t3_sub WHERE EXISTS (SELECT 1 FROM t4 WHERE t4.id = t1.id);""",

            """SELECT COUNT(*), COUNT(col1), COUNT(DISTINCT col1), custom_func(col1, 1, 'a'), CAST(col1 AS VARCHAR(20)) FROM t;""",

            # =========================================================================
            # Common Table Expressions (CTEs)
            # =========================================================================
            
            """WITH my_cte AS (SELECT c1 FROM t1) SELECT * FROM my_cte;""",

            """WITH RECURSIVE R(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM R WHERE n < 5) SELECT * FROM R;""",

            # =========================================================================
            # DDL - Data Definition Language
            # =========================================================================
            
            """CREATE TABLE t (c1 INT PRIMARY KEY, c2 VARCHAR(20) NOT NULL UNIQUE, c3 TEXT DEFAULT 'default_text', c4 DECIMAL(5,2));""",

            """CREATE TABLE t2 AS SELECT * FROM t1 WHERE 1=0;""",
            
            """DROP TABLE IF EXISTS t;""",

            """ALTER TABLE t ADD COLUMN c5 INT DEFAULT 0;""",
            
            """CREATE VIEW v AS SELECT c1, c2 FROM t;""",
            
            """DROP VIEW IF EXISTS v;""",

            # =========================================================================
            # DML - Data Manipulation Language
            # =========================================================================
            
            """INSERT INTO t VALUES (1, 'a', 'b', 1.1);""",

            """INSERT INTO t (c2, c1) VALUES ('a', 1), ('b', 2);""",

            """INSERT INTO t2 SELECT * FROM t1;""",

            """UPDATE t SET c2 = 'new_val', c3 = 'updated' WHERE c1 > 10;""",

            """UPDATE t SET c2 = 'all_new';""",

            """DELETE FROM t WHERE c1 = 1;""",

            """DELETE FROM t;""",

            """VALUES (1, 'a'), (2, 'b');"""
        ]