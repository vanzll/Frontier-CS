import sys
import os

class Solution:
    def solve(self, resources_path: str) -> list[str]:
        """
        Return SQL test cases designed to maximize parser coverage.
        
        Args:
            resources_path: Path to the resources directory.
        
        Returns:
            list[str]: List of SQL statement strings.
        """
        
        # This list of SQL queries is hand-crafted to target a wide range of
        # syntactic constructs in a typical SQL parser. The goal is to maximize
        # line and branch coverage with a minimal number of test cases.
        # Queries are designed to be complex, combining multiple features to
        # exercise many code paths simultaneously.
        sql_queries = [
            # 1. Comprehensive SELECT: Covers aliases, aggregates, distinct, expressions,
            #    LEFT JOIN, complex ON, subqueries (scalar, IN, EXISTS), GROUP BY, HAVING,
            #    ORDER BY (alias, index, direction, nulls), LIMIT, OFFSET.
            """
            SELECT
                t1.col1 AS alias1,
                t2.col2,
                COUNT(DISTINCT t1.col3),
                SUM(t1.col4 + 5.5)
            FROM
                table1 AS t1
            LEFT OUTER JOIN
                table2 t2 ON t1.id = t2.t1_id AND t1.date > '2023-01-01'
            WHERE
                t1.col1 > (SELECT MIN(sub_col) FROM sub_table)
                AND t2.col2 IN ('A', 'B', 'C')
                AND t1.col5 LIKE 'prefix%'
                AND t1.col6 IS NOT NULL
                AND NOT EXISTS (SELECT 1 FROM table3 t3 WHERE t3.id = t1.id)
            GROUP BY
                t1.col1, t2.col2
            HAVING
                COUNT(*) > 1 AND MAX(t1.col4) < 100
            ORDER BY
                alias1 DESC NULLS LAST, 2 ASC NULLS FIRST
            LIMIT 10 OFFSET 5
            """,

            # 2. All other Join types, CASE (simple and searched), CAST, Unary operators.
            """
            SELECT
                CASE t1.c1 WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END,
                CASE WHEN t1.c2 > 10 THEN 'high' ELSE 'low' END,
                CAST(t2.c3 AS BIGINT),
                -t3.c4, +t3.c5
            FROM
                t1
            INNER JOIN t2 ON t1.id = t2.id
            RIGHT JOIN t3 USING(common_col)
            FULL JOIN t4 ON t3.id = t4.id
            CROSS JOIN t5
            WHERE t1.c1 <> 0
            """,

            # 3. DML: INSERT with multiple value tuples and various literal types.
            """
            INSERT INTO my_table (col1, col2, col3, col4, col5) 
            VALUES (1, 'hello', TRUE, 12.34, NULL), (2, 'it''s', FALSE, -5, 1.2e-3)
            """,

            # 4. DML: INSERT without column list.
            "INSERT INTO my_table VALUES (3, 'world', TRUE, 99.9, .5)",

            # 5. DML: UPDATE with expressions in SET and an IS NOT predicate.
            "UPDATE my_table SET col1 = col1 * 2, col2 = 'updated' WHERE col3 IS NOT FALSE",

            # 6. DML: DELETE with BETWEEN and logical OR with NOT LIKE.
            "DELETE FROM my_table WHERE col1 BETWEEN 5 AND 10 OR col2 NOT LIKE 'a_'",

            # 7. DDL: CREATE TABLE with a wide range of constraints and data types, including quoted identifiers.
            """
            CREATE TABLE "new_table" (
                "id" INT PRIMARY KEY,
                name VARCHAR(255) NOT NULL DEFAULT 'anonymous',
                email TEXT UNIQUE,
                value DECIMAL(10, 2) DEFAULT 0.00,
                created_at TIMESTAMP,
                is_active BOOLEAN,
                CHECK (value >= 0 AND id > 0)
            )
            """,
            
            # 8. DDL: CREATE VIEW.
            "CREATE VIEW my_view AS SELECT id, name FROM new_table WHERE is_active = TRUE",

            # 9. DDL: CREATE INDEX on multiple columns with directions.
            "CREATE INDEX my_idx ON new_table (name ASC, email DESC)",

            # 10. DDL: DROP TABLE statement.
            "DROP TABLE new_table",

            # 11. DDL: DROP VIEW statement.
            "DROP VIEW my_view",

            # 12. DDL: DROP INDEX statement (syntax with ON clause).
            "DROP INDEX my_idx ON new_table",

            # 13. UNION, comma join (implicit join), and derived tables.
            """
            SELECT c1, c2 FROM t1
            UNION
            SELECT cA, cB FROM tA
            UNION ALL
            SELECT cX, cY FROM (SELECT cX, cY FROM tX) AS derived, tZ
            """,

            # 14. All standard aggregate functions, including on expressions.
            "SELECT COUNT(*), COUNT(c1), SUM(c1 + 1), AVG(c1), MIN(c1), MAX(c1) FROM tbl",
            
            # 15. Tokenizer test: comments, quoted keywords, escaped strings, special numbers.
            """
            SELECT "select" AS "FROM", -- line comment
                   'string with '' escaped quote',
                   "identifier with spaces"
            /*
             block comment
            */
            """,

            # 16. Expressions without a FROM clause (calculator mode) and logical NOT.
            "SELECT 1 + 1, 'hello' || ' ' || 'world', NOT TRUE",

            # 17. Simple base cases to cover fundamental paths.
            "SELECT * FROM simple_table",
            "SELECT t.a FROM t",
            
            # 18. Subquery in WHERE with NOT IN.
            "SELECT c1 FROM t1 WHERE c1 NOT IN (SELECT c2 FROM t2)",

            # 19. SELECT ALL keyword to hit a specific optional keyword path.
            "SELECT ALL col1 FROM my_table",
        ]

        # Normalize queries to single lines to avoid potential parsing issues with newlines.
        return [" ".join(q.strip().split()) for q in sql_queries]