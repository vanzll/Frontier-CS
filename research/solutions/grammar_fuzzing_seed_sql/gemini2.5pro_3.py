class Solution:
    def solve(self, resources_path: str) -> list[str]:
        """
        Return SQL test cases designed to maximize parser coverage.
        
        The strategy is to manually craft a comprehensive list of SQL statements
        that cover a wide range of grammatical constructs and potential edge cases.
        This list is designed with two goals in mind:
        1.  High Coverage: Include a diverse set of queries that test everything
            from basic statements to complex combinations of clauses, expressions,
            and statement types. This aims to exercise as many code paths in the
            parser, tokenizer, and AST node definitions as possible.
        2.  Efficiency: Keep the number of test cases reasonably low to maximize
            the efficiency bonus. This is achieved by creating complex queries
            that test multiple features simultaneously, in addition to simpler
            "atomic" tests for fundamental features.
        
        The list is structured into two parts:
        - Atomic Queries: Simple statements targeting one specific feature each.
        - Combinatorial Queries: Complex statements that combine multiple features
          to test their interactions.
        """
        
        queries = [
            # --- Atomic Queries for Basic Coverage ---

            # Core SELECT and aliasing
            "SELECT 1;",
            "SELECT * FROM t;",
            "SELECT a, b FROM t;",
            "SELECT a AS x, b AS y FROM t;",
            "SELECT t.a FROM my_table t;",
            "SELECT DISTINCT a FROM t;",

            # Literals and basic expressions
            "SELECT -5, +10.5, 5 % 2, 'foo', TRUE, FALSE, NULL;",
            
            # Basic WHERE clauses and boolean logic
            "SELECT * FROM t WHERE a = 1;",
            "SELECT * FROM t WHERE a > 1 AND b < 'two';",
            "SELECT * FROM t WHERE a = 1 OR b = 2;",
            "SELECT * FROM t WHERE NOT a = 1;",
            "SELECT * FROM t WHERE (a = 1 OR b = 2) AND c = 3;",

            # Specific WHERE operators
            "SELECT * FROM t WHERE a LIKE 'a%';",
            "SELECT * FROM t WHERE a NOT LIKE 'a%';",
            "SELECT * FROM t WHERE a BETWEEN 1 AND 10;",
            "SELECT * FROM t WHERE a NOT BETWEEN 1 AND 10;",
            "SELECT * FROM t WHERE a IS NULL;",
            "SELECT * FROM t WHERE a IS NOT NULL;",
            "SELECT * FROM t WHERE a IN (1, 2, 3);",
            "SELECT * FROM t WHERE a NOT IN ('x', 'y');",

            # JOIN variations
            "SELECT * FROM t1, t2;",
            "SELECT * FROM t1 INNER JOIN t2 ON t1.id = t2.id;",
            "SELECT * FROM t1 LEFT JOIN t2 ON t1.id = t2.id;",
            "SELECT * FROM t1 LEFT OUTER JOIN t2 ON t1.id = t2.id;",
            "SELECT * FROM t1 RIGHT JOIN t2 ON t1.id = t2.id;",
            "SELECT * FROM t1 RIGHT OUTER JOIN t2 ON t1.id = t2.id;",
            "SELECT * FROM t1 FULL JOIN t2 ON t1.id = t2.id;",
            "SELECT * FROM t1 FULL OUTER JOIN t2 ON t1.id = t2.id;",
            "SELECT * FROM t1 CROSS JOIN t2;",
            "SELECT * FROM t1 NATURAL JOIN t2;",
            "SELECT * FROM t1 JOIN t2 USING (id);",

            # Aggregation, Grouping, Ordering, Paging
            "SELECT COUNT(*) FROM t;",
            "SELECT a, SUM(b) FROM t GROUP BY a;",
            "SELECT a, MAX(b) FROM t GROUP BY a HAVING MAX(b) > 100;",
            "SELECT * FROM t ORDER BY a ASC, b DESC;",
            "SELECT * FROM t ORDER BY a NULLS FIRST;",
            "SELECT * FROM t ORDER BY a NULLS LAST;",
            "SELECT * FROM t LIMIT 10;",
            "SELECT * FROM t LIMIT 10 OFFSET 5;",

            # DML statements
            "INSERT INTO t (a, b) VALUES (1, 'one');",
            "INSERT INTO t (a, b) VALUES (1, 'a'), (2, 'b');",
            "INSERT INTO t VALUES (1, 'a', TRUE);",
            "UPDATE t SET a = 1, b = 'two' WHERE c = 3;",
            "DELETE FROM t WHERE a = 1;",
            "DELETE FROM t;",

            # DDL statements
            "CREATE TABLE t (a INT, b VARCHAR(20));",
            "DROP TABLE t;",
            "CREATE INDEX my_index ON t (a);",
            "CREATE UNIQUE INDEX my_unique_index ON t (a, b);",
            "DROP INDEX my_index;",

            # Other statement types
            "EXPLAIN SELECT * FROM t;",

            # --- Complex, Combinatorial Queries ---

            # 1. Comprehensive SELECT with CTE, subqueries, window function
            """
            WITH regional_sales AS (
                SELECT region, SUM(amount) AS total_sales FROM orders GROUP BY region
            )
            SELECT
                p.name, p.category, (SELECT MAX(price) FROM products),
                ROW_NUMBER() OVER (PARTITION BY p.category ORDER BY s.price DESC) as rn
            FROM products AS p JOIN sales s ON p.id = s.product_id
            WHERE
                s.sale_date > '2023-01-01'
                AND p.id IN (SELECT product_id FROM top_products WHERE is_active)
                AND EXISTS (SELECT 1 FROM regional_sales rs WHERE rs.region = p.region AND rs.total_sales > 10000)
            ORDER BY p.category, s.price DESC
            LIMIT 50 OFFSET 10;
            """,

            # 2. Complex expressions, functions, and CASE
            """
            SELECT
                a * (b + c) - d / e, CAST(f AS INT), lower(g),
                CASE h WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END,
                CASE WHEN i > 0 THEN 'positive' ELSE 'non-positive' END
            FROM t WHERE (j, k) IN (SELECT x, y FROM t2);
            """,

            # 3. All set operations combined
            """
            (SELECT a, b FROM t1 UNION ALL SELECT a, b FROM t2)
            EXCEPT
            (SELECT a, b FROM t3 INTERSECT SELECT a, b FROM t4)
            ORDER BY 1, 2 DESC LIMIT 100;
            """,

            # 4. INSERT from a complex SELECT
            "INSERT INTO summary (a, b, c) SELECT c1, COUNT(c2), MAX(c3) FROM source_table GROUP BY c1;",

            # 5. Comprehensive CREATE TABLE
            """
            CREATE TABLE employees (
                id INT PRIMARY KEY,
                first_name VARCHAR(50) NOT NULL,
                last_name VARCHAR(50) NOT NULL,
                email VARCHAR(100) UNIQUE,
                salary DECIMAL(10, 2) DEFAULT 50000.00 CHECK (salary > 0),
                department_id INT,
                CONSTRAINT fk_dept FOREIGN KEY (department_id) REFERENCES departments(id) ON DELETE SET NULL ON UPDATE CASCADE,
                UNIQUE (first_name, last_name)
            );
            """,
            
            # 6. Comprehensive UPDATE with subquery
            "UPDATE t1 SET col1 = (SELECT AVG(col2) FROM t2 WHERE t2.id = t1.id) WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.id = t1.id);",

            # 7. Quoted identifiers and comments
            "/* This is a block comment */ SELECT \"from\" FROM \"my table\" AS \"select\" -- This is a line comment",

            # 8. CREATE TABLE AS SELECT (CTAS)
            "CREATE TABLE new_table AS SELECT * FROM old_table WHERE creation_date > '2023-01-01';",
            
            # 9. Recursive CTE
            "WITH RECURSIVE countdown(n) AS (SELECT 3 UNION ALL SELECT n-1 FROM countdown WHERE n > 1) SELECT * FROM countdown;",
        ]
        
        return queries