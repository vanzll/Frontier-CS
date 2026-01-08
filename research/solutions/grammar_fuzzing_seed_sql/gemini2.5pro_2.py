class Solution:
    def solve(self, resources_path: str) -> list[str]:
        test_cases = [
            # === DDL Statements ===
            # Basic CREATE TABLE with various data types
            "CREATE TABLE t1 (c1 INT, c2 VARCHAR(20), c3 TEXT, c4 FLOAT, c5 BOOLEAN)",
            # CREATE TABLE with various constraints
            "CREATE TABLE products (product_id INT PRIMARY KEY, sku VARCHAR(20) NOT NULL UNIQUE, name TEXT, price FLOAT DEFAULT 0.0, active BOOLEAN, UNIQUE (sku, name))",
            # Basic DROP TABLE
            "DROP TABLE t1",
            # Basic CREATE INDEX
            "CREATE INDEX idx_name ON products (name ASC)",
            # More complex CREATE INDEX
            "CREATE UNIQUE INDEX idx_sku_price ON products (sku DESC, price ASC)",
            # Basic DROP INDEX
            "DROP INDEX idx_name",

            # === DML Statements ===
            # INSERT with all columns specified
            "INSERT INTO products VALUES (1, 'SKU001', 'Apple', 1.50, TRUE)",
            # INSERT with specified columns and multiple rows
            "INSERT INTO products (product_id, sku) VALUES (2, 'SKU002'), (3, 'SKU003')",
            # INSERT from a SELECT statement
            "INSERT INTO products SELECT * FROM old_products",
            # UPDATE with a WHERE clause
            "UPDATE products SET price = price * 1.1, active = FALSE WHERE product_id > 1",
            # UPDATE without a WHERE clause
            "UPDATE products SET name = 'default_name'",
            # DELETE with a WHERE clause
            "DELETE FROM products WHERE active IS FALSE",
            # DELETE without a WHERE clause
            "DELETE FROM products",

            # === SELECT Statements: Clauses ===
            # Basic SELECT with wildcard
            "SELECT * FROM products",
            # SELECT specific columns
            "SELECT name, price FROM products",
            # SELECT with aliases for columns and tables
            "SELECT name AS product_name, price * 2 AS double_price FROM products AS p",
            # SELECT DISTINCT
            "SELECT DISTINCT active FROM products",
            # SELECT with WHERE clause
            "SELECT * FROM products WHERE price > 10.0 AND name = 'Apple'",
            # SELECT with GROUP BY
            "SELECT active, COUNT(*) FROM products GROUP BY active",
            # SELECT with GROUP BY and HAVING
            "SELECT active, COUNT(*) FROM products GROUP BY active HAVING COUNT(*) > 5",
            # SELECT with ORDER BY
            "SELECT * FROM products ORDER BY name",
            # SELECT with ORDER BY multiple columns and directions
            "SELECT * FROM products ORDER BY active DESC, price ASC",
            # SELECT with LIMIT
            "SELECT * FROM products LIMIT 50",
            # SELECT with LIMIT and OFFSET
            "SELECT * FROM products LIMIT 10 OFFSET 20",

            # === SELECT Statements: Joins ===
            "SELECT * FROM t1 JOIN t2 ON t1.id = t2.id",
            "SELECT * FROM t1 INNER JOIN t2 ON t1.id = t2.id",
            "SELECT * FROM t1 LEFT JOIN t2 ON t1.id = t2.id",
            "SELECT * FROM t1 LEFT OUTER JOIN t2 ON t1.id = t2.id",
            "SELECT * FROM t1 RIGHT JOIN t2 ON t1.id = t2.id",
            "SELECT * FROM t1 RIGHT OUTER JOIN t2 ON t1.id = t2.id",
            "SELECT * FROM t1 FULL JOIN t2 ON t1.id = t2.id",
            "SELECT * FROM t1 FULL OUTER JOIN t2 ON t1.id = t2.id",
            "SELECT * FROM t1 CROSS JOIN t2",
            "SELECT * FROM t1 JOIN t2 USING (id, name)",
            "SELECT * FROM t1, t2 WHERE t1.id = t2.id",

            # === SELECT Statements: Expressions & Operators ===
            "SELECT * FROM products WHERE price < 10.0 OR price >= 100.0",
            "SELECT * FROM products WHERE price != 50.0",
            "SELECT * FROM products WHERE price <> 50.0",
            "SELECT * FROM products WHERE NOT active",
            "SELECT * FROM products WHERE name LIKE 'A%'",
            "SELECT * FROM products WHERE name NOT LIKE 'B%'",
            "SELECT * FROM products WHERE product_id IN (1, 2, 3)",
            "SELECT * FROM products WHERE product_id NOT IN (4, 5, 6)",
            "SELECT * FROM products WHERE price BETWEEN 10.0 AND 20.0",
            "SELECT * FROM products WHERE price NOT BETWEEN 30.0 AND 40.0",
            "SELECT * FROM products WHERE name IS NULL",
            "SELECT * FROM products WHERE name IS NOT NULL",
            "SELECT price + 1, price - 1, price * 2, price / 2, -price FROM products",
            "SELECT (price + 5) * 2 FROM products",

            # === SELECT Statements: Subqueries, Functions, CASE ===
            "SELECT * FROM (SELECT name FROM products) AS product_names",
            "SELECT * FROM products WHERE product_id IN (SELECT id FROM user_favorites)",
            "SELECT * FROM products p WHERE EXISTS (SELECT 1 FROM stock s WHERE s.product_id = p.product_id)",
            "SELECT COUNT(*), COUNT(name), SUM(price), AVG(price), MIN(price), MAX(price) FROM products",
            "SELECT LOWER(name), UPPER(name) FROM products",
            "SELECT CASE WHEN price > 20 THEN 'high' WHEN price > 10 THEN 'medium' ELSE 'low' END FROM products",
            "SELECT CASE product_id WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END FROM products",

            # === Syntax & Tokenizer Variations ===
            "SELECT 123, 45.67, 'a string', \"double quoted string\", TRUE, FALSE, NULL",
            "SELECT * FROM \"my table\" WHERE \"my table\".\"my column\" = 1",
            "SELECT p.* FROM products AS p",
            "   SELECT\t*\nFROM\nproducts; -- This is a comment",
            "SELECT * /* This is a block comment */ FROM products",
            "sElEcT 1 FrOm PrOdUcTs",
        ]
        
        cleaned_test_cases = [" ".join(q.split()) for q in test_cases]
        return cleaned_test_cases