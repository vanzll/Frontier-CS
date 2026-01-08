class Solution:
    def solve(self, resources_path: str) -> list[str]:
        """
        Return SQL test cases designed to maximize parser coverage.
        
        Args:
            resources_path: Path to the resources directory.
        
        Returns:
            list[str]: List of SQL statement strings.
        """
        
        # This list of SQL queries is manually crafted to cover a wide range of
        # SQL syntax and parser code paths. The goal is to maximize line and
        # branch coverage with a minimal set of effective test cases, which is
        # rewarded by the scoring formula. The queries cover different
        # statement types (DML, DDL, TCL, DCL), complex expressions, all
        # join types, subqueries, set operations, and tokenizer edge cases.
        test_cases = [
            # 1. Basic statements and keywords
            "SELECT 1;",
            "SELECT * FROM users;",
            "SELECT DISTINCT country_code FROM users;",

            # 2. DML: INSERT variations
            "INSERT INTO products (name, price) VALUES ('Pen', 1.50);",
            "INSERT INTO products VALUES ('Pencil', 0.50), ('Eraser', 0.75);",
            "INSERT INTO archive_products SELECT * FROM products WHERE is_discontinued = TRUE;",

            # 3. DML: UPDATE variations
            "UPDATE employees SET salary = salary * 1.1, bonus = 500 WHERE department_id = 1;",
            "UPDATE products SET price = price * 0.9 WHERE category_id IN (SELECT id FROM categories WHERE name = 'clearance');",

            # 4. DML: DELETE statement
            "DELETE FROM logs WHERE timestamp < '2020-01-01';",

            # 5. DDL: CREATE statements with various constraints
            "CREATE TABLE employees (id INT PRIMARY KEY, first_name VARCHAR(50) NOT NULL, email VARCHAR(100) UNIQUE, hire_date DATE DEFAULT CURRENT_DATE, department_id INT REFERENCES departments(id) ON DELETE SET NULL, salary DECIMAL(10, 2), CONSTRAINT positive_salary CHECK (salary > 0));",
            "CREATE VIEW high_salary_employees AS SELECT id, first_name, last_name FROM employees WHERE salary > 100000;",
            "CREATE UNIQUE INDEX idx_employee_email ON employees (email);",
            
            # 6. DDL: ALTER statements
            "ALTER TABLE employees ADD COLUMN middle_name VARCHAR(50);",
            "ALTER TABLE employees DROP COLUMN middle_name;",
            "ALTER TABLE employees RENAME TO staff;",

            # 7. DDL: DROP statements for different object types
            "DROP TABLE IF EXISTS old_staff;",
            "DROP VIEW IF EXISTS old_high_salary_employees;",
            "DROP INDEX IF EXISTS idx_employee_email;",

            # 8. DDL: TRUNCATE statement
            "TRUNCATE TABLE staging_data;",

            # 9. Comprehensive "Kitchen Sink" SELECT to hit many clauses at once
            "SELECT d.name AS department_name, COUNT(s.id) AS num_staff, AVG(s.salary) AS avg_salary, CASE WHEN AVG(s.salary) > 80000 THEN 'High Paying' ELSE 'Standard' END AS salary_grade FROM departments AS d INNER JOIN staff AS s ON d.id = s.department_id WHERE s.hire_date >= '2021-01-01' AND d.name LIKE 'Eng%' GROUP BY d.name HAVING COUNT(s.id) > 5 ORDER BY num_staff DESC, avg_salary DESC LIMIT 10 OFFSET 5;",

            # 10. All JOIN types in one query
            "SELECT * FROM t1 LEFT JOIN t2 ON t1.id = t2.id RIGHT OUTER JOIN t3 USING (common_col) FULL JOIN t4 ON t1.key = t4.key CROSS JOIN t5;",

            # 11. Complex expressions, functions, and subqueries
            "SELECT -c1, (c1 + c2) * -c3, 'Name: ' || name, CAST(id AS VARCHAR), SUBSTRING(name FROM 1 FOR 3), c1 BETWEEN 10 AND 20, c2 NOT IN (1, 2, 3), c3 IS NOT NULL, EXISTS (SELECT 1 FROM other WHERE other.id = t.id), (SELECT MAX(price) FROM products) AS max_price FROM some_table AS t;",
            
            # 12. Set operations
            "(SELECT id, name FROM table_a) UNION (SELECT id, name FROM table_b) UNION ALL (SELECT id, name FROM table_c) INTERSECT (SELECT id, name FROM table_d) EXCEPT (SELECT id, name FROM table_e);",

            # 13. Transaction Control Language (TCL)
            "BEGIN TRANSACTION;",
            "COMMIT;",
            "ROLLBACK;",

            # 14. Data Control Language (DCL)
            "GRANT SELECT, INSERT ON my_table TO my_user, PUBLIC;",
            "REVOKE UPDATE, DELETE ON my_table FROM an_old_role;",
            
            # 15. Tokenizer edge cases (still valid SQL)
            "SELECT \"col-with-hyphen\", 'a string with an '' escaped quote', 3.14159, 1.2e-3, .5 FROM \"table-with-hyphen\";",
        ]

        # Normalize multiline strings and remove extra whitespace to ensure clean parsing
        normalized_cases = [" ".join(q.strip().split()) for q in test_cases]
        
        return normalized_cases