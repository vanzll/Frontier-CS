import os
import sys
import importlib
import importlib.util


class Solution:
    def solve(self, resources_path: str) -> list[str]:
        # Read grammar text if available (not strictly needed but kept for potential extensions)
        grammar_text = ""
        try:
            grammar_file = os.path.join(resources_path, "sql_grammar.txt")
            if os.path.exists(grammar_file):
                with open(grammar_file, "r", encoding="utf-8") as f:
                    grammar_text = f.read()
        except Exception:
            grammar_text = ""

        parse_sql = None

        # Robustly import sql_engine from the specified resources path
        try:
            pkg_dir = os.path.join(resources_path, "sql_engine")
            init_path = os.path.join(pkg_dir, "__init__.py")
            if os.path.isdir(pkg_dir) and os.path.exists(init_path):
                spec = importlib.util.spec_from_file_location("sql_engine", init_path)
                engine = importlib.util.module_from_spec(spec)
                sys.modules["sql_engine"] = engine
                if spec.loader is not None:
                    spec.loader.exec_module(engine)  # type: ignore
                else:
                    raise ImportError("No loader for sql_engine")
            else:
                if resources_path not in sys.path:
                    sys.path.insert(0, resources_path)
                engine = importlib.import_module("sql_engine")

            parse_sql = getattr(engine, "parse_sql", None)
            if parse_sql is None:
                parser_mod = importlib.import_module("sql_engine.parser")
                parse_sql = getattr(parser_mod, "parse_sql", None)
        except Exception:
            parse_sql = None

        candidates = self._build_candidates(grammar_text)

        # If we can't import/locate parse_sql, just return a reasonably-sized unique subset
        if parse_sql is None:
            unique_sqls = []
            seen = set()
            for sql in candidates:
                if sql not in seen:
                    seen.add(sql)
                    unique_sqls.append(sql)
                if len(unique_sqls) >= 80:
                    break
            return unique_sqls

        # Use parse_sql to filter invalid statements and to approximate coverage via AST node types
        parsed_entries = []
        for sql in candidates:
            try:
                ast = parse_sql(sql)
            except Exception:
                continue

            visited_types = set()
            seen_ids = set()

            def visit(obj):
                if obj is None:
                    return
                oid = id(obj)
                if oid in seen_ids:
                    return
                seen_ids.add(oid)

                if isinstance(obj, (str, int, float, bool, bytes, complex)):
                    return
                if isinstance(obj, (list, tuple, set, frozenset)):
                    for x in obj:
                        visit(x)
                    return
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        visit(k)
                        visit(v)
                    return

                t = type(obj)
                if t.__module__ == "builtins":
                    return
                visited_types.add(t)

                if hasattr(obj, "__dict__"):
                    for v in obj.__dict__.values():
                        visit(v)
                else:
                    for attr_name in dir(obj):
                        if attr_name.startswith("_"):
                            continue
                        try:
                            v = getattr(obj, attr_name)
                        except Exception:
                            continue
                        if callable(v):
                            continue
                        visit(v)

            visit(ast)
            parsed_entries.append({"sql": sql, "types": visited_types})

        # If nothing parsed successfully, fall back to a subset of raw candidates
        if not parsed_entries:
            unique_sqls = []
            seen = set()
            for sql in candidates:
                if sql not in seen:
                    seen.add(sql)
                    unique_sqls.append(sql)
                if len(unique_sqls) >= 50:
                    break
            return unique_sqls

        # Compute all AST node types observed
        all_types = set()
        for entry in parsed_entries:
            all_types.update(entry["types"])

        max_tests = 80

        # If we can't distinguish by AST types, just return valid SQLs up to limit
        if not all_types:
            result_sqls = []
            seen = set()
            for entry in parsed_entries:
                sql = entry["sql"]
                if sql not in seen:
                    seen.add(sql)
                    result_sqls.append(sql)
                if len(result_sqls) >= max_tests:
                    break
            return result_sqls

        # Greedy set cover to select minimal-ish subset covering all observed AST node types
        selected_sqls = []
        used = [False] * len(parsed_entries)
        uncovered = set(all_types)

        while uncovered and len(selected_sqls) < max_tests:
            best_idx = None
            best_gain = 0
            for i, entry in enumerate(parsed_entries):
                if used[i]:
                    continue
                gain = len(entry["types"] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx is None or best_gain == 0:
                break
            used[best_idx] = True
            selected_sqls.append(parsed_entries[best_idx]["sql"])
            uncovered -= parsed_entries[best_idx]["types"]

        # Include additional statements that are likely to exercise tokenizer/parser branches
        important_tokens = [
            " UNION ",
            " INTERSECT ",
            " EXCEPT ",
            " LEFT JOIN ",
            " RIGHT JOIN ",
            " FULL OUTER JOIN ",
            " CROSS JOIN ",
            " NATURAL JOIN ",
            " CASE ",
            " EXISTS ",
            " IN (",
            " BETWEEN ",
            " LIKE ",
            " IS NULL",
            " ORDER BY",
            " GROUP BY",
            " HAVING",
            " LIMIT",
            " OFFSET",
            " INSERT ",
            " UPDATE ",
            " DELETE ",
            " CREATE ",
            " DROP ",
            " ALTER ",
            " INDEX ",
            " BEGIN",
            " COMMIT",
            " ROLLBACK",
            "/*",
            "--",
            " WITH ",
            " OVER (",
            " ROW_NUMBER",
            " RANK(",
            " DISTINCT "
        ]

        selected_set = set(selected_sqls)

        for entry in parsed_entries:
            if len(selected_sqls) >= max_tests:
                break
            sql = entry["sql"]
            if sql in selected_set:
                continue
            upper_sql = " " + sql.upper() + " "
            if any(tok in upper_sql for tok in important_tokens):
                selected_sqls.append(sql)
                selected_set.add(sql)

        # If we still have room, fill with remaining valid statements
        for entry in parsed_entries:
            if len(selected_sqls) >= max_tests:
                break
            sql = entry["sql"]
            if sql not in selected_set:
                selected_sqls.append(sql)
                selected_set.add(sql)

        return selected_sqls

    def _build_candidates(self, grammar_text: str) -> list[str]:
        candidates: list[str] = []

        def add(sql: str):
            s = sql.strip()
            if not s.endswith(";"):
                s = s + ";"
            candidates.append(s)

        # Basic expressions and literals
        add("SELECT 1")
        add("SELECT 1 + 2 * 3 AS result")
        add("SELECT -1 AS neg, 2.5 AS float_value, 'hello' AS greeting")
        add("SELECT 'O''Reilly' AS escaped_string")
        add("SELECT NULL AS nothing, TRUE AS t, FALSE AS f")
        add("SELECT 1 AS one, 2 AS two, 3 AS three")
        add("SELECT (1 + 2) * (3 - 4) / 5.0 AS complex_expr")
        add("SELECT 1e10 AS big, 3.14e-2 AS small, .5 AS half")
        add("SELECT 0 AS zero, -0 AS neg_zero, +0 AS pos_zero")
        add("SELECT 1 / 2 AS half, 5 % 2 AS mod_result")

        # Simple table selects and WHERE conditions
        add("SELECT id, name FROM users")
        add("SELECT DISTINCT status FROM orders")
        add("SELECT * FROM users WHERE age >= 18 AND status = 'active'")
        add("SELECT * FROM users WHERE NOT (age < 18 OR status <> 'active')")
        add("SELECT name FROM products WHERE price BETWEEN 10 AND 100")
        add("SELECT name FROM products WHERE price NOT BETWEEN 10 AND 100")
        add("SELECT name FROM products WHERE name LIKE 'A%' OR name LIKE '%B'")
        add("SELECT name FROM users WHERE age IS NULL OR age IS NOT NULL")
        add("SELECT name FROM users WHERE status IN ('active', 'pending')")
        add("SELECT name FROM users WHERE status NOT IN ('active', 'pending')")
        add("SELECT name FROM users WHERE name LIKE '%test\\_user%' ESCAPE '\\\\'")
        add("SELECT id, name FROM users WHERE age + 10 > 30 * 2 / 3 - 1")

        # Aggregates and scalar functions
        add("SELECT COUNT(*) AS cnt FROM users")
        add("SELECT COUNT(DISTINCT status) AS distinct_status_count FROM users")
        add("SELECT SUM(price * quantity) AS total_revenue FROM orders")
        add("SELECT MAX(age) AS max_age, MIN(age) AS min_age, AVG(age) AS avg_age FROM users")
        add("SELECT COALESCE(name, 'unknown') AS name, LENGTH(name) AS name_len FROM users")
        add("SELECT UPPER(name) AS name_upper, LOWER(name) AS name_lower FROM users")
        add("SELECT ABS(age) AS abs_age FROM users")
        add("SELECT ROUND(price, 2) AS rounded_price FROM products")

        # Joins, GROUP BY, HAVING, ORDER BY
        add("SELECT u.id, u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id")
        add(
            "SELECT u.id, COUNT(*) AS order_count "
            "FROM users u LEFT JOIN orders o ON u.id = o.user_id "
            "GROUP BY u.id HAVING COUNT(*) > 1"
        )
        add(
            "SELECT u.id, COUNT(o.id) AS order_count "
            "FROM users u RIGHT JOIN orders o ON u.id = o.user_id "
            "GROUP BY u.id"
        )
        add(
            "SELECT u.id, o.id "
            "FROM users u FULL OUTER JOIN orders o ON u.id = o.user_id"
        )
        add("SELECT * FROM users CROSS JOIN products")
        add("SELECT * FROM users NATURAL JOIN orders")
        add(
            "SELECT u.name, o.total "
            "FROM users AS u INNER JOIN orders AS o "
            "ON (u.id = o.user_id AND o.total > 0)"
        )
        add(
            "SELECT u.id, u.name, SUM(o.total) AS total, COUNT(DISTINCT p.id) AS product_count "
            "FROM users u "
            "LEFT JOIN orders o ON u.id = o.user_id "
            "LEFT JOIN products p ON p.id = o.product_id "
            "WHERE u.status = 'active' "
            "GROUP BY u.id, u.name "
            "HAVING SUM(o.total) > 100 "
            "ORDER BY total DESC"
        )
        add("SELECT id, name FROM users ORDER BY name ASC, id DESC")
        add("SELECT * FROM orders ORDER BY total DESC LIMIT 10")
        add("SELECT * FROM orders LIMIT 10 OFFSET 20")

        # Subqueries and set operations
        add("SELECT name FROM users WHERE id IN (SELECT user_id FROM orders)")
        add(
            "SELECT name FROM users WHERE EXISTS ("
            "SELECT 1 FROM orders WHERE orders.user_id = users.id)"
        )
        add("SELECT name FROM users WHERE age = (SELECT MAX(age) FROM users)")
        add("SELECT * FROM (SELECT id, name FROM users) AS sub WHERE id > 10")
        add("SELECT id FROM users UNION SELECT user_id AS id FROM orders")
        add("SELECT id FROM users UNION ALL SELECT user_id AS id FROM orders")
        add("SELECT id FROM users INTERSECT SELECT user_id FROM orders")
        add("SELECT id FROM users EXCEPT SELECT user_id FROM orders")
        add(
            "SELECT id, (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id) AS order_count "
            "FROM users u"
        )
        add(
            "SELECT (SELECT MAX(age) FROM users) AS max_age, "
            "(SELECT MIN(age) FROM users) AS min_age"
        )

        # CTEs and window functions (may be skipped by parser if unsupported)
        add(
            "WITH recent_orders AS ("
            "SELECT * FROM orders WHERE created_at > '2020-01-01'"
            ") SELECT * FROM recent_orders"
        )
        add(
            "WITH totals AS ("
            "SELECT user_id, SUM(total) AS total FROM orders GROUP BY user_id"
            ") SELECT * FROM totals WHERE total > 100"
        )
        add(
            "WITH t1 AS (SELECT id FROM users), "
            "t2 AS (SELECT id FROM orders) "
            "SELECT * FROM t1 UNION ALL SELECT * FROM t2"
        )
        add(
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at) AS rn "
            "FROM orders"
        )
        add(
            "SELECT id, "
            "SUM(total) OVER (PARTITION BY user_id ORDER BY created_at "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total "
            "FROM orders"
        )
        add(
            "SELECT id, total, RANK() OVER (ORDER BY total DESC) AS rnk "
            "FROM orders"
        )

        # CASE expressions and complex booleans
        add(
            "SELECT CASE "
            "WHEN age < 18 THEN 'minor' "
            "WHEN age < 65 THEN 'adult' "
            "ELSE 'senior' "
            "END AS age_group "
            "FROM users"
        )
        add(
            "SELECT id, CASE status "
            "WHEN 'active' THEN 1 "
            "WHEN 'inactive' THEN 0 "
            "ELSE NULL END AS status_flag "
            "FROM users"
        )
        add(
            "SELECT id, name FROM users WHERE "
            "CASE WHEN age IS NULL THEN 0 "
            "WHEN age < 18 THEN 0 "
            "ELSE 1 END = 1"
        )
        add(
            "SELECT id, name FROM users WHERE "
            "(age < 18 AND status = 'student') OR "
            "(age >= 18 AND status <> 'student')"
        )

        # DML: INSERT, UPDATE, DELETE
        add(
            "INSERT INTO users (id, name, age, status) "
            "VALUES (1, 'Alice', 30, 'active')"
        )
        add("INSERT INTO users VALUES (2, 'Bob', 25, 'inactive')")
        add(
            "INSERT INTO users (id, name, age, status) VALUES "
            "(3, 'Carol', NULL, 'pending'), "
            "(4, 'Dave', 40, 'active')"
        )
        add(
            "INSERT INTO orders (id, user_id, total) VALUES "
            "(1, 1, 100.0), "
            "(2, 1, 50.0)"
        )
        add(
            "INSERT INTO orders (id, user_id, total) "
            "SELECT id, user_id, total FROM old_orders"
        )
        add("UPDATE users SET name = 'Charlie' WHERE id = 1")
        add("UPDATE users SET age = age + 1, status = 'updated'")
        add(
            "UPDATE orders SET total = total * 1.1 "
            "WHERE status = 'open' AND total > 100"
        )
        add("DELETE FROM users WHERE id = 2")
        add("DELETE FROM users")
        add(
            "DELETE FROM orders WHERE id IN ("
            "SELECT id FROM orders WHERE total = 0)"
        )

        # DDL: CREATE TABLE, ALTER TABLE, DROP, INDEX
        add(
            "CREATE TABLE users ("
            "id INT PRIMARY KEY, "
            "name VARCHAR(100) NOT NULL, "
            "age INT, "
            "status VARCHAR(20) DEFAULT 'active'"
            ")"
        )
        add(
            "CREATE TABLE orders ("
            "id INT PRIMARY KEY, "
            "user_id INT NOT NULL, "
            "total DECIMAL(10,2), "
            "created_at TIMESTAMP"
            ")"
        )
        add(
            "CREATE TABLE products ("
            "id INT, "
            "name TEXT, "
            "price REAL, "
            "CONSTRAINT pk_products PRIMARY KEY (id)"
            ")"
        )
        add("ALTER TABLE users ADD COLUMN email VARCHAR(255)")
        add("ALTER TABLE users DROP COLUMN status")
        add("ALTER TABLE users RENAME TO customers")
        add("CREATE INDEX idx_users_name ON users (name)")
        add("CREATE UNIQUE INDEX idx_orders_user_id ON orders (user_id)")
        add("DROP INDEX idx_users_name")
        add("DROP TABLE users")
        add("DROP TABLE IF EXISTS orders")
        add("TRUNCATE TABLE products")

        # Transactions and comments
        add("BEGIN TRANSACTION")
        add("COMMIT")
        add("ROLLBACK")
        add("SELECT id -- inline comment\nFROM users")
        add("SELECT /* block comment */ id FROM users")

        return candidates