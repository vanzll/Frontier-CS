import os
import re
import random
from collections import defaultdict
from typing import List, Dict, Tuple


class GrammarFuzzer:
    def __init__(self, grammar_file_path: str):
        self.rules: Dict[str, List[str]] = {}
        self.alt_usage: Dict[Tuple[str, int], int] = defaultdict(int)
        self.rng = random.Random(0)
        self.MAX_DEPTH = 10
        try:
            self._parse_grammar(grammar_file_path)
        except Exception:
            # Fail gracefully: no rules => fuzzer will be a no-op
            self.rules = {}

    def _parse_grammar(self, path: str) -> None:
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        current_lhs = None
        current_rhs_parts: List[str] = []

        for raw_line in lines:
            line = raw_line.rstrip("\n")

            # Strip comments: #, --, //
            stripped = line
            for comment_prefix in ("#", "--", "//"):
                pos = stripped.find(comment_prefix)
                if pos != -1:
                    stripped = stripped[:pos]
            stripped = stripped.strip()
            if not stripped:
                continue

            parsed = False
            for op_pattern in ("::=", ":=", "->", "="):
                regex = (
                    r"^\s*(?P<lhs><[^>]+>|[A-Za-z_][A-Za-z0-9_]*)\s*"
                    + re.escape(op_pattern)
                    + r"\s*(?P<rhs>.+)$"
                )
                m = re.match(regex, stripped)
                if m:
                    # Save previous rule if any
                    if current_lhs is not None:
                        full_rhs = " ".join(current_rhs_parts).strip()
                        self._add_rule(current_lhs, full_rhs)
                    current_lhs_raw = m.group("lhs")
                    lhs_name = self._normalize_nonterminal(current_lhs_raw)
                    current_lhs = lhs_name
                    current_rhs_parts = [m.group("rhs").strip()]
                    parsed = True
                    break
            if not parsed:
                # Continuation line for previous RHS
                if current_lhs is not None:
                    current_rhs_parts.append(stripped)
                else:
                    # Line that doesn't belong to any rule; ignore
                    continue

        if current_lhs is not None:
            full_rhs = " ".join(current_rhs_parts).strip()
            self._add_rule(current_lhs, full_rhs)

    def _normalize_nonterminal(self, text: str) -> str:
        t = text.strip()
        if t.startswith("<") and t.endswith(">"):
            t = t[1:-1].strip()
        return t

    def _add_rule(self, lhs: str, rhs_text: str) -> None:
        alts = self._split_alternatives(rhs_text)
        if not alts:
            return
        lst = self.rules.setdefault(lhs, [])
        for alt in alts:
            alt = alt.strip()
            if alt:
                lst.append(alt)

    def _split_alternatives(self, text: str) -> List[str]:
        alts: List[str] = []
        buf: List[str] = []
        in_quote: str | None = None
        angle_depth = 0
        square_depth = 0
        curly_depth = 0
        paren_depth = 0

        for c in text:
            if in_quote is not None:
                buf.append(c)
                if c == in_quote:
                    in_quote = None
                continue

            if c == "'" or c == '"':
                in_quote = c
                buf.append(c)
                continue

            if c == "<":
                angle_depth += 1
                buf.append(c)
                continue
            if c == ">":
                if angle_depth > 0:
                    angle_depth -= 1
                buf.append(c)
                continue
            if c == "[":
                square_depth += 1
                buf.append(c)
                continue
            if c == "]":
                if square_depth > 0:
                    square_depth -= 1
                buf.append(c)
                continue
            if c == "{":
                curly_depth += 1
                buf.append(c)
                continue
            if c == "}":
                if curly_depth > 0:
                    curly_depth -= 1
                buf.append(c)
                continue
            if c == "(":
                paren_depth += 1
                buf.append(c)
                continue
            if c == ")":
                if paren_depth > 0:
                    paren_depth -= 1
                buf.append(c)
                continue

            if (
                c == "|"
                and in_quote is None
                and angle_depth == 0
                and square_depth == 0
                and curly_depth == 0
                and paren_depth == 0
            ):
                alt_text = "".join(buf).strip()
                if alt_text:
                    alts.append(alt_text)
                buf = []
            else:
                buf.append(c)

        final = "".join(buf).strip()
        if final:
            alts.append(final)
        return alts

    def _choose_start_symbol(self) -> str | None:
        if not self.rules:
            return None

        priority_buckets: Dict[int, List[str]] = {0: [], 1: [], 2: [], 3: []}
        for name in self.rules.keys():
            lower = name.lower()
            if any(
                sub in lower
                for sub in (
                    "sql",
                    "statement_list",
                    "statements",
                    "stmt_list",
                    "program",
                    "input",
                    "script",
                )
            ):
                priority_buckets[0].append(name)
            elif any(
                sub in lower
                for sub in (
                    "statement",
                    "query",
                    "select_stmt",
                    "select_statement",
                    "query_specification",
                    "root",
                )
            ):
                priority_buckets[1].append(name)
            elif any(
                sub in lower
                for sub in ("select", "insert", "update", "delete", "create", "drop")
            ):
                priority_buckets[2].append(name)
            else:
                priority_buckets[3].append(name)

        for pr in (0, 1, 2, 3):
            if priority_buckets[pr]:
                # Deterministic choice
                return sorted(priority_buckets[pr])[0]

        for name in self.rules.keys():
            return name
        return None

    def generate(self, num_statements: int) -> List[str]:
        if not self.rules or num_statements <= 0:
            return []
        start = self._choose_start_symbol()
        if start is None:
            return []

        results: List[str] = []
        attempts = 0
        max_attempts = num_statements * 3
        while len(results) < num_statements and attempts < max_attempts:
            attempts += 1
            tokens = self._expand_nonterminal(start, depth=0)
            if not tokens:
                continue
            stmt = self._postprocess(" ".join(tokens).strip())
            if stmt and stmt not in results:
                results.append(stmt)
        return results

    def _expand_nonterminal(self, name: str, depth: int) -> List[str]:
        if depth > self.MAX_DEPTH:
            return self._fallback_for_nonterminal(name)
        if name not in self.rules:
            return self._fallback_for_nonterminal(name)
        alts = self.rules[name]
        if not alts:
            return self._fallback_for_nonterminal(name)

        best_usage = None
        candidate_indices: List[int] = []
        for idx in range(len(alts)):
            u = self.alt_usage[(name, idx)]
            if best_usage is None or u < best_usage:
                best_usage = u
                candidate_indices = [idx]
            elif u == best_usage:
                candidate_indices.append(idx)
        idx = self.rng.choice(candidate_indices)
        self.alt_usage[(name, idx)] += 1
        rhs_text = alts[idx]
        tokens = self._expand_rhs(rhs_text, depth + 1)
        return tokens

    def _expand_rhs(self, text: str, depth: int) -> List[str]:
        tokens: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            c = text[i]
            if c.isspace():
                i += 1
                continue

            if c == "'" or c == '"':
                quote = c
                i += 1
                sb: List[str] = []
                while i < n:
                    ch = text[i]
                    if ch == "\\" and i + 1 < n:
                        sb.append(text[i + 1])
                        i += 2
                        continue
                    if ch == quote:
                        i += 1
                        break
                    sb.append(ch)
                    i += 1
                literal = "".join(sb)
                if literal:
                    tokens.append(self._map_terminal_literal(literal))
                continue

            if c == "[" or c == "{":
                open_char = c
                close_char = "]" if c == "[" else "}"
                j = self._find_matching(text, i, open_char, close_char)
                if j == -1:
                    i += 1
                    continue
                inner = text[i + 1 : j].strip()
                i = j + 1
                if not inner:
                    continue
                alt_inner = self._choose_group_alternative(inner)
                repeat_count = 1
                if open_char == "{":
                    # Repeat lists 1-2 times to exercise repetition constructs
                    repeat_count = 1 + self.rng.randint(0, 1)
                for _ in range(repeat_count):
                    tokens.extend(self._expand_rhs(alt_inner, depth))
                continue

            if c == "<":
                j = text.find(">", i + 1)
                if j == -1:
                    word, new_i = self._read_word(text, i)
                    if word:
                        tokens.append(self._map_terminal_token(word))
                    i = new_i
                    continue
                name = text[i + 1 : j].strip()
                i = j + 1
                expanded = self._expand_nonterminal(name, depth + 1)
                if expanded:
                    tokens.extend(expanded)
                else:
                    tokens.extend(self._fallback_for_nonterminal(name))
                continue

            word, new_i = self._read_word(text, i)
            if word:
                tokens.append(self._map_terminal_token(word))
            i = new_i

        return tokens

    def _read_word(self, text: str, i_start: int) -> Tuple[str, int]:
        i = i_start
        n = len(text)
        buf: List[str] = []
        while i < n:
            c = text[i]
            if c.isspace() or c in ("'", '"', "[", "]", "{", "}", "<", ">", "|"):
                break
            buf.append(c)
            i += 1
        return "".join(buf), i

    def _find_matching(self, text: str, start_idx: int, open_c: str, close_c: str) -> int:
        depth = 0
        n = len(text)
        for i in range(start_idx, n):
            c = text[i]
            if c == open_c:
                depth += 1
            elif c == close_c:
                depth -= 1
                if depth == 0:
                    return i
        return -1

    def _choose_group_alternative(self, inner_text: str) -> str:
        alts = self._split_alternatives(inner_text)
        if not alts:
            return inner_text
        return self.rng.choice(alts)

    def _map_terminal_literal(self, literal: str) -> str:
        # Distinguish between keyword-like and string-like literals
        if re.fullmatch(r"[A-Z_]+", literal):
            return literal
        # Treat as SQL string literal
        lit = literal.replace("'", "''")
        return f"'{lit}'"

    def _map_terminal_token(self, token: str) -> str:
        return token

    def _fallback_for_nonterminal(self, name: str) -> List[str]:
        n = name.strip().lower()
        base = re.sub(r"\W+", "", n) or "x"
        if "table" in n:
            return [f"t_{base}"]
        if "column" in n or "col" in n:
            return [f"c_{base}"]
        if "name" in n or "ident" in n or n == "id":
            return ["x1"]
        if "string" in n or "char" in n or "text" in n:
            return ["'str'"]
        if "int" in n or "integer" in n or "number" in n or "numeric" in n or "digit" in n:
            return ["123"]
        if "date" in n or "time" in n:
            return ["'2020-01-01'"]
        if "bool" in n or "boolean" in n:
            return ["TRUE"]
        if "expr" in n or "expression" in n:
            return ["1 + 2"]
        if "condition" in n or "predicate" in n or "where" in n:
            return ["1 = 1"]
        if "select" in n or "query" in n:
            return ["SELECT 1"]
        return ["x"]

    def _postprocess(self, s: str) -> str:
        s = re.sub(r"\s+", " ", s).strip()
        if s and not s.endswith(";"):
            s = s + ";"
        return s


class Solution:
    def solve(self, resources_path: str) -> List[str]:
        tests: List[str] = []

        # Static, hand-crafted SQL statements to exercise common constructs
        static_tests = self._build_static_queries()

        grammar_tests: List[str] = []
        grammar_path = os.path.join(resources_path, "sql_grammar.txt")
        try:
            fuzzer = GrammarFuzzer(grammar_path)
            if fuzzer.rules:
                total_alts = sum(len(alts) for alts in fuzzer.rules.values())
                num_grammar_stmts = min(max(20, total_alts * 2), 80)
                grammar_tests = fuzzer.generate(num_grammar_stmts)
        except Exception:
            grammar_tests = []

        seen: set[str] = set()
        for stmt in static_tests + grammar_tests:
            stmt_norm = stmt.strip()
            if not stmt_norm:
                continue
            if stmt_norm not in seen:
                seen.add(stmt_norm)
                tests.append(stmt_norm)
        return tests

    def _build_static_queries(self) -> List[str]:
        # A diverse set of reasonably standard SQL statements
        queries: List[str] = [
            "SELECT 1;",
            "SELECT 1 AS one;",
            "SELECT * FROM employees;",
            (
                "SELECT e.id, e.name, d.name AS dept_name "
                "FROM employees e INNER JOIN departments d ON e.dept_id = d.id "
                "WHERE e.salary > 50000 AND d.active = 1 "
                "ORDER BY e.name ASC;"
            ),
            (
                "SELECT department, COUNT(*) AS cnt, AVG(salary) AS avg_sal "
                "FROM employees "
                "GROUP BY department "
                "HAVING COUNT(*) > 5 "
                "ORDER BY avg_sal DESC;"
            ),
            (
                "SELECT DISTINCT status "
                "FROM orders "
                "WHERE order_date BETWEEN '2020-01-01' AND '2020-12-31' "
                "AND amount >= 100.0 "
                "ORDER BY status;"
            ),
            (
                "SELECT c.name, "
                "(SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.id) AS order_count "
                "FROM customers c "
                "WHERE EXISTS (SELECT 1 FROM orders o2 WHERE o2.customer_id = c.id AND o2.status = 'OPEN');"
            ),
            (
                "SELECT p.id, p.name, "
                "CASE "
                "WHEN p.price < 10 THEN 'cheap' "
                "WHEN p.price BETWEEN 10 AND 100 THEN 'mid' "
                "ELSE 'expensive' "
                "END AS price_bucket "
                "FROM products p;"
            ),
            (
                "INSERT INTO employees (id, name, department, salary) "
                "VALUES (1, 'Alice', 'IT', 75000), "
                "(2, 'Bob', 'HR', 50000);"
            ),
            (
                "UPDATE employees "
                "SET salary = salary * 1.1, updated_at = CURRENT_TIMESTAMP "
                "WHERE department = 'IT' AND salary < 80000;"
            ),
            (
                "DELETE FROM employees "
                "WHERE terminated = 1 OR last_day < '2020-01-01';"
            ),
            (
                "CREATE TABLE employees ("
                "id INTEGER PRIMARY KEY, "
                "name VARCHAR(100) NOT NULL, "
                "department VARCHAR(50), "
                "salary DECIMAL(10,2) DEFAULT 0, "
                "manager_id INTEGER, "
                "CONSTRAINT fk_manager FOREIGN KEY (manager_id) REFERENCES employees(id)"
                ");"
            ),
            (
                "CREATE INDEX idx_employees_dept_salary "
                "ON employees (department, salary DESC);"
            ),
            "DROP INDEX idx_employees_dept_salary;",
            (
                "CREATE VIEW active_employees AS "
                "SELECT id, name, department FROM employees WHERE terminated = 0;"
            ),
            "DROP VIEW active_employees;",
            (
                "CREATE TABLE departments ("
                "id INTEGER PRIMARY KEY, "
                "name VARCHAR(100) UNIQUE, "
                "active INTEGER DEFAULT 1"
                ");"
            ),
            "ALTER TABLE employees ADD COLUMN hire_date DATE;",
            "ALTER TABLE employees DROP COLUMN hire_date;",
            "DROP TABLE departments;",
            (
                "SELECT o.id, o.amount, c.name "
                "FROM orders o "
                "LEFT OUTER JOIN customers c ON o.customer_id = c.id "
                "WHERE o.status IN ('OPEN', 'PENDING') "
                "ORDER BY o.order_date DESC, o.id;"
            ),
            (
                "SELECT p.category, SUM(p.price) AS total_price "
                "FROM products p "
                "WHERE p.discontinued = 0 "
                "GROUP BY p.category "
                "HAVING SUM(p.price) > 1000 "
                "ORDER BY total_price DESC "
                "LIMIT 10 OFFSET 5;"
            ),
            "SELECT 1 + 2 * 3 AS result, -4 AS negative, (5 + 6) / 7 AS fraction;",
            (
                "SELECT name FROM employees "
                "WHERE salary IS NULL OR (salary IS NOT NULL AND name LIKE 'A%');"
            ),
            (
                "SELECT e.name FROM employees e "
                "WHERE e.salary > ALL (SELECT salary FROM employees WHERE department = 'HR');"
            ),
        ]
        return queries