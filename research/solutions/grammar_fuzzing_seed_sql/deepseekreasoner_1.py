import os
import sys
import random
import re
import itertools
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional, Any

class Solution:
    def solve(self, resources_path: str) -> List[str]:
        # Read and parse the grammar
        grammar_path = os.path.join(resources_path, "sql_grammar.txt")
        with open(grammar_path, 'r') as f:
            grammar_lines = f.readlines()
        
        # Parse BNF grammar
        grammar_rules = self._parse_grammar(grammar_lines)
        
        # Read parser source to understand coverage targets
        parser_path = os.path.join(resources_path, "sql_engine", "parser.py")
        tokenizer_path = os.path.join(resources_path, "sql_engine", "tokenizer.py")
        ast_path = os.path.join(resources_path, "sql_engine", "ast_nodes.py")
        
        # Analyze parser structure to identify important constructs
        parser_constructs = self._analyze_parser(parser_path, tokenizer_path, ast_path)
        
        # Generate diverse SQL statements
        statements = self._generate_statements(grammar_rules, parser_constructs)
        
        return statements
    
    def _parse_grammar(self, lines: List[str]) -> Dict[str, List[List[str]]]:
        """Parse BNF grammar into a dictionary of rules."""
        rules = {}
        current_rule = None
        current_productions = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Check for rule definition: <rule> ::= production
            match = re.match(r'<(\w+)>\s*::=\s*(.+)', line)
            if match:
                if current_rule:
                    rules[current_rule] = current_productions
                current_rule = match.group(1)
                current_productions = [self._parse_production(match.group(2))]
            elif '|' in line:
                # Alternative production for current rule
                current_productions.extend(self._parse_production(line.strip()))
            elif current_rule and line:
                # Continuation line
                current_productions.extend(self._parse_production(line))
        
        if current_rule:
            rules[current_rule] = current_productions
        
        return rules
    
    def _parse_production(self, prod_str: str) -> List[List[str]]:
        """Parse a production string into list of symbols."""
        productions = []
        for alt in prod_str.split('|'):
            symbols = []
            tokens = re.findall(r'<\w+>|"[^"]+"|\'[^\']+\'|\w+|.', alt.strip())
            for token in tokens:
                if token.startswith('<') and token.endswith('>'):
                    symbols.append(token[1:-1])  # Remove <>
                elif token and not token.isspace():
                    symbols.append(token.strip('"\''))
            if symbols:
                productions.append(symbols)
        return productions
    
    def _analyze_parser(self, parser_path: str, tokenizer_path: str, ast_path: str) -> Dict[str, Any]:
        """Analyze parser source code to identify important constructs."""
        constructs = {
            'statement_types': set(),
            'join_types': set(),
            'functions': set(),
            'clauses': set(),
            'operators': set(),
            'data_types': set()
        }
        
        # Read parser file
        with open(parser_path, 'r') as f:
            parser_content = f.read()
        
        # Identify statement types from parser functions
        stmt_patterns = [
            r'def parse_(select|insert|update|delete|create|alter|drop)_stmt',
            r'parse_(select|insert|update|delete|create|alter|drop)',
            r'class (Select|Insert|Update|Delete|Create|Alter|Drop)'
        ]
        
        for pattern in stmt_patterns:
            for match in re.finditer(pattern, parser_content, re.IGNORECASE):
                stmt_type = match.group(1).lower() if match.group(1) else match.group(2).lower()
                constructs['statement_types'].add(stmt_type)
        
        # Identify join types
        join_patterns = [
            r'JOIN.*(INNER|LEFT|RIGHT|FULL|CROSS|NATURAL)',
            r'(inner|left|right|full|cross|natural)_join'
        ]
        for pattern in join_patterns:
            for match in re.finditer(pattern, parser_content, re.IGNORECASE):
                join_type = match.group(1).lower()
                constructs['join_types'].add(join_type)
        
        # Read AST nodes to understand structure
        with open(ast_path, 'r') as f:
            ast_content = f.read()
        
        # Identify functions from AST
        func_pattern = r'class (\w+Function)'
        for match in re.finditer(func_pattern, ast_content):
            constructs['functions'].add(match.group(1).replace('Function', '').lower())
        
        # Common SQL clauses
        common_clauses = ['where', 'group by', 'having', 'order by', 'limit', 'offset']
        constructs['clauses'].update(common_clauses)
        
        # Operators
        common_ops = ['+', '-', '*', '/', '=', '!=', '<', '>', '<=', '>=', 'and', 'or', 'not', 'in', 'like', 'between']
        constructs['operators'].update(common_ops)
        
        # Data types
        common_types = ['int', 'integer', 'varchar', 'text', 'real', 'float', 'date', 'timestamp', 'boolean']
        constructs['data_types'].update(common_types)
        
        return constructs
    
    def _generate_statements(self, grammar_rules: Dict, constructs: Dict[str, Any]) -> List[str]:
        """Generate diverse SQL statements."""
        statements = []
        
        # Start symbols typically include statement or query
        start_symbols = ['statement', 'sql_stmt', 'select_stmt', 'query']
        start_symbol = None
        for sym in start_symbols:
            if sym in grammar_rules:
                start_symbol = sym
                break
        
        if not start_symbol:
            # Fallback to first rule
            start_symbol = next(iter(grammar_rules.keys()))
        
        # Generate statements with increasing complexity
        complexity_levels = [1, 2, 3, 4, 5]
        
        for level in complexity_levels:
            num_statements = 5 + level * 3  # More statements at higher complexity
            
            for _ in range(num_statements):
                try:
                    stmt = self._generate_from_symbol(start_symbol, grammar_rules, 
                                                     constructs, level, max_depth=10)
                    if stmt and self._is_valid_sql(stmt):
                        statements.append(stmt)
                except:
                    continue
        
        # Add specific targeted statements for common coverage paths
        targeted = self._generate_targeted_statements(constructs)
        statements.extend(targeted)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_statements = []
        for stmt in statements:
            if stmt not in seen:
                seen.add(stmt)
                unique_statements.append(stmt)
        
        # Limit to reasonable number for efficiency bonus
        return unique_statements[:80]
    
    def _generate_from_symbol(self, symbol: str, grammar_rules: Dict, 
                            constructs: Dict[str, Any], level: int, 
                            max_depth: int, current_depth: int = 0) -> str:
        """Recursively generate from a grammar symbol."""
        if current_depth > max_depth:
            return ''
        
        if symbol not in grammar_rules:
            # Terminal symbol
            return symbol
        
        productions = grammar_rules[symbol]
        
        # Weight productions based on complexity and level
        weights = []
        for prod in productions:
            weight = 1.0
            # Prefer productions with keywords for our target constructs
            for token in prod:
                if token in constructs['statement_types']:
                    weight *= 2.0
                if token in constructs['join_types']:
                    weight *= 1.5
                if token in constructs['functions']:
                    weight *= 1.3
            
            # Adjust based on level - higher levels use more complex productions
            if len(prod) > 3:
                weight *= min(level / 3.0, 1.5)
            
            weights.append(weight)
        
        # Normalize weights
        total = sum(weights)
        if total == 0:
            weights = [1.0] * len(weights)
            total = len(weights)
        
        weights = [w / total for w in weights]
        
        # Select production
        selected = random.choices(productions, weights=weights, k=1)[0]
        
        # Generate from each symbol in production
        parts = []
        for token in selected:
            part = self._generate_from_symbol(token, grammar_rules, constructs, 
                                            level, max_depth, current_depth + 1)
            if part:
                parts.append(part)
        
        return ' '.join(parts)
    
    def _is_valid_sql(self, stmt: str) -> bool:
        """Basic validation to ensure SQL looks reasonable."""
        # Check for common SQL keywords
        keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP',
                   'FROM', 'WHERE', 'JOIN', 'GROUP BY', 'ORDER BY', 'LIMIT']
        
        stmt_upper = stmt.upper()
        has_keyword = any(keyword in stmt_upper for keyword in keywords)
        
        # Check for balanced parentheses
        paren_count = 0
        for char in stmt:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count < 0:
                    return False
        
        return has_keyword and paren_count == 0
    
    def _generate_targeted_statements(self, constructs: Dict[str, Any]) -> List[str]:
        """Generate statements targeting specific parser constructs."""
        targeted = []
        
        # Basic SELECT statements
        targeted.extend([
            "SELECT * FROM users",
            "SELECT id, name FROM users WHERE id = 1",
            "SELECT COUNT(*) FROM users",
            "SELECT * FROM users ORDER BY name",
            "SELECT * FROM users LIMIT 10",
        ])
        
        # JOIN statements
        if constructs['join_types']:
            targeted.extend([
                "SELECT u.*, o.* FROM users u JOIN orders o ON u.id = o.user_id",
                "SELECT * FROM users LEFT JOIN orders ON users.id = orders.user_id",
                "SELECT * FROM users RIGHT JOIN orders ON users.id = orders.user_id",
                "SELECT * FROM users u1 INNER JOIN users u2 ON u1.id = u2.manager_id",
                "SELECT * FROM a CROSS JOIN b",
            ])
        
        # Aggregate functions
        if constructs['functions']:
            targeted.extend([
                "SELECT AVG(salary) FROM employees",
                "SELECT SUM(amount) FROM transactions",
                "SELECT MIN(price), MAX(price) FROM products",
                "SELECT department, COUNT(*) FROM employees GROUP BY department",
                "SELECT department, AVG(salary) FROM employees GROUP BY department HAVING AVG(salary) > 50000",
            ])
        
        # Subqueries
        targeted.extend([
            "SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)",
            "SELECT name, (SELECT COUNT(*) FROM orders WHERE orders.user_id = users.id) FROM users",
            "SELECT * FROM (SELECT * FROM users WHERE active = 1) AS active_users",
        ])
        
        # INSERT/UPDATE/DELETE
        targeted.extend([
            "INSERT INTO users (name, email) VALUES ('John', 'john@example.com')",
            "UPDATE users SET email = 'new@example.com' WHERE id = 1",
            "DELETE FROM users WHERE id = 1",
        ])
        
        # CREATE TABLE
        targeted.extend([
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100), email TEXT)",
            "CREATE TABLE orders (id INT, user_id INT, amount REAL, FOREIGN KEY (user_id) REFERENCES users(id))",
        ])
        
        # Complex expressions
        targeted.extend([
            "SELECT (price * quantity) AS total FROM order_items",
            "SELECT * FROM products WHERE price BETWEEN 10 AND 100 AND category IN ('electronics', 'books')",
            "SELECT * FROM logs WHERE timestamp > '2023-01-01' AND (level = 'ERROR' OR level = 'WARNING')",
        ])
        
        # UNION and set operations
        targeted.extend([
            "SELECT id, name FROM users WHERE active = 1 UNION SELECT id, name FROM customers",
            "SELECT id FROM users INTERSECT SELECT user_id FROM orders",
        ])
        
        # Window functions (if supported)
        targeted.append("SELECT id, name, RANK() OVER (ORDER BY salary DESC) FROM employees")
        
        # CASE expressions
        targeted.extend([
            "SELECT name, CASE WHEN age < 18 THEN 'minor' WHEN age < 65 THEN 'adult' ELSE 'senior' END FROM people",
            "SELECT product_id, SUM(CASE WHEN status = 'completed' THEN amount ELSE 0 END) FROM orders GROUP BY product_id",
        ])
        
        # NULL handling
        targeted.extend([
            "SELECT * FROM users WHERE email IS NULL",
            "SELECT COALESCE(email, 'N/A') FROM users",
        ])
        
        # DISTINCT and ORDER BY multiple columns
        targeted.extend([
            "SELECT DISTINCT department FROM employees",
            "SELECT * FROM products ORDER BY category, price DESC",
        ])
        
        # LIMIT with OFFSET
        targeted.extend([
            "SELECT * FROM users LIMIT 10 OFFSET 20",
            "SELECT * FROM products ORDER BY id LIMIT 5",
        ])
        
        # LIKE and pattern matching
        targeted.extend([
            "SELECT * FROM users WHERE name LIKE 'J%'",
            "SELECT * FROM products WHERE description LIKE '%sale%'",
        ])
        
        # Mathematical operations
        targeted.extend([
            "SELECT 1 + 2 * 3",
            "SELECT POWER(2, 10)",
            "SELECT ROUND(price * 1.1, 2) FROM products",
        ])
        
        # Date functions
        targeted.extend([
            "SELECT CURRENT_DATE",
            "SELECT * FROM events WHERE event_date > DATE('2023-01-01')",
            "SELECT DATE_ADD(CURRENT_DATE, INTERVAL 7 DAY)",
        ])
        
        # String functions
        targeted.extend([
            "SELECT UPPER(name) FROM users",
            "SELECT CONCAT(first_name, ' ', last_name) AS full_name FROM customers",
            "SELECT SUBSTRING(email, 1, LOCATE('@', email) - 1) FROM users",
        ])
        
        # Nested queries in FROM clause
        targeted.append("SELECT * FROM (SELECT user_id, COUNT(*) as order_count FROM orders GROUP BY user_id) AS user_stats WHERE order_count > 5")
        
        # Multiple JOINs
        targeted.append("SELECT u.name, o.order_date, p.product_name FROM users u JOIN orders o ON u.id = o.user_id JOIN products p ON o.product_id = p.id")
        
        # EXISTS
        targeted.append("SELECT * FROM users WHERE EXISTS (SELECT 1 FROM orders WHERE orders.user_id = users.id)")
        
        # ALTER TABLE
        targeted.extend([
            "ALTER TABLE users ADD COLUMN phone VARCHAR(20)",
            "ALTER TABLE users DROP COLUMN old_column",
        ])
        
        # DROP TABLE
        targeted.append("DROP TABLE IF EXISTS temp_table")
        
        # Index creation
        targeted.append("CREATE INDEX idx_users_email ON users(email)")
        
        # Transactions
        targeted.extend([
            "BEGIN TRANSACTION",
            "COMMIT",
            "ROLLBACK",
        ])
        
        # Views
        targeted.extend([
            "CREATE VIEW active_users AS SELECT * FROM users WHERE active = 1",
            "SELECT * FROM active_users",
        ])
        
        # Casting
        targeted.append("SELECT CAST(price AS INTEGER) FROM products")
        
        # Group by with rollup
        targeted.append("SELECT department, gender, COUNT(*) FROM employees GROUP BY ROLLUP(department, gender)")
        
        # Common table expressions (CTEs)
        targeted.append("WITH recent_orders AS (SELECT * FROM orders WHERE order_date > '2023-01-01') SELECT * FROM recent_orders")
        
        return targeted