import os
import sys
import importlib.util
import ast
import random
import re
from collections import defaultdict, deque
from typing import List, Set, Dict, Tuple, Optional, Any
import itertools

class Solution:
    def solve(self, resources_path: str) -> List[str]:
        # Load the grammar and analyze the parser
        grammar_path = os.path.join(resources_path, "sql_grammar.txt")
        engine_path = os.path.join(resources_path, "sql_engine")
        
        # Add engine path to sys.path to import parser
        sys.path.insert(0, resources_path)
        
        try:
            # Import the SQL engine modules
            spec = importlib.util.spec_from_file_location("sql_engine.parser", 
                                                         os.path.join(engine_path, "parser.py"))
            parser_module = importlib.util.module_from_spec(spec)
            sys.modules["sql_engine.parser"] = parser_module
            spec.loader.exec_module(parser_module)
            
            spec = importlib.util.spec_from_file_location("sql_engine.tokenizer", 
                                                         os.path.join(engine_path, "tokenizer.py"))
            tokenizer_module = importlib.util.module_from_spec(spec)
            sys.modules["sql_engine.tokenizer"] = tokenizer_module
            spec.loader.exec_module(tokenizer_module)
            
            spec = importlib.util.spec_from_file_location("sql_engine.ast_nodes", 
                                                         os.path.join(engine_path, "ast_nodes.py"))
            ast_nodes_module = importlib.util.module_from_spec(spec)
            sys.modules["sql_engine.ast_nodes"] = ast_nodes_module
            spec.loader.exec_module(ast_nodes_module)
            
        except Exception:
            # If import fails, generate generic SQL statements
            return self._generate_generic_sql()
        
        # Analyze parser structure to understand coverage targets
        parser_analysis = self._analyze_parser(parser_module, tokenizer_module, ast_nodes_module)
        
        # Read grammar
        grammar_rules = self._read_grammar(grammar_path)
        
        # Generate comprehensive test cases
        test_cases = self._generate_comprehensive_tests(grammar_rules, parser_analysis)
        
        return test_cases
    
    def _read_grammar(self, grammar_path: str) -> Dict[str, List[str]]:
        """Read BNF grammar rules from file."""
        grammar = defaultdict(list)
        try:
            with open(grammar_path, 'r') as f:
                current_rule = None
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if '::=' in line:
                        parts = line.split('::=', 1)
                        current_rule = parts[0].strip()
                        production = parts[1].strip()
                        if current_rule and production:
                            grammar[current_rule].append(production)
                    elif current_rule and line:
                        grammar[current_rule].append(line)
        except Exception:
            pass
        return dict(grammar)
    
    def _analyze_parser(self, parser_module, tokenizer_module, ast_nodes_module) -> Dict[str, Any]:
        """Analyze parser structure to identify coverage targets."""
        analysis = {
            'functions': set(),
            'ast_classes': set(),
            'keywords': set(),
            'patterns': set()
        }
        
        # Collect parser functions
        for name in dir(parser_module):
            if name.startswith('parse_'):
                analysis['functions'].add(name)
        
        # Collect AST node classes
        for name in dir(ast_nodes_module):
            if name.endswith('Node') or name[0].isupper():
                analysis['ast_classes'].add(name)
        
        # Extract patterns from tokenizer if available
        try:
            for name in dir(tokenizer_module):
                if 'TOKEN' in name or 'KEYWORD' in name:
                    value = getattr(tokenizer_module, name)
                    if isinstance(value, str):
                        analysis['keywords'].add(value.upper())
        except Exception:
            pass
        
        # Add common SQL patterns
        common_patterns = {
            'SELECT_WITH_JOIN',
            'SELECT_WITH_SUBQUERY',
            'SELECT_WITH_AGGREGATE',
            'INSERT_VALUES',
            'UPDATE_WITH_WHERE',
            'DELETE_WITH_WHERE',
            'CREATE_TABLE',
            'DROP_TABLE',
            'ALTER_TABLE',
            'NESTED_EXPRESSIONS',
            'FUNCTION_CALLS',
            'CASE_STATEMENTS',
            'UNION_QUERIES',
            'GROUP_BY_HAVING',
            'ORDER_BY_LIMIT',
            'WITH_CTE'
        }
        analysis['patterns'].update(common_patterns)
        
        return analysis
    
    def _generate_generic_sql(self) -> List[str]:
        """Generate generic SQL test cases when parser analysis fails."""
        test_cases = []
        
        # Basic SELECT statements
        test_cases.extend([
            "SELECT * FROM users",
            "SELECT id, name FROM users WHERE age > 18",
            "SELECT COUNT(*) FROM orders",
            "SELECT department, AVG(salary) FROM employees GROUP BY department",
            "SELECT * FROM users ORDER BY name DESC",
            "SELECT * FROM users LIMIT 10",
            "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id",
            "SELECT * FROM users WHERE id IN (1, 2, 3)",
            "SELECT * FROM users WHERE name LIKE 'J%'",
            "SELECT * FROM products WHERE price BETWEEN 10 AND 100",
        ])
        
        # Complex SELECT with subqueries
        test_cases.extend([
            "SELECT * FROM (SELECT id, name FROM users) AS sub",
            "SELECT name FROM users WHERE id = (SELECT MAX(user_id) FROM orders)",
            "SELECT * FROM users WHERE EXISTS (SELECT 1 FROM orders WHERE orders.user_id = users.id)",
            "SELECT department, (SELECT COUNT(*) FROM employees e2 WHERE e2.department = e1.department) FROM employees e1 GROUP BY department",
        ])
        
        # INSERT statements
        test_cases.extend([
            "INSERT INTO users (id, name, age) VALUES (1, 'John', 30)",
            "INSERT INTO users VALUES (1, 'John', 30, 'john@example.com')",
            "INSERT INTO orders SELECT * FROM pending_orders WHERE status = 'approved'",
        ])
        
        # UPDATE statements
        test_cases.extend([
            "UPDATE users SET age = 31 WHERE id = 1",
            "UPDATE employees SET salary = salary * 1.1 WHERE department = 'Engineering'",
        ])
        
        # DELETE statements
        test_cases.extend([
            "DELETE FROM users WHERE id = 1",
            "DELETE FROM logs WHERE created_at < '2023-01-01'",
        ])
        
        # CREATE statements
        test_cases.extend([
            "CREATE TABLE users (id INT, name VARCHAR(50), age INT)",
            "CREATE INDEX idx_name ON users (name)",
        ])
        
        # DROP statements
        test_cases.extend([
            "DROP TABLE users",
            "DROP INDEX idx_name",
        ])
        
        # Complex expressions and functions
        test_cases.extend([
            "SELECT (price * quantity) AS total FROM order_items",
            "SELECT CONCAT(first_name, ' ', last_name) AS full_name FROM employees",
            "SELECT UPPER(name), LOWER(email) FROM users",
            "SELECT COALESCE(middle_name, '') FROM persons",
            "SELECT CASE WHEN age < 18 THEN 'minor' WHEN age < 65 THEN 'adult' ELSE 'senior' END FROM users",
            "SELECT SUM(CASE WHEN status = 'completed' THEN total ELSE 0 END) FROM orders",
        ])
        
        # JOIN variations
        test_cases.extend([
            "SELECT * FROM users LEFT JOIN orders ON users.id = orders.user_id",
            "SELECT * FROM users RIGHT JOIN orders ON users.id = orders.user_id",
            "SELECT * FROM users INNER JOIN orders ON users.id = orders.user_id",
            "SELECT * FROM users FULL OUTER JOIN orders ON users.id = orders.user_id",
            "SELECT * FROM a CROSS JOIN b",
            "SELECT * FROM users u1 JOIN users u2 ON u1.manager_id = u2.id",
        ])
        
        # WITH CTE
        test_cases.extend([
            "WITH recent_orders AS (SELECT * FROM orders WHERE order_date > '2023-01-01') SELECT * FROM recent_orders",
            "WITH recursive_cte AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM recursive_cte WHERE n < 10) SELECT * FROM recursive_cte",
        ])
        
        # UNION/INTERSECT/EXCEPT
        test_cases.extend([
            "SELECT id FROM active_users UNION SELECT id FROM archived_users",
            "SELECT id FROM users INTERSECT SELECT user_id FROM orders",
            "SELECT id FROM all_users EXCEPT SELECT id FROM banned_users",
        ])
        
        # ALTER statements
        test_cases.extend([
            "ALTER TABLE users ADD COLUMN email VARCHAR(100)",
            "ALTER TABLE users DROP COLUMN middle_name",
            "ALTER TABLE users MODIFY COLUMN age BIGINT",
        ])
        
        # Aggregations with HAVING
        test_cases.extend([
            "SELECT department, COUNT(*) FROM employees GROUP BY department HAVING COUNT(*) > 5",
            "SELECT category, AVG(price) FROM products GROUP BY category HAVING AVG(price) > 100",
        ])
        
        # Date functions and intervals
        test_cases.extend([
            "SELECT * FROM orders WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'",
            "SELECT DATE_ADD(created_at, INTERVAL 1 MONTH) FROM events",
            "SELECT EXTRACT(YEAR FROM order_date) FROM orders",
        ])
        
        # Window functions
        test_cases.extend([
            "SELECT id, name, salary, RANK() OVER (PARTITION BY department ORDER BY salary DESC) FROM employees",
            "SELECT id, amount, SUM(amount) OVER (ORDER BY date) FROM transactions",
        ])
        
        # NULL handling
        test_cases.extend([
            "SELECT * FROM users WHERE email IS NULL",
            "SELECT * FROM users WHERE email IS NOT NULL",
            "SELECT IFNULL(middle_name, 'N/A') FROM persons",
        ])
        
        # String operations
        test_cases.extend([
            "SELECT SUBSTRING(name, 1, 3) FROM users",
            "SELECT REPLACE(description, 'old', 'new') FROM products",
            "SELECT TRIM(name) FROM users",
        ])
        
        # Mathematical operations
        test_cases.extend([
            "SELECT price * 1.1 AS price_with_tax FROM products",
            "SELECT POWER(2, 10) AS kilobyte",
            "SELECT ROUND(price, 2) FROM products",
        ])
        
        # Limit variations
        test_cases.extend([
            "SELECT * FROM users LIMIT 10 OFFSET 20",
            "SELECT * FROM users FETCH FIRST 10 ROWS ONLY",
        ])
        
        # Complex WHERE conditions
        test_cases.extend([
            "SELECT * FROM users WHERE (age > 18 AND status = 'active') OR (age <= 18 AND guardian_id IS NOT NULL)",
            "SELECT * FROM products WHERE price > 100 AND (category = 'electronics' OR category = 'appliances')",
        ])
        
        # Cast operations
        test_cases.extend([
            "SELECT CAST(price AS DECIMAL(10,2)) FROM products",
            "SELECT id::text FROM users",
        ])
        
        return test_cases[:100]  # Return up to 100 test cases
    
    def _generate_comprehensive_tests(self, grammar_rules: Dict[str, List[str]], 
                                    parser_analysis: Dict[str, Any]) -> List[str]:
        """Generate comprehensive test cases based on grammar and parser analysis."""
        test_cases = []
        
        # Start with generic test cases
        test_cases.extend(self._generate_generic_sql())
        
        # Add variations based on grammar analysis
        if grammar_rules:
            additional_cases = self._generate_from_grammar(grammar_rules)
            test_cases.extend(additional_cases)
        
        # Ensure we have a diverse set of statements
        # Remove duplicates while preserving order
        seen = set()
        unique_cases = []
        for case in test_cases:
            if case not in seen:
                seen.add(case)
                unique_cases.append(case)
        
        # Prioritize complex statements first (they tend to cover more code paths)
        unique_cases.sort(key=lambda x: (-len(x), x))
        
        # Take up to 150 test cases (balance between coverage and efficiency)
        return unique_cases[:150]
    
    def _generate_from_grammar(self, grammar_rules: Dict[str, List[str]]) -> List[str]:
        """Generate SQL statements from grammar rules."""
        test_cases = []
        
        # Extract key patterns from grammar
        start_symbols = ['<statement>', '<query>', '<select-statement>', '<sql>']
        
        for symbol in start_symbols:
            if symbol in grammar_rules:
                # Generate basic statements from each production rule
                for production in grammar_rules[symbol][:5]:  # Limit to 5 per rule
                    test_case = self._expand_production(production, grammar_rules)
                    if test_case:
                        test_cases.append(test_case)
        
        # Generate additional variations
        variations = self._generate_variations(test_cases)
        test_cases.extend(variations)
        
        return test_cases
    
    def _expand_production(self, production: str, grammar_rules: Dict[str, List[str]]) -> str:
        """Expand a production rule into a concrete SQL statement."""
        # Simple expansion - replace non-terminals with example values
        replacements = {
            '<table-name>': 'users',
            '<column-name>': 'id',
            '<identifier>': 'my_table',
            '<string-literal>': "'example'",
            '<number>': '42',
            '<expression>': '1 + 1',
            '<condition>': 'id > 0',
            '<value>': 'NULL',
            '<data-type>': 'INT',
            '<function-name>': 'COUNT',
            '<parameter>': 'param',
            '<alias>': 'alias',
        }
        
        result = production
        for non_terminal, replacement in replacements.items():
            result = result.replace(non_terminal, replacement)
        
        # Clean up the result
        result = re.sub(r'\s+', ' ', result).strip()
        result = result.replace('|', '').replace('[', '').replace(']', '')
        result = re.sub(r'\s*,\s*', ', ', result)
        result = re.sub(r'\(\s+', '(', result)
        result = re.sub(r'\s+\)', ')', result)
        
        # Capitalize SQL keywords (common convention)
        sql_keywords = ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 
                       'CREATE', 'DROP', 'ALTER', 'TABLE', 'INDEX', 'VALUES',
                       'SET', 'INTO', 'JOIN', 'ON', 'GROUP BY', 'ORDER BY',
                       'HAVING', 'LIMIT', 'OFFSET', 'UNION', 'INTERSECT',
                       'EXCEPT', 'WITH', 'AS', 'AND', 'OR', 'NOT', 'NULL',
                       'IS', 'IN', 'BETWEEN', 'LIKE', 'EXISTS', 'CASE',
                       'WHEN', 'THEN', 'ELSE', 'END', 'DISTINCT', 'ALL',
                       'ANY', 'SOME', 'ASC', 'DESC', 'PRIMARY', 'KEY',
                       'FOREIGN', 'REFERENCES', 'CHECK', 'DEFAULT',
                       'UNIQUE', 'CONSTRAINT']
        
        for keyword in sql_keywords:
            pattern = rf'\b{keyword.lower()}\b'
            result = re.sub(pattern, keyword, result, flags=re.IGNORECASE)
        
        return result if result and len(result) > 5 else None
    
    def _generate_variations(self, base_cases: List[str]) -> List[str]:
        """Generate variations of base test cases."""
        variations = []
        
        for case in base_cases[:20]:  # Limit to first 20 base cases
            # Add NULL variations
            if 'WHERE' in case:
                variations.append(case + ' AND column IS NOT NULL')
            
            # Add ORDER BY variations
            if 'SELECT' in case and 'ORDER BY' not in case and 'FROM' in case:
                variations.append(case + ' ORDER BY 1')
            
            # Add LIMIT variations
            if 'SELECT' in case and 'LIMIT' not in case and 'FROM' in case:
                variations.append(case + ' LIMIT 5')
            
            # Add alias variations
            if 'FROM' in case and ' AS ' not in case:
                # Add table alias
                parts = case.split('FROM')
                if len(parts) == 2:
                    variations.append(parts[0] + 'FROM ' + parts[1].strip().split()[0] + ' AS t')
            
            # Add function call variations
            if 'SELECT' in case and 'COUNT' not in case.upper():
                # Replace simple column with function
                modified = re.sub(r'SELECT\s+(\w+)', r'SELECT COUNT(\1)', case, flags=re.IGNORECASE)
                if modified != case:
                    variations.append(modified)
        
        return variations