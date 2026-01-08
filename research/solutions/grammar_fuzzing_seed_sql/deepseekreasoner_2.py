import os
import re
import sys
import ast
import random
import itertools
import collections
from typing import List, Dict, Set, Tuple, Optional, Any
from pathlib import Path

class GrammarParser:
    """Parse BNF-style grammar file into production rules"""
    
    def __init__(self, grammar_path: str):
        self.rules = {}
        self.start_symbol = None
        self.parse_grammar(grammar_path)
    
    def parse_grammar(self, grammar_path: str):
        """Parse grammar file into rules dictionary"""
        with open(grammar_path, 'r') as f:
            lines = f.readlines()
        
        current_rule = None
        current_productions = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Check for rule definition: <rule> ::= production
            match = re.match(r'<([^>]+)>\s*::=\s*(.*)', line)
            if match:
                # Save previous rule if exists
                if current_rule:
                    self.rules[current_rule] = current_productions
                
                current_rule = match.group(1)
                current_productions = []
                production = match.group(2).strip()
                if production:
                    current_productions.append(production)
            
            # Check for continuation: | production
            elif line.startswith('|'):
                production = line[1:].strip()
                if production:
                    current_productions.append(production)
            
            # Check for alternative format without pipe
            elif current_rule:
                current_productions.append(line)
        
        # Save the last rule
        if current_rule:
            self.rules[current_rule] = current_productions
        
        # Determine start symbol (usually statement or query)
        if 'statement' in self.rules:
            self.start_symbol = 'statement'
        elif 'query' in self.rules:
            self.start_symbol = 'query'
        else:
            self.start_symbol = list(self.rules.keys())[0] if self.rules else None
    
    def get_all_symbols(self) -> Set[str]:
        """Get all non-terminal symbols in the grammar"""
        symbols = set()
        for rule, productions in self.rules.items():
            symbols.add(f"<{rule}>")
            for prod in productions:
                # Find all non-terminals in production
                for match in re.finditer(r'<([^>]+)>', prod):
                    symbols.add(f"<{match.group(1)}>")
        return symbols

class StatementGenerator:
    """Generate SQL statements from grammar rules"""
    
    def __init__(self, grammar: GrammarParser, max_depth: int = 5):
        self.grammar = grammar
        self.max_depth = max_depth
        self.terminal_cache = {}
        
        # Common terminal values for different non-terminals
        self.terminal_values = {
            '<identifier>': ['table1', 'table2', 'col1', 'col2', 'col3', 'id', 'name', 'value', 'amount', 'date'],
            '<table_name>': ['users', 'orders', 'products', 'customers', 'employees', 'inventory'],
            '<column_name>': ['id', 'name', 'age', 'salary', 'price', 'quantity', 'total', 'created_at'],
            '<literal>': ['42', "'John'", "'Doe'", "'2023-01-01'", '3.14', 'true', 'false', 'null'],
            '<number>': ['1', '2', '10', '100', '1000', '3.14', '0.5'],
            '<string_literal>': ["'hello'", "'world'", "'test'", "'sample'"],
            '<boolean_literal>': ['true', 'false'],
            '<function_name>': ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'UPPER', 'LOWER', 'COALESCE'],
            '<operator>': ['=', '!=', '<', '>', '<=', '>=', 'LIKE', 'IN', 'IS', 'IS NOT'],
            '<join_type>': ['INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN'],
            '<data_type>': ['INTEGER', 'VARCHAR(255)', 'TEXT', 'DATE', 'BOOLEAN', 'DECIMAL(10,2)'],
        }
    
    def generate_from_rule(self, rule: str, depth: int = 0) -> str:
        """Generate a string from a grammar rule"""
        if depth > self.max_depth:
            return self._generate_terminal(rule)
        
        # Check if it's a non-terminal
        match = re.match(r'<([^>]+)>', rule)
        if match:
            rule_name = match.group(1)
            if rule_name in self.grammar.rules:
                # Randomly choose a production
                productions = self.grammar.rules[rule_name]
                if productions:
                    chosen = random.choice(productions)
                    return self._expand_production(chosen, depth + 1)
        
        return self._generate_terminal(rule)
    
    def _expand_production(self, production: str, depth: int) -> str:
        """Expand a production string by replacing non-terminals"""
        # Split into tokens while preserving non-terminals
        tokens = []
        pos = 0
        while pos < len(production):
            # Look for non-terminal
            match = re.search(r'<([^>]+)>', production[pos:])
            if match:
                start = pos + match.start()
                end = pos + match.end()
                
                # Add text before non-terminal
                if start > pos:
                    tokens.append(production[pos:start])
                
                # Add non-terminal
                tokens.append(production[start:end])
                pos = end
            else:
                # Add remaining text
                if pos < len(production):
                    tokens.append(production[pos:])
                break
        
        # Expand each token
        result_parts = []
        for token in tokens:
            if token.startswith('<'):
                expanded = self.generate_from_rule(token, depth)
                result_parts.append(expanded)
            else:
                result_parts.append(token)
        
        return ''.join(result_parts).strip()
    
    def _generate_terminal(self, rule: str) -> str:
        """Generate a terminal value for a rule"""
        # Check cache first
        if rule in self.terminal_cache:
            return random.choice(self.terminal_cache[rule])
        
        # Check predefined values
        if rule in self.terminal_values:
            return random.choice(self.terminal_values[rule])
        
        # Generate based on rule type
        if rule == '<identifier>' or rule == '<column_name>':
            return random.choice(['col1', 'col2', 'col3', 'name', 'value'])
        elif rule == '<table_name>':
            return random.choice(['table1', 'table2', 't'])
        elif rule == '<literal>':
            return random.choice(['42', "'text'", 'null'])
        elif rule == '<number>':
            return random.choice(['1', '100', '3.14'])
        
        # Remove angle brackets for unknown rules
        if rule.startswith('<') and rule.endswith('>'):
            return rule[1:-1]
        
        return rule

class ParserAnalyzer:
    """Analyze parser source code to identify coverage targets"""
    
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.import_statements = []
        self.function_defs = []
        self.class_defs = []
        self.if_statements = []
        self.try_blocks = []
        
        self.analyze_parser()
    
    def analyze_parser(self):
        """Analyze parser source files to understand structure"""
        parser_path = os.path.join(self.engine_path, 'parser.py')
        tokenizer_path = os.path.join(self.engine_path, 'tokenizer.py')
        ast_path = os.path.join(self.engine_path, 'ast_nodes.py')
        
        for path in [parser_path, tokenizer_path, ast_path]:
            if os.path.exists(path):
                self._analyze_file(path)
    
    def _analyze_file(self, path: str):
        """Analyze a Python file for structure"""
        try:
            with open(path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.import_statements.append(node)
                elif isinstance(node, ast.ImportFrom):
                    self.import_statements.append(node)
                elif isinstance(node, ast.FunctionDef):
                    self.function_defs.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    self.class_defs.append(node.name)
                elif isinstance(node, ast.If):
                    self.if_statements.append(node)
                elif isinstance(node, ast.Try):
                    self.try_blocks.append(node)
        
        except Exception as e:
            # If we can't parse, continue with what we have
            pass

class CoverageOptimizer:
    """Optimize test cases for maximum coverage"""
    
    def __init__(self, grammar_path: str, engine_path: str):
        self.grammar = GrammarParser(grammar_path)
        self.analyzer = ParserAnalyzer(engine_path)
        self.generator = StatementGenerator(self.grammar)
        
        # Categories of SQL statements to cover
        self.statement_categories = [
            'simple_select',
            'select_with_where',
            'select_with_join',
            'select_with_group_by',
            'select_with_having',
            'select_with_order_by',
            'select_with_limit',
            'select_with_subquery',
            'insert_statement',
            'update_statement',
            'delete_statement',
            'create_table',
            'drop_table',
            'alter_table',
            'complex_nested',
        ]
    
    def generate_diverse_statements(self) -> List[str]:
        """Generate diverse SQL statements covering various patterns"""
        statements = []
        
        # Basic SELECT statements
        statements.extend(self._generate_select_variants())
        
        # DML statements
        statements.extend(self._generate_dml_statements())
        
        # DDL statements
        statements.extend(self._generate_ddl_statements())
        
        # Complex queries
        statements.extend(self._generate_complex_queries())
        
        # Edge cases
        statements.extend(self._generate_edge_cases())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_statements = []
        for stmt in statements:
            if stmt not in seen:
                seen.add(stmt)
                unique_statements.append(stmt)
        
        return unique_statements
    
    def _generate_select_variants(self) -> List[str]:
        """Generate various SELECT statement variants"""
        variants = []
        
        # Simple SELECT
        variants.extend([
            "SELECT * FROM users",
            "SELECT id, name FROM users",
            "SELECT DISTINCT name FROM users",
            "SELECT 1, 2, 3",
            "SELECT * FROM (SELECT * FROM users) AS t",
        ])
        
        # SELECT with WHERE
        variants.extend([
            "SELECT * FROM users WHERE id = 1",
            "SELECT * FROM users WHERE name = 'John' AND age > 18",
            "SELECT * FROM users WHERE id IN (1, 2, 3)",
            "SELECT * FROM users WHERE name LIKE '%John%'",
            "SELECT * FROM users WHERE age BETWEEN 18 AND 65",
        ])
        
        # SELECT with JOIN
        variants.extend([
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id",
            "SELECT users.name, orders.total FROM users LEFT JOIN orders ON users.id = orders.user_id",
            "SELECT * FROM users INNER JOIN orders ON users.id = orders.user_id WHERE orders.total > 100",
            "SELECT * FROM users u1 JOIN users u2 ON u1.id = u2.manager_id",
        ])
        
        # SELECT with GROUP BY
        variants.extend([
            "SELECT department, COUNT(*) FROM employees GROUP BY department",
            "SELECT department, AVG(salary) FROM employees GROUP BY department HAVING AVG(salary) > 50000",
            "SELECT YEAR(created_at), COUNT(*) FROM orders GROUP BY YEAR(created_at)",
        ])
        
        # SELECT with ORDER BY and LIMIT
        variants.extend([
            "SELECT * FROM users ORDER BY name",
            "SELECT * FROM users ORDER BY name DESC, id ASC",
            "SELECT * FROM users LIMIT 10",
            "SELECT * FROM users ORDER BY id LIMIT 10 OFFSET 5",
        ])
        
        # SELECT with subqueries
        variants.extend([
            "SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)",
            "SELECT name, (SELECT COUNT(*) FROM orders WHERE orders.user_id = users.id) AS order_count FROM users",
            "SELECT * FROM (SELECT * FROM users WHERE active = true) AS active_users",
        ])
        
        return variants
    
    def _generate_dml_statements(self) -> List[str]:
        """Generate DML statements (INSERT, UPDATE, DELETE)"""
        statements = []
        
        # INSERT statements
        statements.extend([
            "INSERT INTO users (id, name) VALUES (1, 'John')",
            "INSERT INTO users VALUES (1, 'John', 30)",
            "INSERT INTO users SELECT * FROM old_users",
            "INSERT INTO users (name, age) VALUES ('John', 30), ('Jane', 25)",
        ])
        
        # UPDATE statements
        statements.extend([
            "UPDATE users SET name = 'John' WHERE id = 1",
            "UPDATE users SET age = age + 1, active = true",
            "UPDATE users SET name = 'John' WHERE id IN (SELECT user_id FROM orders)",
        ])
        
        # DELETE statements
        statements.extend([
            "DELETE FROM users WHERE id = 1",
            "DELETE FROM users WHERE age < 18",
            "DELETE FROM users WHERE id IN (SELECT user_id FROM inactive_orders)",
        ])
        
        return statements
    
    def _generate_ddl_statements(self) -> List[str]:
        """Generate DDL statements (CREATE, DROP, ALTER)"""
        statements = []
        
        # CREATE TABLE
        statements.extend([
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)",
            "CREATE TABLE orders (id INTEGER, user_id INTEGER, total DECIMAL(10,2), FOREIGN KEY (user_id) REFERENCES users(id))",
            "CREATE TABLE IF NOT EXISTS users (id INTEGER, name VARCHAR(255))",
            "CREATE TEMPORARY TABLE temp_users (id INTEGER, name TEXT)",
        ])
        
        # DROP TABLE
        statements.extend([
            "DROP TABLE users",
            "DROP TABLE IF EXISTS users",
            "DROP TABLE users CASCADE",
        ])
        
        # ALTER TABLE
        statements.extend([
            "ALTER TABLE users ADD COLUMN email TEXT",
            "ALTER TABLE users DROP COLUMN email",
            "ALTER TABLE users RENAME TO customers",
            "ALTER TABLE users RENAME COLUMN name TO full_name",
        ])
        
        return statements
    
    def _generate_complex_queries(self) -> List[str]:
        """Generate complex queries with multiple clauses"""
        statements = []
        
        statements.extend([
            "SELECT u.name, COUNT(o.id) as order_count, SUM(o.total) as total_spent "
            "FROM users u "
            "LEFT JOIN orders o ON u.id = o.user_id "
            "WHERE u.active = true "
            "GROUP BY u.id, u.name "
            "HAVING COUNT(o.id) > 5 "
            "ORDER BY total_spent DESC "
            "LIMIT 10",
            
            "SELECT department, "
            "       COUNT(*) as emp_count, "
            "       AVG(salary) as avg_salary, "
            "       MAX(salary) as max_salary "
            "FROM employees "
            "WHERE hire_date > '2020-01-01' "
            "GROUP BY department "
            "HAVING AVG(salary) > 50000 "
            "ORDER BY avg_salary DESC",
            
            "WITH RECURSIVE cte (n) AS ("
            "    SELECT 1 "
            "    UNION ALL "
            "    SELECT n + 1 FROM cte WHERE n < 10"
            ") "
            "SELECT * FROM cte",
            
            "SELECT * FROM ("
            "    SELECT id, name, "
            "           ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as rank "
            "    FROM employees"
            ") ranked "
            "WHERE rank <= 3",
        ])
        
        return statements
    
    def _generate_edge_cases(self) -> List[str]:
        """Generate edge case statements"""
        statements = []
        
        statements.extend([
            # Empty values
            "SELECT * FROM users WHERE 1=0",
            "SELECT NULL",
            
            # Complex expressions
            "SELECT CASE WHEN age < 18 THEN 'minor' WHEN age < 65 THEN 'adult' ELSE 'senior' END FROM users",
            "SELECT COALESCE(name, 'Unknown'), NULLIF(age, 0) FROM users",
            
            # Multiple joins
            "SELECT * FROM t1 JOIN t2 ON t1.id = t2.id JOIN t3 ON t2.id = t3.id",
            
            # Nested subqueries
            "SELECT * FROM (SELECT * FROM (SELECT * FROM users) AS t1) AS t2",
            
            # Complex WHERE conditions
            "SELECT * FROM users WHERE (id = 1 OR id = 2) AND (name LIKE '%John%' OR name LIKE '%Jane%')",
            
            # Functions and aggregates
            "SELECT UPPER(name), LOWER(name), LENGTH(name) FROM users",
            "SELECT COUNT(*), COUNT(DISTINCT department), SUM(salary) FROM employees",
            
            # Window functions
            "SELECT name, salary, AVG(salary) OVER (PARTITION BY department) FROM employees",
            
            # Set operations
            "SELECT * FROM users WHERE age < 18 UNION SELECT * FROM users WHERE age > 65",
            "SELECT * FROM table1 INTERSECT SELECT * FROM table2",
            "SELECT * FROM table1 EXCEPT SELECT * FROM table2",
        ])
        
        return statements

class Solution:
    def solve(self, resources_path: str) -> List[str]:
        """
        Return SQL test cases designed to maximize parser coverage.
        """
        grammar_path = os.path.join(resources_path, 'sql_grammar.txt')
        engine_path = os.path.join(resources_path, 'sql_engine')
        
        # Initialize optimizer
        optimizer = CoverageOptimizer(grammar_path, engine_path)
        
        # Generate diverse statements
        statements = optimizer.generate_diverse_statements()
        
        # Limit to reasonable number for efficiency (target ~30-50 statements)
        # This balances coverage with efficiency bonus
        if len(statements) > 50:
            # Prioritize variety: take at least one from each category pattern
            prioritized = []
            
            # Ensure we have different types
            seen_patterns = set()
            for stmt in statements:
                pattern = self._classify_statement(stmt)
                if pattern not in seen_patterns:
                    seen_patterns.add(pattern)
                    prioritized.append(stmt)
            
            # Add remaining statements up to limit
            remaining = [s for s in statements if s not in prioritized]
            prioritized.extend(remaining[:50 - len(prioritized)])
            
            statements = prioritized
        
        return statements[:50]  # Ensure we don't exceed 50
    
    def _classify_statement(self, stmt: str) -> str:
        """Classify SQL statement into broad categories"""
        stmt_upper = stmt.upper().strip()
        
        if stmt_upper.startswith('SELECT'):
            if 'JOIN' in stmt_upper:
                return 'SELECT_JOIN'
            elif 'WHERE' in stmt_upper:
                return 'SELECT_WHERE'
            elif 'GROUP BY' in stmt_upper:
                return 'SELECT_GROUP'
            elif 'ORDER BY' in stmt_upper:
                return 'SELECT_ORDER'
            elif 'UNION' in stmt_upper or 'INTERSECT' in stmt_upper or 'EXCEPT' in stmt_upper:
                return 'SELECT_SET'
            elif 'WITH' in stmt_upper:
                return 'SELECT_WITH'
            else:
                return 'SELECT_SIMPLE'
        elif stmt_upper.startswith('INSERT'):
            return 'INSERT'
        elif stmt_upper.startswith('UPDATE'):
            return 'UPDATE'
        elif stmt_upper.startswith('DELETE'):
            return 'DELETE'
        elif stmt_upper.startswith('CREATE'):
            return 'CREATE'
        elif stmt_upper.startswith('DROP'):
            return 'DROP'
        elif stmt_upper.startswith('ALTER'):
            return 'ALTER'
        else:
            return 'OTHER'