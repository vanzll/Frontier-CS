import os
import re
import random
from typing import List, Dict, Set, Tuple, Optional
import ast
import sys
from pathlib import Path

class SQLGrammar:
    """Parse and represent SQL grammar from BNF file"""
    
    def __init__(self, grammar_path: str):
        self.rules: Dict[str, List[List[str]]] = {}
        self.terminals: Set[str] = set()
        self.non_terminals: Set[str] = set()
        self.start_symbol = "sql_stmt"
        self._parse_grammar(grammar_path)
    
    def _parse_grammar(self, grammar_path: str):
        """Parse BNF-style grammar file"""
        with open(grammar_path, 'r') as f:
            lines = f.read().splitlines()
        
        current_rule = None
        current_alternatives = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Check for rule definition: <rule> ::=
            match = re.match(r'^<([^>]+)>\s*::=\s*(.*)$', line)
            if match:
                if current_rule:
                    self.rules[current_rule] = current_alternatives
                    self.non_terminals.add(f"<{current_rule}>")
                
                current_rule = match.group(1)
                current_alternatives = []
                rhs = match.group(2).strip()
                if rhs:
                    self._parse_rhs(rhs, current_alternatives)
            elif current_rule and line.startswith('|'):
                # Alternative for current rule
                rhs = line[1:].strip()
                self._parse_rhs(rhs, current_alternatives)
            elif current_rule and not line.startswith('<'):
                # Continuation line
                self._parse_rhs(line, current_alternatives[-1])
    
    def _parse_rhs(self, rhs: str, target_list):
        """Parse right-hand side of grammar rule"""
        # Split by spaces but keep quoted strings and <> together
        tokens = []
        in_quote = False
        in_angle = False
        current = ""
        
        for char in rhs:
            if char == '"' and not in_angle:
                in_quote = not in_quote
                current += char
                if not in_quote:
                    tokens.append(current)
                    current = ""
            elif char == '<' and not in_quote:
                if current and not in_angle:
                    tokens.append(current.strip())
                in_angle = True
                current = "<"
            elif char == '>' and in_angle and not in_quote:
                current += char
                tokens.append(current)
                current = ""
                in_angle = False
            elif char == ' ' and not in_quote and not in_angle:
                if current:
                    tokens.append(current.strip())
                    current = ""
            else:
                current += char
        
        if current:
            tokens.append(current.strip())
        
        # Filter empty tokens and clean up
        tokens = [t for t in tokens if t and t != '|']
        
        # Group tokens into production alternatives
        alt = []
        for token in tokens:
            if token == '|':
                if alt:
                    target_list.append(alt)
                    alt = []
            else:
                # Classify as terminal or non-terminal
                if token.startswith('<') and token.endswith('>'):
                    self.non_terminals.add(token)
                    alt.append(('nonterm', token[1:-1]))
                elif token.startswith('"') and token.endswith('"'):
                    term = token[1:-1]
                    self.terminals.add(term)
                    alt.append(('term', term))
                else:
                    # Keywords and operators (treated as terminals)
                    self.terminals.add(token)
                    alt.append(('term', token))
        
        if alt:
            target_list.append(alt)

class ASTAnalyzer:
    """Analyze parser source code to understand coverage targets"""
    
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.functions: Set[str] = set()
        self.classes: Set[str] = set()
        self.branches: List[Tuple[str, int]] = []
        self._analyze_code()
    
    def _analyze_code(self):
        """Analyze parser source files for functions, classes, and control flow"""
        files = ['parser.py', 'tokenizer.py', 'ast_nodes.py']
        
        for file in files:
            path = os.path.join(self.engine_path, file)
            if not os.path.exists(path):
                continue
            
            with open(path, 'r') as f:
                content = f.read()
            
            try:
                tree = ast.parse(content)
                self._extract_elements(tree, file)
            except SyntaxError:
                # Fall back to simple regex parsing
                self._regex_extract(content, file)
    
    def _extract_elements(self, tree: ast.AST, filename: str):
        """Extract functions, classes, and branches from AST"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self.functions.add(f"{filename}:{node.name}")
                # Count branches (if/elif/else, loops, try/except)
                self._count_branches(node, filename)
            elif isinstance(node, ast.ClassDef):
                self.classes.add(f"{filename}:{node.name}")
    
    def _count_branches(self, node: ast.AST, filename: str):
        """Count control flow branches in a function"""
        branch_nodes = (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler)
        
        for child in ast.walk(node):
            if isinstance(child, ast.If):
                # Each if creates at least 2 branches (true/false)
                self.branches.append((filename, child.lineno))
            elif isinstance(child, ast.Try):
                # Try block with except creates branches
                self.branches.append((filename, child.lineno))
    
    def _regex_extract(self, content: str, filename: str):
        """Fallback regex extraction when AST parsing fails"""
        # Extract function definitions
        func_pattern = r'^\s*def\s+(\w+)\s*\('
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            self.functions.add(f"{filename}:{match.group(1)}")
        
        # Extract class definitions
        class_pattern = r'^\s*class\s+(\w+)'
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            self.classes.add(f"{filename}:{match.group(1)}")
        
        # Count if statements as branches
        if_pattern = r'^\s*if\s+[^:]+:'
        for match in re.finditer(if_pattern, content, re.MULTILINE):
            # Approximate line number by counting newlines
            line_num = content[:match.start()].count('\n') + 1
            self.branches.append((filename, line_num))

class SQLGenerator:
    """Generate SQL statements from grammar"""
    
    def __init__(self, grammar: SQLGrammar, analyzer: ASTAnalyzer):
        self.grammar = grammar
        self.analyzer = analyzer
        self.max_depth = 15
        self.max_statements = 100
        self.cache: Dict[str, List[str]] = {}
        
        # Priority rules based on AST analysis
        self.priority_rules = self._identify_priority_rules()
    
    def _identify_priority_rules(self) -> Set[str]:
        """Identify grammar rules that likely correspond to important parser functions"""
        priority = {
            'select_stmt', 'expr', 'where_clause', 'join_clause',
            'function_call', 'subquery', 'create_table_stmt',
            'insert_stmt', 'update_stmt', 'delete_stmt',
            'with_clause', 'order_by_clause', 'group_by_clause',
            'having_clause', 'limit_clause'
        }
        
        # Add rules mentioned in function names
        for func in self.analyzer.functions:
            for rule in priority:
                if rule in func.lower():
                    priority.add(rule)
        
        return priority
    
    def generate_from_rule(self, rule: str, depth: int = 0) -> List[str]:
        """Generate strings from a grammar rule"""
        if depth > self.max_depth:
            return ['']
        
        if rule in self.cache:
            return self.cache[rule]
        
        if rule not in self.grammar.rules:
            return ['']
        
        alternatives = self.grammar.rules[rule]
        results = []
        
        # Prefer alternatives with more terminals for early depth
        if depth < 3:
            alternatives = sorted(alternatives, 
                                key=lambda alt: sum(1 for t, _ in alt if t == 'term'),
                                reverse=True)
        
        # Try up to 3 alternatives
        for alt in alternatives[:3]:
            alt_results = ['']
            for token_type, value in alt:
                if token_type == 'term':
                    # Terminal token
                    for i in range(len(alt_results)):
                        alt_results[i] += (' ' if alt_results[i] else '') + value
                else:
                    # Non-terminal - expand it
                    sub_results = self.generate_from_rule(value, depth + 1)
                    new_alt_results = []
                    for prefix in alt_results:
                        for suffix in sub_results:
                            if suffix:
                                new_str = prefix + (' ' if prefix else '') + suffix
                                new_alt_results.append(new_str)
                    alt_results = new_alt_results
            
            results.extend(alt_results)
        
        # Filter empty and deduplicate
        results = [r.strip() for r in results if r.strip()]
        results = list(set(results))
        
        self.cache[rule] = results[:10]  # Cache up to 10 results
        return results[:10]
    
    def generate_diverse_statements(self) -> List[str]:
        """Generate diverse SQL statements targeting parser coverage"""
        statements = []
        
        # 1. Start with basic statements from important rules
        start_rules = ['sql_stmt', 'select_stmt', 'create_table_stmt', 
                      'insert_stmt', 'update_stmt', 'delete_stmt']
        
        for rule in start_rules:
            if rule in self.grammar.rules:
                stmts = self.generate_from_rule(rule)
                statements.extend(stmts[:5])  # Take up to 5 from each
        
        # 2. Generate statements with various clauses
        clause_combinations = [
            ("SELECT * FROM t", []),
            ("SELECT * FROM t WHERE x = 1", ["where_clause"]),
            ("SELECT * FROM t ORDER BY x", ["order_by_clause"]),
            ("SELECT * FROM t GROUP BY x", ["group_by_clause"]),
            ("SELECT * FROM t LIMIT 10", ["limit_clause"]),
            ("SELECT * FROM t WHERE x = 1 ORDER BY y", ["where_clause", "order_by_clause"]),
            ("SELECT * FROM t GROUP BY x HAVING COUNT(*) > 1", ["group_by_clause", "having_clause"]),
            ("WITH cte AS (SELECT 1) SELECT * FROM cte", ["with_clause"]),
        ]
        
        for base, clauses in clause_combinations:
            statements.append(base)
        
        # 3. Generate statements with joins
        join_types = ["INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN", "CROSS JOIN"]
        for join_type in join_types:
            stmt = f"SELECT * FROM t1 {join_type} t2 ON t1.id = t2.id"
            statements.append(stmt)
        
        # 4. Generate statements with subqueries
        subquery_positions = [
            "SELECT * FROM (SELECT * FROM t) AS sub",
            "SELECT * FROM t WHERE x IN (SELECT y FROM s)",
            "SELECT (SELECT COUNT(*) FROM s) AS count FROM t",
            "SELECT * FROM t WHERE EXISTS (SELECT 1 FROM s WHERE s.id = t.id)",
        ]
        statements.extend(subquery_positions)
        
        # 5. Generate statements with functions
        functions = [
            "COUNT(*)", "SUM(x)", "AVG(x)", "MIN(x)", "MAX(x)",
            "UPPER(name)", "LOWER(name)", "SUBSTR(name, 1, 3)",
            "COALESCE(x, 0)", "NULLIF(x, y)", "CASE WHEN x > 0 THEN 'pos' ELSE 'neg' END"
        ]
        for func in functions[:5]:  # Use first 5 functions
            stmt = f"SELECT {func} FROM t"
            statements.append(stmt)
        
        # 6. Generate DDL statements
        ddl_statements = [
            "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, value REAL)",
            "CREATE TABLE t (id INT, name VARCHAR(255), created DATETIME)",
            "DROP TABLE t",
            "CREATE INDEX idx_name ON t(name)",
            "ALTER TABLE t ADD COLUMN new_col TEXT",
        ]
        statements.extend(ddl_statements)
        
        # 7. Generate DML statements
        dml_statements = [
            "INSERT INTO t (id, name) VALUES (1, 'test')",
            "INSERT INTO t VALUES (1, 'test', 3.14)",
            "UPDATE t SET name = 'updated' WHERE id = 1",
            "DELETE FROM t WHERE id = 1",
        ]
        statements.extend(dml_statements)
        
        # 8. Generate statements with complex expressions
        complex_exprs = [
            "SELECT 1 + 2 * 3 FROM t",
            "SELECT (x + y) * z FROM t",
            "SELECT x BETWEEN 1 AND 10 FROM t",
            "SELECT x LIKE '%test%' FROM t",
            "SELECT x IS NULL, y IS NOT NULL FROM t",
            "SELECT x AND y OR z FROM t",
            "SELECT NOT x FROM t",
        ]
        statements.extend(complex_exprs)
        
        # 9. Generate statements targeting specific parser functions
        # Based on function names from AST analysis
        for func_name in list(self.analyzer.functions)[:10]:
            if 'parse' in func_name.lower():
                # Extract rule name from parse function
                match = re.search(r'parse_([a-z_]+)', func_name.lower())
                if match:
                    rule = match.group(1)
                    if rule in self.grammar.rules:
                        stmts = self.generate_from_rule(rule)
                        if stmts:
                            statements.append(stmts[0])
        
        # 10. Add edge cases and special syntax
        edge_cases = [
            "SELECT DISTINCT * FROM t",
            "SELECT ALL * FROM t",
            "SELECT * FROM t UNION SELECT * FROM s",
            "SELECT * FROM t INTERSECT SELECT * FROM s",
            "SELECT * FROM t EXCEPT SELECT * FROM s",
            "SELECT * FROM t1, t2, t3",
            "SELECT * FROM t WHERE x IN (1, 2, 3)",
            "SELECT * FROM t WHERE x NOT IN (1, 2, 3)",
            "SELECT * FROM t ORDER BY x ASC, y DESC",
            "SELECT * FROM t LIMIT 10 OFFSET 5",
            "SELECT * FROM t FETCH FIRST 10 ROWS ONLY",
            "SELECT * FROM t FOR UPDATE",
            "BEGIN TRANSACTION",
            "COMMIT",
            "ROLLBACK",
        ]
        statements.extend(edge_cases)
        
        # Ensure we don't exceed max statements
        statements = list(set(statements))[:self.max_statements]
        
        # Add semicolons if not present
        for i in range(len(statements)):
            if not statements[i].strip().endswith(';'):
                statements[i] = statements[i].strip() + ';'
        
        return statements

class Solution:
    def solve(self, resources_path: str) -> list[str]:
        """
        Return SQL test cases designed to maximize parser coverage.
        """
        # Parse grammar
        grammar_file = os.path.join(resources_path, "sql_grammar.txt")
        if not os.path.exists(grammar_file):
            # Fallback to generate basic SQL if grammar file not found
            return self._generate_fallback_sql()
        
        grammar = SQLGrammar(grammar_file)
        
        # Analyze parser source code
        engine_path = os.path.join(resources_path, "sql_engine")
        analyzer = ASTAnalyzer(engine_path)
        
        # Generate SQL statements
        generator = SQLGenerator(grammar, analyzer)
        statements = generator.generate_diverse_statements()
        
        # Ensure we have valid SQL (basic validation)
        valid_statements = []
        for stmt in statements:
            # Basic syntax check
            if not stmt or len(stmt.strip()) < 5:
                continue
            
            # Check for common SQL keywords
            stmt_upper = stmt.upper()
            if any(keyword in stmt_upper for keyword in 
                   ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'WITH']):
                valid_statements.append(stmt)
            elif stmt_upper.startswith(('BEGIN', 'COMMIT', 'ROLLBACK')):
                valid_statements.append(stmt)
        
        # If no valid statements generated, use fallback
        if not valid_statements:
            return self._generate_fallback_sql()
        
        # Limit to reasonable number (balance coverage vs efficiency)
        return valid_statements[:80]
    
    def _generate_fallback_sql(self) -> list[str]:
        """Fallback SQL generation if grammar parsing fails"""
        return [
            "SELECT 1;",
            "SELECT * FROM users;",
            "SELECT id, name FROM users WHERE active = 1;",
            "SELECT COUNT(*) FROM users;",
            "SELECT name, COUNT(*) FROM users GROUP BY name HAVING COUNT(*) > 1;",
            "SELECT * FROM users ORDER BY name DESC;",
            "SELECT * FROM users LIMIT 10;",
            "SELECT u.*, o.amount FROM users u INNER JOIN orders o ON u.id = o.user_id;",
            "SELECT * FROM users WHERE id IN (SELECT user_id FROM orders);",
            "INSERT INTO users (id, name) VALUES (1, 'Alice');",
            "UPDATE users SET name = 'Bob' WHERE id = 1;",
            "DELETE FROM users WHERE id = 1;",
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER);",
            "DROP TABLE users;",
            "WITH recent_orders AS (SELECT * FROM orders WHERE date > '2023-01-01') SELECT * FROM recent_orders;",
            "SELECT * FROM users WHERE name LIKE 'A%';",
            "SELECT * FROM users WHERE age BETWEEN 18 AND 65;",
            "SELECT COALESCE(name, 'Unknown') FROM users;",
            "SELECT * FROM users WHERE name IS NOT NULL;",
            "BEGIN TRANSACTION; UPDATE accounts SET balance = balance - 100 WHERE id = 1; COMMIT;",
        ]