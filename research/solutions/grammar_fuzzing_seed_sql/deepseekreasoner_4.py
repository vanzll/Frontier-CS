import os
import re
import random
import itertools
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional
import ast
import inspect

class GrammarParser:
    """Parser for BNF-style SQL grammar"""
    
    def __init__(self, grammar_path: str):
        self.rules = {}
        self.start_symbols = set()
        self.parse_grammar(grammar_path)
        
    def parse_grammar(self, grammar_path: str):
        """Parse BNF grammar file"""
        with open(grammar_path, 'r') as f:
            lines = f.readlines()
            
        current_rule = None
        current_productions = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Check for rule definition
            match = re.match(r'<([^>]+)>', line)
            if match:
                if current_rule:
                    self.rules[current_rule] = current_productions
                current_rule = match.group(1)
                current_productions = []
                # Remove rule name from line
                line = line[match.end():].strip()
                if '::=' in line:
                    line = line.split('::=', 1)[1].strip()
            
            # Split by '|' for alternatives
            if '|' in line:
                parts = [p.strip() for p in line.split('|')]
                for part in parts:
                    if part:
                        current_productions.append(self.parse_production(part))
            elif line:
                current_productions.append(self.parse_production(line))
        
        if current_rule:
            self.rules[current_rule] = current_productions
            
        # Find start symbols (rules not used in RHS of other rules)
        all_references = set()
        for prods in self.rules.values():
            for prod in prods:
                for token in prod:
                    if isinstance(token, tuple) and token[0] == 'nonterm':
                        all_references.add(token[1])
        
        for rule in self.rules:
            if rule not in all_references:
                self.start_symbols.add(rule)
                
    def parse_production(self, prod_str: str) -> List[Tuple[str, str]]:
        """Parse a single production into tokens"""
        tokens = []
        i = 0
        while i < len(prod_str):
            if prod_str[i] == '<':
                # Non-terminal
                j = prod_str.find('>', i)
                if j != -1:
                    tokens.append(('nonterm', prod_str[i+1:j]))
                    i = j + 1
                else:
                    i += 1
            elif prod_str[i] == '[':
                # Optional group
                j = prod_str.find(']', i)
                if j != -1:
                    content = prod_str[i+1:j]
                    # Parse recursively and mark as optional
                    optional_tokens = self.parse_production(content)
                    tokens.append(('optional', optional_tokens))
                    i = j + 1
                else:
                    i += 1
            elif prod_str[i] == '{':
                # Zero-or-more group
                j = prod_str.find('}', i)
                if j != -1:
                    content = prod_str[i+1:j]
                    repeat_tokens = self.parse_production(content)
                    tokens.append(('repeat', repeat_tokens))
                    i = j + 1
                else:
                    i += 1
            elif prod_str[i] == '(':
                # Group
                j = prod_str.find(')', i)
                if j != -1:
                    content = prod_str[i+1:j]
                    group_tokens = self.parse_production(content)
                    tokens.append(('group', group_tokens))
                    i = j + 1
                else:
                    i += 1
            elif prod_str[i] == '|':
                # Alternative (should be handled at higher level)
                break
            elif prod_str[i].isspace():
                i += 1
            else:
                # Terminal
                j = i
                while j < len(prod_str) and not prod_str[j].isspace() and prod_str[j] not in '<>[]{}()|':
                    j += 1
                terminal = prod_str[i:j].strip()
                if terminal:
                    tokens.append(('term', terminal))
                i = j
        
        return tokens

class StatementGenerator:
    """Generate SQL statements from grammar"""
    
    def __init__(self, grammar_parser: GrammarParser):
        self.grammar = grammar_parser
        self.terminal_values = self._init_terminal_values()
        self.max_depth = 5
        self.max_recursion = 3
        
    def _init_terminal_values(self) -> Dict[str, List[str]]:
        """Initialize possible values for terminals"""
        return {
            'identifier': ['id', 'name', 'value', 'table1', 'table2', 'column1', 'column2', 
                          'col1', 'col2', 'col3', 't1', 't2', 'a', 'b', 'c', 'x', 'y', 'z'],
            'string_literal': ["'text'", "'value'", "'name'", "'test'", "'data'", "'example'"],
            'number': ['1', '2', '3', '10', '100', '0', 'NULL'],
            'function_name': ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'LENGTH', 'UPPER', 'LOWER'],
            'column_name': ['col1', 'col2', 'col3', 'id', 'name', 'value', 'amount', 'date'],
            'table_name': ['table1', 'table2', 'users', 'orders', 'products', 'items'],
            'data_type': ['INTEGER', 'VARCHAR(255)', 'TEXT', 'DATE', 'BOOLEAN', 'FLOAT'],
            'operator': ['=', '<>', '>', '<', '>=', '<=', 'LIKE', 'IN', 'BETWEEN', 'IS'],
            'join_type': ['INNER', 'LEFT', 'RIGHT', 'FULL', 'CROSS'],
            'sort_order': ['ASC', 'DESC'],
            'boolean': ['TRUE', 'FALSE']
        }
    
    def expand_token(self, token: Tuple[str, str], depth: int = 0, 
                    recursion_count: Dict[str, int] = None) -> str:
        """Expand a token into string"""
        if recursion_count is None:
            recursion_count = defaultdict(int)
            
        token_type, token_value = token
        
        if token_type == 'term':
            # Return terminal or choose from possible values
            if token_value in self.terminal_values:
                return random.choice(self.terminal_values[token_value])
            return token_value
            
        elif token_type == 'nonterm':
            # Handle recursion limit
            if recursion_count[token_value] >= self.max_recursion:
                # Return simplest production
                prods = self.grammar.rules.get(token_value, [])
                if prods:
                    # Find shortest production
                    shortest = min(prods, key=lambda p: len(p))
                    return self.expand_production(shortest, depth + 1, recursion_count)
                return ''
            
            recursion_count[token_value] += 1
            prods = self.grammar.rules.get(token_value, [])
            if not prods:
                return ''
            
            # Choose production based on depth
            if depth >= self.max_depth:
                # Choose simpler productions at high depth
                simple_prods = [p for p in prods if len(p) < 3]
                prod = random.choice(simple_prods) if simple_prods else random.choice(prods)
            else:
                prod = random.choice(prods)
                
            result = self.expand_production(prod, depth + 1, recursion_count)
            recursion_count[token_value] -= 1
            return result
            
        elif token_type == 'optional':
            # 50% chance to include optional part
            if random.random() < 0.5 and depth < self.max_depth:
                return self.expand_production(token_value, depth + 1, recursion_count)
            return ''
            
        elif token_type == 'repeat':
            # Repeat 0-2 times
            repeats = random.randint(0, 2)
            parts = []
            for _ in range(repeats):
                if depth < self.max_depth:
                    parts.append(self.expand_production(token_value, depth + 1, recursion_count))
            return ' '.join(parts)
            
        elif token_type == 'group':
            return self.expand_production(token_value, depth + 1, recursion_count)
            
        return ''
    
    def expand_production(self, production: List[Tuple[str, str]], depth: int = 0,
                         recursion_count: Dict[str, int] = None) -> str:
        """Expand a production into string"""
        if recursion_count is None:
            recursion_count = defaultdict(int)
            
        parts = []
        for token in production:
            if depth >= self.max_depth and token[0] == 'nonterm':
                # Skip complex expansions at max depth
                continue
            expanded = self.expand_token(token, depth, recursion_count)
            if expanded:
                parts.append(expanded)
        
        result = ' '.join(parts)
        # Clean up extra spaces
        result = re.sub(r'\s+', ' ', result).strip()
        result = re.sub(r'\s*([,;()])\s*', r'\1 ', result).strip()
        result = re.sub(r'\s+$', '', result)
        
        return result
    
    def generate_statement(self, start_symbol: str = None) -> str:
        """Generate a single SQL statement"""
        if start_symbol is None:
            start_symbol = random.choice(list(self.grammar.start_symbols))
            
        productions = self.grammar.rules.get(start_symbol, [])
        if not productions:
            return ''
            
        # Choose production with preference for complex ones
        weights = [min(len(p), 10) for p in productions]
        prod = random.choices(productions, weights=weights, k=1)[0]
        
        statement = self.expand_production(prod)
        
        # Ensure statement ends properly
        if not statement.endswith(';'):
            statement += ';'
            
        return statement
    
    def generate_diverse_statements(self, count: int) -> List[str]:
        """Generate diverse SQL statements covering different patterns"""
        statements = []
        patterns = set()
        
        # Try to generate statements from different start symbols
        start_symbols = list(self.grammar.start_symbols)
        start_symbols.sort()  # For deterministic behavior
        
        for i in range(count):
            # Cycle through start symbols
            start_symbol = start_symbols[i % len(start_symbols)]
            
            # Generate with variations
            for attempt in range(10):  # Try up to 10 times to get unique pattern
                stmt = self.generate_statement(start_symbol)
                if not stmt:
                    continue
                    
                # Create pattern by replacing identifiers with placeholders
                pattern = re.sub(r'\b(table\d+|col\d+|id\d*|name\d*)\b', '{id}', stmt)
                pattern = re.sub(r"'[^']*'", "'{str}'", pattern)
                pattern = re.sub(r'\b\d+\b', '{num}', pattern)
                
                if pattern not in patterns or len(patterns) < len(start_symbols):
                    statements.append(stmt)
                    patterns.add(pattern)
                    break
            else:
                # Fallback to any statement
                stmt = self.generate_statement(start_symbol)
                if stmt:
                    statements.append(stmt)
        
        return statements[:count]

class ParserAnalyzer:
    """Analyze parser source code to identify coverage targets"""
    
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.import_parser()
        
    def import_parser(self):
        """Import the parser module"""
        import sys
        sys.path.insert(0, self.engine_path)
        
        try:
            from sql_engine import parser
            from sql_engine import tokenizer
            from sql_engine import ast_nodes
            self.parser_module = parser
            self.tokenizer_module = tokenizer
            self.ast_nodes_module = ast_nodes
        except ImportError:
            # Create dummy modules for analysis
            self.parser_module = None
            self.tokenizer_module = None
            self.ast_nodes_module = None
    
    def analyze_coverage_targets(self) -> Dict[str, Set[str]]:
        """Analyze parser source to identify code patterns to target"""
        targets = {
            'keywords': set(),
            'clauses': set(),
            'functions': set(),
            'operators': set(),
            'data_types': set(),
            'joins': set()
        }
        
        # Analyze parser.py for patterns
        if self.parser_module and hasattr(self.parser_module, '__file__'):
            with open(self.parser_module.__file__, 'r') as f:
                parser_code = f.read()
            
            # Look for parse_ methods to understand what can be parsed
            parse_methods = re.findall(r'def parse_(\w+)', parser_code)
            targets['clauses'].update(parse_methods)
            
            # Look for token checks
            token_checks = re.findall(r'token\.type\s*==\s*["\'](\w+)["\']', parser_code)
            targets['keywords'].update(token_checks)
            
            # Look for function calls in expressions
            func_calls = re.findall(r'parse_function_call', parser_code)
            if func_calls:
                targets['functions'].add('functions')
        
        # Common SQL patterns to target based on typical parsers
        targets['keywords'].update([
            'SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'HAVING', 'ORDER',
            'INSERT', 'INTO', 'VALUES', 'UPDATE', 'SET', 'DELETE',
            'JOIN', 'INNER', 'LEFT', 'RIGHT', 'FULL', 'OUTER', 'ON',
            'UNION', 'ALL', 'INTERSECT', 'EXCEPT', 'DISTINCT',
            'AND', 'OR', 'NOT', 'NULL', 'TRUE', 'FALSE',
            'LIMIT', 'OFFSET', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END',
            'EXISTS', 'IN', 'BETWEEN', 'LIKE', 'IS'
        ])
        
        targets['clauses'].update([
            'select', 'from', 'where', 'group_by', 'having', 'order_by',
            'limit', 'offset', 'join', 'subquery', 'expression',
            'function_call', 'case_expression', 'between_expression'
        ])
        
        targets['functions'].update([
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'COALESCE', 'NULLIF',
            'UPPER', 'LOWER', 'SUBSTR', 'LENGTH', 'TRIM'
        ])
        
        targets['operators'].update([
            '=', '<>', '!=', '>', '<', '>=', '<=', 'LIKE', 'NOT LIKE',
            'IN', 'NOT IN', 'BETWEEN', 'NOT BETWEEN', 'IS', 'IS NOT',
            'AND', 'OR', 'NOT', '+', '-', '*', '/', '%', '||'
        ])
        
        targets['data_types'].update([
            'INTEGER', 'INT', 'SMALLINT', 'BIGINT', 'NUMERIC', 'DECIMAL',
            'REAL', 'DOUBLE', 'FLOAT', 'CHAR', 'VARCHAR', 'TEXT',
            'DATE', 'TIME', 'TIMESTAMP', 'BOOLEAN', 'BLOB'
        ])
        
        targets['joins'].update([
            'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN', 'CROSS JOIN',
            'LEFT OUTER JOIN', 'RIGHT OUTER JOIN', 'FULL OUTER JOIN'
        ])
        
        return targets

class EnhancedGenerator(StatementGenerator):
    """Enhanced generator that targets specific parser coverage"""
    
    def __init__(self, grammar_parser: GrammarParser, coverage_targets: Dict[str, Set[str]]):
        super().__init__(grammar_parser)
        self.coverage_targets = coverage_targets
        self.generated_patterns = set()
        
    def generate_targeted_statements(self, count: int) -> List[str]:
        """Generate statements targeting specific coverage areas"""
        statements = []
        
        # Generate statements for different SQL statement types
        statement_types = [
            ('select_statement', self._generate_select),
            ('insert_statement', self._generate_insert),
            ('update_statement', self._generate_update),
            ('delete_statement', self._generate_delete),
            ('create_table_statement', self._generate_create_table),
            ('drop_table_statement', self._generate_drop_table),
            ('alter_table_statement', self._generate_alter_table),
            ('with_statement', self._generate_with)
        ]
        
        # Distribute count among statement types
        base_count = max(1, count // len(statement_types))
        remainder = count % len(statement_types)
        
        for i, (stmt_type, generator) in enumerate(statement_types):
            stmt_count = base_count + (1 if i < remainder else 0)
            for _ in range(stmt_count):
                stmt = generator()
                if stmt:
                    statements.append(stmt)
        
        # Fill remaining slots with diverse statements
        while len(statements) < count:
            stmt = self.generate_statement()
            if stmt and stmt not in statements:
                statements.append(stmt)
        
        return statements[:count]
    
    def _generate_select(self) -> str:
        """Generate SELECT statements with various features"""
        templates = [
            # Basic SELECT
            "SELECT {cols} FROM {table};",
            # SELECT with WHERE
            "SELECT {cols} FROM {table} WHERE {condition};",
            # SELECT with JOIN
            "SELECT {cols} FROM {table1} {join} {table2} ON {join_condition};",
            # SELECT with GROUP BY
            "SELECT {cols} FROM {table} GROUP BY {group_cols} HAVING {having_condition};",
            # SELECT with ORDER BY
            "SELECT {cols} FROM {table} ORDER BY {order_cols};",
            # SELECT with LIMIT
            "SELECT {cols} FROM {table} LIMIT {limit};",
            # SELECT DISTINCT
            "SELECT DISTINCT {cols} FROM {table};",
            # SELECT with subquery
            "SELECT {cols} FROM (SELECT * FROM {sub_table}) AS sub;",
            # SELECT with CASE
            "SELECT CASE WHEN {condition} THEN {true_val} ELSE {false_val} END FROM {table};",
            # SELECT with UNION
            "SELECT {cols} FROM {table1} UNION SELECT {cols} FROM {table2};",
            # SELECT with functions
            "SELECT {func}({col}) FROM {table};",
            # Complex SELECT
            "SELECT {cols}, {func}({col2}) FROM {table1} {join} {table2} ON {join_condition} WHERE {condition} GROUP BY {group_cols} HAVING {having_condition} ORDER BY {order_cols} LIMIT {limit};"
        ]
        
        # Choose template
        template = random.choice(templates)
        
        # Fill template
        replacements = {
            '{cols}': self._generate_column_list(),
            '{col}': random.choice(self.terminal_values['column_name']),
            '{col2}': random.choice(self.terminal_values['column_name']),
            '{table}': random.choice(self.terminal_values['table_name']),
            '{table1}': random.choice(self.terminal_values['table_name']),
            '{table2}': random.choice(self.terminal_values['table_name']),
            '{sub_table}': random.choice(self.terminal_values['table_name']),
            '{condition}': self._generate_condition(),
            '{join_condition}': f"{random.choice(self.terminal_values['column_name'])} = {random.choice(self.terminal_values['column_name'])}",
            '{group_cols}': self._generate_column_list(),
            '{having_condition}': self._generate_condition(),
            '{order_cols}': self._generate_column_list(),
            '{limit}': random.choice(['1', '10', '100']),
            '{true_val}': random.choice(["'yes'", '1', 'TRUE']),
            '{false_val}': random.choice(["'no'", '0', 'FALSE']),
            '{join}': random.choice(['JOIN', 'LEFT JOIN', 'INNER JOIN', 'RIGHT JOIN']),
            '{func}': random.choice(self.terminal_values['function_name'])
        }
        
        statement = template
        for key, value in replacements.items():
            statement = statement.replace(key, value)
            
        return statement
    
    def _generate_insert(self) -> str:
        """Generate INSERT statements"""
        templates = [
            "INSERT INTO {table} VALUES ({values});",
            "INSERT INTO {table} ({columns}) VALUES ({values});",
            "INSERT INTO {table} SELECT {cols} FROM {source_table};"
        ]
        
        template = random.choice(templates)
        
        replacements = {
            '{table}': random.choice(self.terminal_values['table_name']),
            '{columns}': self._generate_column_list(),
            '{values}': self._generate_value_list(),
            '{cols}': self._generate_column_list(),
            '{source_table}': random.choice(self.terminal_values['table_name'])
        }
        
        statement = template
        for key, value in replacements.items():
            statement = statement.replace(key, value)
            
        return statement
    
    def _generate_update(self) -> str:
        """Generate UPDATE statements"""
        templates = [
            "UPDATE {table} SET {column} = {value};",
            "UPDATE {table} SET {column} = {value} WHERE {condition};",
            "UPDATE {table} SET {set_clause};"
        ]
        
        template = random.choice(templates)
        
        set_clause = f"{random.choice(self.terminal_values['column_name'])} = {random.choice(['1', "'new_value'", 'NULL'])}"
        
        replacements = {
            '{table}': random.choice(self.terminal_values['table_name']),
            '{column}': random.choice(self.terminal_values['column_name']),
            '{value}': random.choice(['1', "'updated'", 'NULL', 'col2']),
            '{condition}': self._generate_condition(),
            '{set_clause}': set_clause
        }
        
        statement = template
        for key, value in replacements.items():
            statement = statement.replace(key, value)
            
        return statement
    
    def _generate_delete(self) -> str:
        """Generate DELETE statements"""
        templates = [
            "DELETE FROM {table};",
            "DELETE FROM {table} WHERE {condition};"
        ]
        
        template = random.choice(templates)
        
        replacements = {
            '{table}': random.choice(self.terminal_values['table_name']),
            '{condition}': self._generate_condition()
        }
        
        statement = template
        for key, value in replacements.items():
            statement = statement.replace(key, value)
            
        return statement
    
    def _generate_create_table(self) -> str:
        """Generate CREATE TABLE statements"""
        templates = [
            "CREATE TABLE {table} ({columns});",
            "CREATE TABLE IF NOT EXISTS {table} ({columns});"
        ]
        
        template = random.choice(templates)
        
        # Generate column definitions
        col_defs = []
        for i in range(random.randint(1, 4)):
            col_name = random.choice(['id', 'name', 'value', 'amount', 'date'])
            col_type = random.choice(self.terminal_values['data_type'])
            constraints = []
            if random.random() > 0.5:
                constraints.append('NOT NULL')
            if random.random() > 0.7:
                constraints.append('PRIMARY KEY')
            col_def = f"{col_name} {col_type}"
            if constraints:
                col_def += ' ' + ' '.join(constraints)
            col_defs.append(col_def)
        
        replacements = {
            '{table}': random.choice(self.terminal_values['table_name']),
            '{columns}': ', '.join(col_defs)
        }
        
        statement = template
        for key, value in replacements.items():
            statement = statement.replace(key, value)
            
        return statement
    
    def _generate_drop_table(self) -> str:
        """Generate DROP TABLE statements"""
        templates = [
            "DROP TABLE {table};",
            "DROP TABLE IF EXISTS {table};"
        ]
        
        template = random.choice(templates)
        
        replacements = {
            '{table}': random.choice(self.terminal_values['table_name'])
        }
        
        statement = template
        for key, value in replacements.items():
            statement = statement.replace(key, value)
            
        return statement
    
    def _generate_alter_table(self) -> str:
        """Generate ALTER TABLE statements"""
        templates = [
            "ALTER TABLE {table} ADD COLUMN {column} {type};",
            "ALTER TABLE {table} DROP COLUMN {column};",
            "ALTER TABLE {table} RENAME COLUMN {old_column} TO {new_column};"
        ]
        
        template = random.choice(templates)
        
        replacements = {
            '{table}': random.choice(self.terminal_values['table_name']),
            '{column}': random.choice(self.terminal_values['column_name']),
            '{old_column}': random.choice(self.terminal_values['column_name']),
            '{new_column}': random.choice(self.terminal_values['column_name']),
            '{type}': random.choice(self.terminal_values['data_type'])
        }
        
        statement = template
        for key, value in replacements.items():
            statement = statement.replace(key, value)
            
        return statement
    
    def _generate_with(self) -> str:
        """Generate WITH (CTE) statements"""
        templates = [
            "WITH cte AS (SELECT {cols} FROM {table}) SELECT * FROM cte;",
            "WITH cte1 AS (SELECT {cols} FROM {table1}), cte2 AS (SELECT {cols} FROM {table2}) SELECT * FROM cte1 UNION SELECT * FROM cte2;"
        ]
        
        template = random.choice(templates)
        
        replacements = {
            '{cols}': self._generate_column_list(),
            '{table}': random.choice(self.terminal_values['table_name']),
            '{table1}': random.choice(self.terminal_values['table_name']),
            '{table2}': random.choice(self.terminal_values['table_name'])
        }
        
        statement = template
        for key, value in replacements.items():
            statement = statement.replace(key, value)
            
        return statement
    
    def _generate_column_list(self) -> str:
        """Generate a list of columns"""
        count = random.randint(1, 4)
        columns = random.sample(self.terminal_values['column_name'], min(count, len(self.terminal_values['column_name'])))
        if random.random() > 0.7:
            columns.append('*')
        return ', '.join(columns)
    
    def _generate_value_list(self) -> str:
        """Generate a list of values"""
        count = random.randint(1, 4)
        values = []
        for _ in range(count):
            if random.random() > 0.5:
                values.append(random.choice(["'value'", "'text'", "'data'"]))
            else:
                values.append(random.choice(['1', '2', '3', 'NULL']))
        return ', '.join(values)
    
    def _generate_condition(self) -> str:
        """Generate a condition expression"""
        conditions = [
            f"{random.choice(self.terminal_values['column_name'])} = {random.choice(['1', "'value'", 'NULL'])}",
            f"{random.choice(self.terminal_values['column_name'])} > {random.choice(['0', '10', '100'])}",
            f"{random.choice(self.terminal_values['column_name'])} LIKE '{random.choice(['%text%', 'A%', '%Z'])}'",
            f"{random.choice(self.terminal_values['column_name'])} IN ({self._generate_value_list()})",
            f"{random.choice(self.terminal_values['column_name'])} BETWEEN 1 AND 100",
            f"{random.choice(self.terminal_values['column_name'])} IS NOT NULL",
            f"EXISTS (SELECT 1 FROM {random.choice(self.terminal_values['table_name'])})"
        ]
        
        # Combine conditions with AND/OR
        if random.random() > 0.7:
            cond1 = random.choice(conditions)
            cond2 = random.choice(conditions)
            operator = random.choice(['AND', 'OR'])
            return f"({cond1} {operator} {cond2})"
        else:
            return random.choice(conditions)

class Solution:
    def solve(self, resources_path: str) -> list[str]:
        """
        Return SQL test cases designed to maximize parser coverage.
        """
        random.seed(42)  # For reproducible results
        
        # Parse grammar
        grammar_path = os.path.join(resources_path, 'sql_grammar.txt')
        grammar_parser = GrammarParser(grammar_path)
        
        # Analyze parser for coverage targets
        engine_path = os.path.join(resources_path, 'sql_engine')
        analyzer = ParserAnalyzer(engine_path)
        coverage_targets = analyzer.analyze_coverage_targets()
        
        # Generate statements using enhanced generator
        generator = EnhancedGenerator(grammar_parser, coverage_targets)
        
        # Target around 40-50 statements for efficiency bonus
        target_count = 45
        
        # Generate targeted statements
        statements = generator.generate_targeted_statements(target_count)
        
        # Ensure we have valid SQL syntax
        valid_statements = []
        for stmt in statements:
            # Basic validation
            if not stmt or len(stmt) < 5:
                continue
            if not stmt.endswith(';'):
                stmt = stmt + ';'
            valid_statements.append(stmt)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_statements = []
        for stmt in valid_statements:
            if stmt not in seen:
                seen.add(stmt)
                unique_statements.append(stmt)
        
        # Limit to target count
        return unique_statements[:target_count]