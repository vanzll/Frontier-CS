import os
import random
import time
import hashlib
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional
import json
import re

class Solution:
    def solve(self, resources_path: str) -> dict:
        # Read grammar file
        grammar_path = os.path.join(resources_path, "sql_grammar.txt")
        with open(grammar_path, 'r') as f:
            grammar_text = f.read()
        
        # Generate fuzzer code
        fuzzer_code = self.generate_fuzzer_code(grammar_text)
        
        return {"code": fuzzer_code}
    
    def generate_fuzzer_code(self, grammar_text: str) -> str:
        return f'''
import random
import time
import hashlib
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional
import re
import itertools
import math

# Global state for the fuzzer
class FuzzerState:
    def __init__(self):
        # Grammar components
        self.keywords = [
            "SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "UPDATE", "SET",
            "DELETE", "CREATE", "TABLE", "DROP", "ALTER", "ADD", "COLUMN", "INDEX",
            "PRIMARY", "KEY", "FOREIGN", "REFERENCES", "UNIQUE", "CHECK", "DEFAULT",
            "NOT", "NULL", "AND", "OR", "BETWEEN", "IN", "LIKE", "IS", "EXISTS",
            "JOIN", "INNER", "LEFT", "RIGHT", "OUTER", "ON", "GROUP", "BY", "HAVING",
            "ORDER", "ASC", "DESC", "LIMIT", "OFFSET", "UNION", "ALL", "DISTINCT",
            "AS", "CASE", "WHEN", "THEN", "ELSE", "END", "CAST", "COALESCE",
            "COUNT", "SUM", "AVG", "MIN", "MAX", "BEGIN", "COMMIT", "ROLLBACK",
            "TRANSACTION", "SAVEPOINT", "RELEASE", "WITH", "RECURSIVE"
        ]
        
        self.functions = [
            "ABS", "ROUND", "CEIL", "FLOOR", "RANDOM", "UPPER", "LOWER", "LENGTH",
            "SUBSTR", "REPLACE", "TRIM", "DATE", "TIME", "DATETIME", "JULIANDAY",
            "STRFTIME", "CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP",
            "TYPEOF", "NULLIF", "ZEROBLOB", "HEX", "QUOTE", "SOUNDEX"
        ]
        
        self.data_types = [
            "INTEGER", "INT", "REAL", "TEXT", "BLOB", "NUMERIC", "BOOLEAN",
            "DATE", "DATETIME", "VARCHAR", "CHAR", "DECIMAL", "FLOAT", "DOUBLE"
        ]
        
        self.operators = ["=", "<>", "!=", "<", ">", "<=", ">=", "+", "-", "*", "/", "%", "||"]
        
        # Test corpus
        self.corpus = []
        self.corpus_hashes = set()
        
        # Coverage tracking
        self.seen_patterns = defaultdict(int)
        self.statement_types = defaultdict(int)
        self.depth_tracker = defaultdict(int)
        
        # Generation parameters
        self.statement_count = 0
        self.batch_size = 50
        self.max_depth = 5
        self.mutation_rate = 0.3
        
        # Edge case patterns
        self.edge_cases = [
            # Empty/whitespace
            "",
            " ",
            "  ",
            "\\t",
            "\\n",
            
            # Minimal valid statements
            "SELECT 1",
            "SELECT NULL",
            "SELECT *",
            "SELECT 1 WHERE 1",
            
            # Extreme values
            "SELECT {" + "9"*100 + "}",
            "SELECT '" + "A"*1000 + "'",
            
            # Special characters
            "SELECT '\\\\x00'",
            "SELECT '\\\\n\\\\t\\\\r'",
            "SELECT '''''",
            'SELECT """',
            
            # Complex nested
            "SELECT (SELECT (SELECT 1))",
            "SELECT CASE WHEN 1 THEN (SELECT 1) ELSE NULL END",
            
            # Various literals
            "SELECT 0xFF",
            "SELECT 1.23e-4",
            "SELECT 1.7976931348623157e308",
            "SELECT -1.7976931348623157e308",
            "SELECT 0",
            "SELECT -0",
            "SELECT +0",
            "SELECT 1/0",
            
            # Unusual but valid syntax
            "SELECT +1, -2, ~3",
            "SELECT 1 COLLATE BINARY",
            "SELECT 1 ESCAPE '\\\\'",
        ]
        
        # Initialize with edge cases
        for ec in self.edge_cases:
            self._add_to_corpus(ec)
    
    def _hash_statement(self, stmt: str) -> str:
        """Create a hash of statement for deduplication."""
        return hashlib.md5(stmt.encode()).hexdigest()
    
    def _add_to_corpus(self, stmt: str):
        """Add statement to corpus if not duplicate."""
        stmt_hash = self._hash_statement(stmt)
        if stmt_hash not in self.corpus_hashes:
            self.corpus.append(stmt)
            self.corpus_hashes.add(stmt_hash)
    
    def generate_select(self, depth: int = 0) -> str:
        """Generate a SELECT statement."""
        if depth > self.max_depth:
            return "SELECT 1"
        
        patterns = [
            # Simple select
            lambda: f"SELECT {{expr_list}} {{from_clause}} {{where_clause}} {{group_by}} {{order_by}} {{limit_clause}}",
            
            # Select with subquery
            lambda: f"SELECT ({{subquery}}) {{from_clause}}",
            
            # Compound select
            lambda: f"{{select}} UNION {{select}}",
            lambda: f"{{select}} UNION ALL {{select}}",
            lambda: f"{{select}} INTERSECT {{select}}",
            lambda: f"{{select}} EXCEPT {{select}}",
            
            # Select with JOINs
            lambda: f"SELECT * FROM {{table}} {{join_clause}} {{where_clause}}",
            
            # Select with window functions
            lambda: f"SELECT {{expr}}, ROW_NUMBER() OVER ({{window_spec}}) FROM {{table}}",
        ]
        
        pattern = random.choice(patterns)
        return self._expand_template(pattern(), depth + 1)
    
    def generate_insert(self, depth: int = 0) -> str:
        """Generate an INSERT statement."""
        patterns = [
            lambda: f"INSERT INTO {{table}} VALUES {{value_list}}",
            lambda: f"INSERT INTO {{table}} ({{column_list}}) VALUES {{value_list}}",
            lambda: f"INSERT INTO {{table}} SELECT {{expr_list}} FROM {{table2}}",
            lambda: f"REPLACE INTO {{table}} VALUES {{value_list}}",
        ]
        return self._expand_template(random.choice(patterns)(), depth + 1)
    
    def generate_update(self, depth: int = 0) -> str:
        """Generate an UPDATE statement."""
        return self._expand_template(
            f"UPDATE {{table}} SET {{assignment_list}} {{where_clause}}",
            depth + 1
        )
    
    def generate_delete(self, depth: int = 0) -> str:
        """Generate a DELETE statement."""
        return self._expand_template(
            f"DELETE FROM {{table}} {{where_clause}}",
            depth + 1
        )
    
    def generate_create(self, depth: int = 0) -> str:
        """Generate a CREATE statement."""
        patterns = [
            lambda: f"CREATE TABLE {{table}} ({{column_defs}})",
            lambda: f"CREATE TABLE {{table}} AS {{select}}",
            lambda: f"CREATE INDEX {{index_name}} ON {{table}} ({{column_list}})",
            lambda: f"CREATE UNIQUE INDEX {{index_name}} ON {{table}} ({{column_list}})",
            lambda: f"CREATE VIEW {{view_name}} AS {{select}}",
            lambda: f"CREATE TEMPORARY TABLE {{table}} ({{column_defs}})",
        ]
        return self._expand_template(random.choice(patterns)(), depth + 1)
    
    def generate_drop(self, depth: int = 0) -> str:
        """Generate a DROP statement."""
        patterns = [
            lambda: f"DROP TABLE {{table}}",
            lambda: f"DROP TABLE IF EXISTS {{table}}",
            lambda: f"DROP INDEX {{index_name}}",
            lambda: f"DROP VIEW {{view_name}}",
        ]
        return self._expand_template(random.choice(patterns)(), depth + 1)
    
    def generate_alter(self, depth: int = 0) -> str:
        """Generate an ALTER statement."""
        patterns = [
            lambda: f"ALTER TABLE {{table}} ADD COLUMN {{column_def}}",
            lambda: f"ALTER TABLE {{table}} DROP COLUMN {{column}}",
            lambda: f"ALTER TABLE {{table}} RENAME TO {{new_table}}",
            lambda: f"ALTER TABLE {{table}} RENAME COLUMN {{old_column}} TO {{new_column}}",
        ]
        return self._expand_template(random.choice(patterns)(), depth + 1)
    
    def generate_expression(self, depth: int = 0) -> str:
        """Generate a SQL expression."""
        if depth > 3:
            return self._literal()
        
        patterns = [
            # Literal
            lambda: self._literal(),
            
            # Column reference
            lambda: f"{{column}}",
            lambda: f"{{table}}.{{column}}",
            
            # Function call
            lambda: f"{{function}}({{expr_list}})",
            lambda: f"{{function}}(*)",
            lambda: f"{{function}}(DISTINCT {{expr}})",
            
            # Binary operation
            lambda: f"{{expr}} {{operator}} {{expr}}",
            
            # Unary operation
            lambda: f"{{unary_op}}{{expr}}",
            
            # Cast
            lambda: f"CAST({{expr}} AS {{data_type}})",
            
            # Case expression
            lambda: f"CASE {{when_clauses}} END",
            lambda: f"CASE {{expr}} {{when_clauses}} END",
            
            # Subquery expression
            lambda: f"({{select}})",
            lambda: f"EXISTS ({{select}})",
            lambda: f"{{expr}} IN ({{select}})",
            lambda: f"{{expr}} NOT IN ({{select}})",
            
            # Collate
            lambda: f"{{expr}} COLLATE BINARY",
            
            # Between
            lambda: f"{{expr}} BETWEEN {{expr}} AND {{expr}}",
            
            # Is NULL/NOT NULL
            lambda: f"{{expr}} IS NULL",
            lambda: f"{{expr}} IS NOT NULL",
        ]
        
        return self._expand_template(random.choice(patterns)(), depth + 1)
    
    def _literal(self) -> str:
        """Generate a literal value."""
        patterns = [
            # Numbers
            lambda: str(random.randint(-1000, 1000)),
            lambda: f"{random.uniform(-1000, 1000):.10f}",
            lambda: "NULL",
            lambda: "TRUE",
            lambda: "FALSE",
            
            # Strings
            lambda: f"'{self._random_string()}'",
            lambda: "''",
            lambda: "'\"'",
            lambda: "'\\\\''",
            
            # Blob
            lambda: f"X'{random.randbytes(4).hex()}'",
            
            # Current date/time
            lambda: "CURRENT_TIMESTAMP",
            lambda: "CURRENT_DATE",
            lambda: "CURRENT_TIME",
        ]
        
        # Occasionally generate extreme values
        if random.random() < 0.05:
            extreme = [
                "1e308",
                "-1e308",
                "1e-308",
                "0.0",
                "-0.0",
                "1.7976931348623157e308",
                "2.2250738585072014e-308",
                "Infinity",
                "-Infinity",
                "NaN",
            ]
            return random.choice(extreme)
        
        return random.choice(patterns)()
    
    def _random_string(self, max_len: int = 20) -> str:
        """Generate a random string."""
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-.,:;!?@#$%^&*()[]{}<>/\\\\'\"`~"
        length = random.randint(1, max_len)
        
        # Occasionally include special characters
        if random.random() < 0.2:
            special = "\\n\\t\\r\\x00\\x1F"
            result = []
            for _ in range(length):
                if random.random() < 0.1:
                    result.append(random.choice(special))
                else:
                    result.append(random.choice(chars))
            return ''.join(result)
        
        return ''.join(random.choice(chars) for _ in range(length))
    
    def _expand_template(self, template: str, depth: int = 0) -> str:
        """Expand a template with placeholders."""
        # Define expansion functions for each placeholder
        expansions = {
            "{select}": lambda: self.generate_select(depth),
            "{expr}": lambda: self.generate_expression(depth),
            "{expr_list}": lambda: self._comma_list(lambda: self.generate_expression(depth + 1), 1, 5),
            "{table}": lambda: f"t{random.randint(1, 10)}",
            "{table2}": lambda: f"t{random.randint(1, 10)}",
            "{column}": lambda: f"col{random.randint(1, 10)}",
            "{column_list}": lambda: self._comma_list(lambda: f"col{random.randint(1, 10)}", 1, 5),
            "{value_list}": lambda: self._paren_list(lambda: self.generate_expression(depth + 1), 1, 5),
            "{assignment_list}": lambda: self._comma_list(lambda: f"col{random.randint(1, 10)} = {self.generate_expression(depth + 1)}", 1, 3),
            "{where_clause}": lambda: f"WHERE {self.generate_expression(depth + 1)}" if random.random() > 0.3 else "",
            "{from_clause}": lambda: f"FROM {self._table_ref(depth + 1)}" if random.random() > 0.1 else "",
            "{group_by}": lambda: f"GROUP BY {self._comma_list(lambda: f'col{random.randint(1, 10)}', 1, 3)}" if random.random() > 0.7 else "",
            "{order_by}": lambda: f"ORDER BY {self._comma_list(lambda: f'col{random.randint(1, 10)}', 1, 3)}" if random.random() > 0.7 else "",
            "{limit_clause}": lambda: f"LIMIT {random.randint(1, 100)}" if random.random() > 0.8 else "",
            "{subquery}": lambda: self.generate_select(depth + 1),
            "{join_clause}": lambda: self._join_clause(depth + 1),
            "{window_spec}": lambda: f"PARTITION BY col{random.randint(1, 10)} ORDER BY col{random.randint(1, 10)}",
            "{column_defs}": lambda: self._comma_list(self._column_def, 1, 5),
            "{column_def}": lambda: f"col{random.randint(1, 10)} {random.choice(self.data_types)}",
            "{index_name}": lambda: f"idx_{random.randint(1, 100)}",
            "{view_name}": lambda: f"v{random.randint(1, 10)}",
            "{new_table}": lambda: f"t{random.randint(11, 20)}",
            "{old_column}": lambda: f"col{random.randint(1, 5)}",
            "{new_column}": lambda: f"col{random.randint(6, 10)}",
            "{function}": lambda: random.choice(self.functions),
            "{operator}": lambda: random.choice(self.operators),
            "{unary_op}": lambda: random.choice(["+", "-", "~", "NOT "]),
            "{data_type}": lambda: random.choice(self.data_types),
            "{when_clauses}": lambda: ' '.join([f"WHEN {self.generate_expression(depth + 1)} THEN {self.generate_expression(depth + 1)}" for _ in range(random.randint(1, 3))]) + f" ELSE {self.generate_expression(depth + 1)}",
        }
        
        # Replace placeholders
        result = template
        for placeholder, expand_func in expansions.items():
            while placeholder in result:
                result = result.replace(placeholder, expand_func(), 1)
        
        return result
    
    def _comma_list(self, item_func, min_items: int = 1, max_items: int = 3) -> str:
        """Generate a comma-separated list."""
        n = random.randint(min_items, max_items)
        return ', '.join(item_func() for _ in range(n))
    
    def _paren_list(self, item_func, min_items: int = 1, max_items: int = 3) -> str:
        """Generate a parenthesized list."""
        items = self._comma_list(item_func, min_items, max_items)
        return f"({items})"
    
    def _table_ref(self, depth: int) -> str:
        """Generate a table reference."""
        patterns = [
            lambda: f"t{random.randint(1, 10)}",
            lambda: f"({{select}})",
            lambda: f"t{random.randint(1, 10)} AS alias{random.randint(1, 10)}",
        ]
        return self._expand_template(random.choice(patterns)(), depth)
    
    def _join_clause(self, depth: int) -> str:
        """Generate a JOIN clause."""
        join_types = ["", "INNER", "LEFT", "LEFT OUTER", "RIGHT", "RIGHT OUTER", "CROSS", "NATURAL"]
        n_joins = random.randint(0, 3)
        
        result = []
        for _ in range(n_joins):
            join_type = random.choice(join_types)
            if join_type:
                result.append(f"{join_type} JOIN {self._table_ref(depth)} ON {self.generate_expression(depth + 1)}")
            else:
                result.append(f"JOIN {self._table_ref(depth)} ON {self.generate_expression(depth + 1)}")
        
        return ' '.join(result)
    
    def _column_def(self) -> str:
        """Generate a column definition."""
        base = f"col{random.randint(1, 10)} {random.choice(self.data_types)}"
        
        # Add constraints
        constraints = []
        if random.random() > 0.5:
            constraints.append("PRIMARY KEY")
        if random.random() > 0.5:
            constraints.append("NOT NULL")
        if random.random() > 0.7:
            constraints.append("UNIQUE")
        if random.random() > 0.8:
            constraints.append(f"DEFAULT {self._literal()}")
        if random.random() > 0.9:
            constraints.append(f"CHECK ({self.generate_expression(1)})")
        
        if constraints:
            base += ' ' + ' '.join(constraints)
        
        return base
    
    def mutate_statement(self, stmt: str) -> str:
        """Mutate a SQL statement."""
        if not stmt or random.random() < 0.2:
            return self._generate_random_statement()
        
        mutations = [
            self._mutate_keyword,
            self._mutate_literal,
            self._delete_part,
            self._insert_part,
            self._swap_parts,
            self._add_nesting,
            self._remove_nesting,
            self._change_operator,
            self._add_function,
            self._remove_function,
        ]
        
        # Apply 1-3 mutations
        result = stmt
        for _ in range(random.randint(1, 3)):
            if random.random() < self.mutation_rate:
                try:
                    result = random.choice(mutations)(result)
                except:
                    pass
        
        return result
    
    def _mutate_keyword(self, stmt: str) -> str:
        """Mutate a keyword in the statement."""
        words = stmt.split()
        if not words:
            return stmt
        
        for i, word in enumerate(words):
            if word.upper() in self.keywords:
                if random.random() > 0.5:
                    # Replace with another keyword
                    words[i] = random.choice(self.keywords)
                else:
                    # Delete the keyword
                    words[i] = ""
                break
        
        return ' '.join(filter(None, words))
    
    def _mutate_literal(self, stmt: str) -> str:
        """Mutate a literal value."""
        # Find numbers and strings
        import re
        
        # Replace numbers
        def replace_number(match):
            num = match.group()
            if random.random() < 0.5:
                return str(random.randint(-1000, 1000))
            else:
                return "NULL"
        
        # Replace strings
        def replace_string(match):
            if random.random() < 0.5:
                return f"'{self._random_string()}'"
            else:
                return "NULL"
        
        # Apply replacements
        result = re.sub(r'\\b\\d+\\.?\\d*\\b', replace_number, stmt)
        result = re.sub(r"'[^']*'", replace_string, result)
        
        return result
    
    def _delete_part(self, stmt: str) -> str:
        """Delete a part of the statement."""
        parts = re.split(r'(\\s+|[,()])', stmt)
        if len(parts) <= 3:
            return stmt
        
        # Delete 1-3 random parts
        to_delete = random.randint(1, min(3, len(parts) // 2))
        for _ in range(to_delete):
            idx = random.randint(0, len(parts) - 1)
            parts.pop(idx)
        
        return ''.join(parts)
    
    def _insert_part(self, stmt: str) -> str:
        """Insert a random part into the statement."""
        parts = stmt.split()
        if not parts:
            return random.choice(self.keywords)
        
        idx = random.randint(0, len(parts))
        insertions = [
            random.choice(self.keywords),
            self._literal(),
            "(" + self.generate_expression(1) + ")",
            random.choice(["+", "-", "*", "/", "||"]),
        ]
        
        parts.insert(idx, random.choice(insertions))
        return ' '.join(parts)
    
    def _swap_parts(self, stmt: str) -> str:
        """Swap two parts of the statement."""
        parts = stmt.split()
        if len(parts) < 2:
            return stmt
        
        i, j = random.sample(range(len(parts)), 2)
        parts[i], parts[j] = parts[j], parts[i]
        return ' '.join(parts)
    
    def _add_nesting(self, stmt: str) -> str:
        """Add nesting to the statement."""
        # Find a suitable place to add a subquery
        if "SELECT" in stmt.upper():
            # Already has SELECT, maybe wrap it
            return f"({stmt})"
        else:
            # Add a subquery expression
            return f"SELECT * FROM ({stmt})"
    
    def _remove_nesting(self, stmt: str) -> str:
        """Remove nesting from the statement."""
        # Remove outer parentheses
        if stmt.startswith('(') and stmt.endswith(')'):
            return stmt[1:-1]
        
        # Remove subqueries
        import re
        result = re.sub(r'\\(SELECT[^)]+\\)', '1', stmt, flags=re.IGNORECASE)
        return result
    
    def _change_operator(self, stmt: str) -> str:
        """Change an operator in the statement."""
        operators = ['=', '<>', '!=', '<', '>', '<=', '>=', 'LIKE', 'NOT LIKE', 'IN', 'NOT IN', 'IS', 'IS NOT']
        
        for op in operators:
            if op in stmt:
                new_op = random.choice(operators)
                return stmt.replace(op, new_op, 1)
        
        return stmt
    
    def _add_function(self, stmt: str) -> str:
        """Add a function call to the statement."""
        # Find a literal or column reference to wrap
        import re
        
        def wrap_match(match):
            return f"{random.choice(self.functions)}({match.group()})"
        
        # Try to wrap numbers, strings, or column names
        result = re.sub(r'\\b\\d+\\.?\\d*\\b', wrap_match, stmt)
        if result != stmt:
            return result
        
        result = re.sub(r"'[^']*'", wrap_match, stmt)
        if result != stmt:
            return result
        
        result = re.sub(r'\\bcol\\d+\\b', wrap_match, stmt)
        return result
    
    def _remove_function(self, stmt: str) -> str:
        """Remove a function call from the statement."""
        import re
        
        # Remove function calls, keeping their arguments
        def unwrap_match(match):
            # Extract argument from function(arg)
            content = match.group(1)
            if ',' in content:
                # Take first argument
                return content.split(',')[0].strip()
            return content
        
        result = re.sub(r'\\b[A-Z_]+\\s*\\(([^)]+)\\)', unwrap_match, stmt)
        return result
    
    def _generate_random_statement(self) -> str:
        """Generate a random SQL statement."""
        generators = [
            self.generate_select,
            self.generate_insert,
            self.generate_update,
            self.generate_delete,
            self.generate_create,
            self.generate_drop,
            self.generate_alter,
        ]
        
        weights = [0.4, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05]
        
        generator = random.choices(generators, weights=weights)[0]
        return generator()
    
    def generate_batch(self) -> List[str]:
        """Generate a batch of statements."""
        batch = []
        
        # Generate new statements
        n_new = int(self.batch_size * 0.7)
        for _ in range(n_new):
            stmt = self._generate_random_statement()
            batch.append(stmt)
            self._add_to_corpus(stmt)
        
        # Mutate existing statements
        n_mutate = self.batch_size - n_new
        if self.corpus and n_mutate > 0:
            for _ in range(n_mutate):
                parent = random.choice(self.corpus)
                mutated = self.mutate_statement(parent)
                batch.append(mutated)
                self._add_to_corpus(mutated)
        
        # Track statement types for adaptive generation
        for stmt in batch:
            stmt_upper = stmt.upper()
            for stmt_type in ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"]:
                if stmt_type in stmt_upper:
                    self.statement_types[stmt_type] += 1
                    break
        
        self.statement_count += len(batch)
        
        # Dynamically adjust batch size based on progress
        if self.statement_count > 1000:
            self.batch_size = min(100, self.batch_size)
        elif self.statement_count > 5000:
            self.batch_size = min(50, self.batch_size)
        
        return batch

# Global state instance
state = FuzzerState()

def fuzz(parse_sql):
    """
    Generate SQL statements and execute them through the parser.
    
    This function will be called repeatedly by the evaluator until the time
    budget (60 seconds) is exhausted. Each call should generate a batch of
    SQL statements and pass them to parse_sql for execution.
    """
    try:
        # Generate a batch of statements
        statements = state.generate_batch()
        
        # Execute through parser
        if statements:
            parse_sql(statements)
        
        # Always continue (let evaluator handle time limit)
        return True
        
    except Exception as e:
        # If any error occurs, still continue fuzzing
        return True
'''