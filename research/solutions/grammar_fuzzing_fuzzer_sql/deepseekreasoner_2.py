import os
import random
import time
import math
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict, deque
import re
import itertools
import copy

class Solution:
    def solve(self, resources_path: str) -> dict:
        fuzzer_code = self.generate_fuzzer_code(resources_path)
        return {"code": fuzzer_code}
    
    def generate_fuzzer_code(self, resources_path: str) -> str:
        """Generate the complete fuzzer code as a string."""
        return f'''
import os
import random
import time
import math
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict, deque
import itertools
import re
import copy

# Global state maintained across fuzz() calls
class FuzzerState:
    def __init__(self, grammar_path: str):
        self.start_time = time.time()
        self.time_budget = 60.0
        self.grammar = self.load_grammar(grammar_path)
        self.corpus = []
        self.coverage_weights = defaultdict(int)
        self.statement_counts = defaultdict(int)
        self.unique_statements = set()
        self.valid_statements = []
        self.invalid_statements = []
        self.mutation_weights = defaultdict(int)
        self.last_coverage = 0
        self.iteration = 0
        self.max_depth = 5
        self.batch_size = 50
        self.max_statement_length = 500
        self.structure_pool = []
        self.recent_coverage_gain = False
        self.coverage_gain_count = 0
        
        # Initialize with seed statements
        self.seed_statements = [
            "SELECT 1",
            "SELECT * FROM t",
            "SELECT a, b, c FROM t1, t2 WHERE x = y",
            "INSERT INTO t VALUES (1, 2, 3)",
            "UPDATE t SET a = 1 WHERE b = 2",
            "DELETE FROM t WHERE id = 1",
            "CREATE TABLE t (id INT, name TEXT)",
            "DROP TABLE IF EXISTS t",
            "SELECT COUNT(*) FROM t",
            "SELECT MAX(id) FROM t GROUP BY category",
            "SELECT * FROM t ORDER BY id DESC",
            "SELECT DISTINCT name FROM users",
            "SELECT * FROM t1 JOIN t2 ON t1.id = t2.id",
            "SELECT * FROM t WHERE x IN (1, 2, 3)",
            "SELECT * FROM t WHERE x BETWEEN 1 AND 10",
            "SELECT * FROM t WHERE name LIKE '%test%'",
            "SELECT CASE WHEN x = 1 THEN 'one' ELSE 'other' END FROM t",
            "SELECT * FROM t LIMIT 10 OFFSET 5",
            "SELECT * FROM (SELECT * FROM t) AS sub",
            "WITH cte AS (SELECT * FROM t) SELECT * FROM cte",
        ]
        
        # Initialize corpus with seeds
        for stmt in self.seed_statements:
            if len(stmt) <= self.max_statement_length:
                self.corpus.append(stmt)
                self.unique_statements.add(stmt)
        
        # Initialize structure patterns
        self.init_structure_patterns()
    
    def load_grammar(self, grammar_path: str) -> Dict[str, List[str]]:
        """Load and parse the SQL grammar file."""
        grammar = defaultdict(list)
        try:
            with open(grammar_path, 'r') as f:
                lines = f.readlines()
            
            current_rule = None
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Check for rule definition
                if '::=' in line:
                    parts = line.split('::=')
                    if len(parts) == 2:
                        current_rule = parts[0].strip()
                        production = parts[1].strip()
                        if current_rule and production:
                            grammar[current_rule].append(production)
                elif current_rule and line:
                    # Continuation line
                    grammar[current_rule].append(line.strip())
        
        except Exception:
            # Fallback to basic grammar if file parsing fails
            grammar = self.create_basic_grammar()
        
        return grammar
    
    def create_basic_grammar(self) -> Dict[str, List[str]]:
        """Create a basic SQL grammar as fallback."""
        return {
            '<select>': ['SELECT <select-list> FROM <table> <where> <group> <order> <limit>'],
            '<select-list>': ['*', '<column>', '<column>, <select-list>'],
            '<table>': ['t', 't1', 't2', 'users', 'orders', 'products'],
            '<column>': ['id', 'name', 'value', 'price', 'category', 'COUNT(*)'],
            '<where>': ['', 'WHERE <condition>'],
            '<condition>': ['<column> = <value>', '<column> > <value>', '<column> IN (<values>)'],
            '<value>': ['1', "'test'", 'NULL'],
            '<values>': ['<value>', '<value>, <values>'],
            '<group>': ['', 'GROUP BY <column>'],
            '<order>': ['', 'ORDER BY <column> <direction>'],
            '<direction>': ['ASC', 'DESC'],
            '<limit>': ['', 'LIMIT <number>', 'LIMIT <number> OFFSET <number>'],
            '<number>': ['1', '10', '100'],
            '<insert>': ['INSERT INTO <table> VALUES (<values>)'],
            '<update>': ['UPDATE <table> SET <column> = <value> <where>'],
            '<delete>': ['DELETE FROM <table> <where>'],
            '<create>': ['CREATE TABLE <table> (<columns>)'],
            '<columns>': ['<column> <type>', '<column> <type>, <columns>'],
            '<type>': ['INT', 'TEXT', 'REAL'],
        }
    
    def init_structure_patterns(self):
        """Initialize SQL statement structure patterns."""
        self.structure_patterns = [
            # Basic patterns
            "SELECT {col} FROM {table}",
            "SELECT {col} FROM {table} WHERE {cond}",
            "SELECT {col} FROM {table} WHERE {cond} ORDER BY {col}",
            "SELECT {col} FROM {table} GROUP BY {col} HAVING {cond}",
            "SELECT {col} FROM {table} JOIN {table2} ON {cond}",
            
            # DML patterns
            "INSERT INTO {table} VALUES ({vals})",
            "INSERT INTO {table} ({cols}) VALUES ({vals})",
            "UPDATE {table} SET {col} = {val} WHERE {cond}",
            "DELETE FROM {table} WHERE {cond}",
            
            # DDL patterns
            "CREATE TABLE {table} ({col_defs})",
            "DROP TABLE {table}",
            "ALTER TABLE {table} ADD COLUMN {col_def}",
            
            # Complex patterns
            "WITH {cte} AS (SELECT {col} FROM {table}) SELECT {col} FROM {cte}",
            "SELECT {col} FROM (SELECT {col} FROM {table}) AS {alias}",
            "SELECT {col} FROM {table} WHERE {col} IN (SELECT {col} FROM {table2})",
            "SELECT {func}({col}) FROM {table}",
            "SELECT CASE WHEN {cond} THEN {val} ELSE {val2} END FROM {table}",
        ]
    
    def time_remaining(self) -> float:
        """Calculate remaining time budget."""
        elapsed = time.time() - self.start_time
        return max(0.0, self.time_budget - elapsed)
    
    def should_continue(self) -> bool:
        """Check if we should continue fuzzing."""
        return self.time_remaining() > 0.5  # Leave 0.5s margin
    
    def generate_from_grammar(self, rule: str = '<select>', depth: int = 0) -> str:
        """Generate a SQL statement from grammar."""
        if depth > self.max_depth or rule not in self.grammar:
            return self.get_terminal(rule)
        
        productions = self.grammar.get(rule, [])
        if not productions:
            return self.get_terminal(rule)
        
        # Choose production based on weights or random
        production = random.choice(productions)
        
        # Expand non-terminals
        result_parts = []
        tokens = production.split()
        
        for token in tokens:
            if token.startswith('<') and token.endswith('>'):
                result_parts.append(self.generate_from_grammar(token, depth + 1))
            else:
                result_parts.append(token)
        
        return ' '.join(result_parts)
    
    def get_terminal(self, rule: str) -> str:
        """Get a terminal value for a rule."""
        terminals = {
            '<table>': ['t', 't1', 't2', 'users', 'orders', 'products', 'customers', 'items'],
            '<column>': ['id', 'name', 'value', 'price', 'amount', 'quantity', 'category', 'status'],
            '<value>': ['1', '0', 'NULL', "'test'", "'hello'", "'world'", '123.45', 'true', 'false'],
            '<number>': ['1', '10', '100', '1000', '0', '-1', '999'],
            '<func>': ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'UPPER', 'LOWER'],
            '<op>': ['=', '!=', '<', '>', '<=', '>=', 'LIKE', 'IN', 'BETWEEN'],
        }
        
        if rule in terminals:
            return random.choice(terminals[rule])
        elif rule.startswith('<') and rule.endswith('>'):
            # Try to extract meaningful part
            inner = rule[1:-1]
            if inner.endswith('s'):
                return inner[:-1] + '1'
            return inner + '1'
        return rule
    
    def mutate_statement(self, stmt: str) -> str:
        """Apply various mutations to a SQL statement."""
        if not stmt or len(stmt) > self.max_statement_length * 2:
            return stmt
        
        mutations = [
            self.mutate_replace_token,
            self.mutate_delete_part,
            self.mutate_insert_token,
            self.mutate_swap_parts,
            self.mutate_change_case,
            self.mutate_add_parentheses,
            self.mutate_remove_spaces,
            self.mutate_add_junk,
        ]
        
        # Choose 1-3 mutations to apply
        num_mutations = random.randint(1, 3)
        result = stmt
        
        for _ in range(num_mutations):
            if random.random() < 0.7:  # 70% chance to apply each mutation
                mutator = random.choice(mutations)
                try:
                    result = mutator(result)
                except Exception:
                    pass
        
        # Ensure result isn't too long
        if len(result) > self.max_statement_length:
            result = result[:self.max_statement_length]
        
        return result
    
    def mutate_replace_token(self, stmt: str) -> str:
        """Replace a token in the statement."""
        tokens = re.split(r'(\\s+|[,()=])', stmt)
        if len(tokens) < 2:
            return stmt
        
        idx = random.randint(0, len(tokens) - 1)
        replacements = ['NULL', '1', '0', "'x'", '*', '=', '!=', '<>', 'LIKE']
        tokens[idx] = random.choice(replacements)
        return ''.join(tokens)
    
    def mutate_delete_part(self, stmt: str) -> str:
        """Delete a part of the statement."""
        if len(stmt) < 10:
            return stmt
        
        start = random.randint(0, len(stmt) // 2)
        end = random.randint(start + 1, min(len(stmt), start + len(stmt) // 3))
        return stmt[:start] + stmt[end:]
    
    def mutate_insert_token(self, stmt: str) -> str:
        """Insert a random token."""
        pos = random.randint(0, len(stmt))
        tokens = ['NULL', ',', '(', ')', '=', 'AND', 'OR', '1', "'x'"]
        return stmt[:pos] + random.choice(tokens) + stmt[pos:]
    
    def mutate_swap_parts(self, stmt: str) -> str:
        """Swap two parts of the statement."""
        parts = re.split(r'(\\s+AND\\s+|\\s+OR\\s+|\\s+WHERE\\s+|\\s+FROM\\s+)', stmt, maxsplit=2)
        if len(parts) >= 5:
            idx1, idx2 = 0, 2
            parts[idx1], parts[idx2] = parts[idx2], parts[idx1]
            return ''.join(parts)
        return stmt
    
    def mutate_change_case(self, stmt: str) -> str:
        """Randomly change case of parts."""
        if random.random() < 0.5:
            return stmt.upper()
        else:
            return stmt.lower()
    
    def mutate_add_parentheses(self, stmt: str) -> str:
        """Add unnecessary parentheses."""
        if '(' in stmt and ')' in stmt:
            return stmt
        return '(' + stmt + ')'
    
    def mutate_remove_spaces(self, stmt: str) -> str:
        """Remove some spaces."""
        return stmt.replace(' ', '', random.randint(1, 3))
    
    def mutate_add_junk(self, stmt: str) -> str:
        """Add random junk characters."""
        junk = ['/*comment*/', '-- comment', ';', '`', '@', '#']
        pos = random.randint(0, len(stmt))
        return stmt[:pos] + random.choice(junk) + stmt[pos:]
    
    def generate_from_pattern(self) -> str:
        """Generate statement from pattern."""
        pattern = random.choice(self.structure_patterns)
        
        replacements = {
            '{table}': random.choice(['t', 'users', 'orders', 'products']),
            '{table2}': random.choice(['t2', 'customers', 'items']),
            '{col}': random.choice(['id', 'name', 'value', 'price']),
            '{cols}': 'id, name, value',
            '{col_defs}': 'id INT, name TEXT',
            '{col_def}': 'new_col TEXT',
            '{cond}': random.choice(['id = 1', 'name = "test"', 'value > 0']),
            '{val}': random.choice(['1', "'x'", 'NULL']),
            '{vals}': '1, 2, 3',
            '{alias}': 'sub',
            '{cte}': 'cte',
            '{func}': random.choice(['COUNT', 'SUM', 'AVG']),
            '{val2}': random.choice(['0', "'y'", 'NULL']),
        }
        
        result = pattern
        for key, value in replacements.items():
            result = result.replace(key, value)
        
        return result
    
    def generate_edge_cases(self) -> List[str]:
        """Generate edge case statements."""
        edge_cases = [
            # Empty/whitespace
            "",
            "   ",
            ";",
            
            # Very long identifiers
            "SELECT " + "x" * 100 + " FROM t",
            "SELECT * FROM " + "t" * 50,
            
            # Unicode
            "SELECT '🚀' FROM t",
            "SELECT * FROM müßig",
            
            # Nested parentheses
            "SELECT (((1)))",
            "SELECT * FROM (((t)))",
            
            # Multiple statements
            "SELECT 1; SELECT 2; SELECT 3",
            "INSERT INTO t VALUES (1); DELETE FROM t",
            
            # Invalid but interesting
            "SELECT SELECT SELECT",
            "FROM SELECT WHERE",
            "1 2 3 4 5",
            
            # Edge values
            "SELECT 999999999999999999999999",
            "SELECT 0.000000000000000000001",
            "SELECT -1",
            "SELECT 1/0",
            
            # Special keywords
            "SELECT NULL FROM NULL WHERE NULL = NULL",
            "SELECT TRUE, FALSE",
            
            # Mixed case
            "SeLeCt * FrOm TaBlE",
            "select * from TABLE",
            
            # Missing parts
            "SELECT FROM",
            "INSERT VALUES",
            "UPDATE SET",
        ]
        
        return [case for case in edge_cases if len(case) <= self.max_statement_length]
    
    def generate_batch(self) -> List[str]:
        """Generate a batch of statements using multiple strategies."""
        batch = []
        remaining_time = self.time_remaining()
        
        # Adjust strategy based on remaining time
        if remaining_time < 10:
            # Final phase: focus on mutations and edge cases
            strategy_weights = [0.1, 0.5, 0.3, 0.1]
        elif remaining_time < 30:
            # Middle phase: balanced approach
            strategy_weights = [0.3, 0.4, 0.2, 0.1]
        else:
            # Initial phase: more grammar-based generation
            strategy_weights = [0.5, 0.3, 0.1, 0.1]
        
        strategies = [
            self.generate_grammar_based,
            self.generate_mutation_based,
            self.generate_pattern_based,
            self.generate_edge_case_based,
        ]
        
        target_size = min(self.batch_size, int(remaining_time * 10))
        
        while len(batch) < target_size:
            strategy_idx = random.choices(range(len(strategies)), weights=strategy_weights)[0]
            strategy = strategies[strategy_idx]
            
            try:
                stmts = strategy()
                for stmt in stmts:
                    if stmt and stmt not in self.unique_statements and len(stmt) <= self.max_statement_length:
                        batch.append(stmt)
                        self.unique_statements.add(stmt)
                        if len(batch) >= target_size:
                            break
            except Exception:
                continue
        
        return batch[:target_size]
    
    def generate_grammar_based(self) -> List[str]:
        """Generate statements using grammar."""
        statements = []
        for _ in range(random.randint(5, 15)):
            try:
                # Choose starting rule based on coverage
                rules = ['<select>', '<insert>', '<update>', '<delete>', '<create>']
                rule = random.choice(rules)
                stmt = self.generate_from_grammar(rule)
                if stmt:
                    statements.append(stmt)
            except Exception:
                pass
        return statements
    
    def generate_mutation_based(self) -> List[str]:
        """Generate statements by mutating existing ones."""
        statements = []
        if not self.corpus:
            return statements
        
        for _ in range(random.randint(10, 20)):
            try:
                parent = random.choice(self.corpus)
                mutated = self.mutate_statement(parent)
                if mutated and mutated != parent:
                    statements.append(mutated)
            except Exception:
                pass
        
        return statements
    
    def generate_pattern_based(self) -> List[str]:
        """Generate statements using patterns."""
        statements = []
        for _ in range(random.randint(5, 10)):
            try:
                stmt = self.generate_from_pattern()
                if stmt:
                    statements.append(stmt)
            except Exception:
                pass
        return statements
    
    def generate_edge_case_based(self) -> List[str]:
        """Generate edge case statements."""
        return self.generate_edge_cases()
    
    def update_corpus(self, new_statements: List[str]):
        """Update corpus with new statements."""
        for stmt in new_statements:
            if stmt and stmt not in self.unique_statements:
                self.corpus.append(stmt)
                self.unique_statements.add(stmt)
                
                # Keep corpus size manageable
                if len(self.corpus) > 1000:
                    # Remove older entries
                    self.corpus = self.corpus[-800:]
    
    def adjust_strategy(self):
        """Adjust generation strategy based on progress."""
        self.iteration += 1
        
        # Every 10 iterations, increase complexity
        if self.iteration % 10 == 0 and self.max_depth < 8:
            self.max_depth += 1
        
        # Adjust batch size based on remaining time
        remaining = self.time_remaining()
        if remaining < 5:
            self.batch_size = 100  # Large batches at the end
        elif remaining < 20:
            self.batch_size = 75
        else:
            self.batch_size = 50

# Global state instance
state = None

def fuzz(parse_sql):
    """
    Main fuzzing function called by evaluator.
    """
    global state
    
    # Initialize on first call
    if state is None:
        grammar_path = os.path.join(os.path.dirname(__file__), 'resources', 'sql_grammar.txt')
        if not os.path.exists(grammar_path):
            # Try alternative path
            grammar_path = os.path.join('resources', 'sql_grammar.txt')
        
        state = FuzzerState(grammar_path)
    
    # Check if time is up
    if not state.should_continue():
        return False
    
    # Generate batch of statements
    batch = state.generate_batch()
    
    if batch:
        # Parse the batch
        parse_sql(batch)
        
        # Update corpus
        state.update_corpus(batch)
        
        # Adjust strategy for next iteration
        state.adjust_strategy()
    
    return state.should_continue()
'''

# Output the code
if __name__ == "__main__":
    print(Solution().generate_fuzzer_code("resources"))