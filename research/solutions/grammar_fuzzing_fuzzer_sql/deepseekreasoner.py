import sys
import os
import random
import time
import math
import re
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional
import hashlib
import itertools

class Solution:
    def solve(self, resources_path: str) -> dict:
        fuzzer_code = f'''import sys
import os
import random
import time
import math
import re
import json
from collections import defaultdict, deque, Counter
from typing import List, Dict, Set, Tuple, Optional
import hashlib
import itertools
import string

# Add resources path to access grammar
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "{resources_path}"))

try:
    from sql_engine import parser, tokenizer
    HAS_PARSER = True
except ImportError:
    HAS_PARSER = False

class GrammarFuzzer:
    def __init__(self, grammar_path):
        self.grammar = self._load_grammar(grammar_path)
        self.start_symbol = '<sql_stmt>'
        self.cache = {{}}
        self.non_terminals = {{k for k in self.grammar.keys()}}
        self.max_depth = 15
        self.max_expansion = 50
        
    def _load_grammar(self, path):
        grammar = {{}}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            current_nt = None
            productions = []
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '::=' in line:
                    if current_nt and productions:
                        grammar[current_nt] = productions
                    parts = line.split('::=', 1)
                    current_nt = parts[0].strip()
                    productions = [parts[1].strip()] if parts[1].strip() else []
                elif '|' in line:
                    if current_nt:
                        prods = [p.strip() for p in line.split('|') if p.strip()]
                        productions.extend(prods)
                else:
                    if current_nt and line:
                        productions.append(line)
            
            if current_nt and productions:
                grammar[current_nt] = productions
                
        except Exception:
            # Fallback grammar if file parsing fails
            grammar = self._get_fallback_grammar()
        
        return grammar
    
    def _get_fallback_grammar(self):
        return {{
            '<sql_stmt>': [
                '<select_stmt>', '<insert_stmt>', '<update_stmt>',
                '<delete_stmt>', '<create_stmt>', '<drop_stmt>',
                '<alter_stmt>', '<compound_stmt>'
            ],
            '<select_stmt>': [
                'SELECT <select_list> FROM <table_ref> <where_clause> <group_clause> <order_clause> <limit_clause>',
                'SELECT DISTINCT <select_list> FROM <table_ref> <where_clause>',
                'SELECT <select_list> FROM <table_ref> JOIN <table_ref> ON <condition>'
            ],
            '<insert_stmt>': [
                'INSERT INTO <table_name> (<column_list>) VALUES (<value_list>)',
                'INSERT INTO <table_name> VALUES (<value_list>)'
            ],
            '<update_stmt>': [
                'UPDATE <table_name> SET <set_clause> <where_clause>'
            ],
            '<delete_stmt>': [
                'DELETE FROM <table_name> <where_clause>'
            ],
            '<create_stmt>': [
                'CREATE TABLE <table_name> (<column_defs>)',
                'CREATE INDEX <index_name> ON <table_name> (<column_list>)'
            ],
            '<drop_stmt>': [
                'DROP TABLE <table_name>',
                'DROP INDEX <index_name>'
            ],
            '<alter_stmt>': [
                'ALTER TABLE <table_name> ADD COLUMN <column_def>',
                'ALTER TABLE <table_name> DROP COLUMN <column_name>'
            ],
            '<compound_stmt>': [
                'BEGIN; <sql_stmt>; <sql_stmt>; END;',
                'WITH <cte> SELECT <select_list> FROM <table_ref>'
            ],
            '<select_list>': ['*', '<column_name>', '<column_list>'],
            '<table_ref>': ['<table_name>', '<table_name> AS <alias>'],
            '<where_clause>': ['', 'WHERE <condition>'],
            '<group_clause>': ['', 'GROUP BY <column_list>'],
            '<order_clause>': ['', 'ORDER BY <column_list>'],
            '<limit_clause>': ['', 'LIMIT <number>'],
            '<column_list>': ['<column_name>', '<column_name>, <column_list>'],
            '<value_list>': ['<value>', '<value>, <value_list>'],
            '<set_clause>': ['<column_name> = <value>', '<column_name> = <value>, <set_clause>'],
            '<condition>': [
                '<column_name> = <value>',
                '<column_name> != <value>',
                '<column_name> > <value>',
                '<column_name> < <value>',
                '<column_name> BETWEEN <value> AND <value>',
                '<column_name> IN (<value_list>)',
                '<condition> AND <condition>',
                '<condition> OR <condition>',
                'NOT <condition>',
                'EXISTS (<select_stmt>)'
            ],
            '<column_defs>': ['<column_def>', '<column_def>, <column_defs>'],
            '<column_def>': ['<column_name> <data_type>', '<column_name> <data_type> PRIMARY KEY'],
            '<data_type>': ['INT', 'VARCHAR(255)', 'TEXT', 'BOOLEAN', 'DATE', 'TIMESTAMP'],
            '<cte>': ['<alias> AS (<select_stmt>)'],
            '<table_name>': ['users', 'orders', 'products', 'customers', 'employees'],
            '<column_name>': ['id', 'name', 'age', 'price', 'quantity', 'created_at'],
            '<index_name>': ['idx_id', 'idx_name'],
            '<alias>': ['t1', 't2', 'a', 'b'],
            '<value>': ['<number>', "'<string>'", 'NULL', 'TRUE', 'FALSE'],
            '<number>': ['1', '0', '100', '-5', '3.14'],
            '<string>': ['test', 'hello', 'world', 'foo', 'bar']
        }}
    
    def expand_symbol(self, symbol, depth=0):
        if depth > self.max_depth:
            return ''
        
        if symbol not in self.grammar:
            return symbol
        
        if symbol in self.cache and depth > 5:
            return random.choice(self.cache[symbol]) if self.cache[symbol] else ''
        
        productions = self.grammar[symbol]
        weights = [1.0] * len(productions)
        
        # Bias towards shorter productions for deeper recursion
        if depth > 8:
            for i, prod in enumerate(productions):
                if len(prod.split()) < 5:
                    weights[i] *= 2.0
        
        chosen = random.choices(productions, weights=weights, k=1)[0]
        
        # Expand each part
        parts = re.split(r'(<[^>]+>|[^<]+)', chosen)
        result_parts = []
        
        for part in parts:
            if not part:
                continue
            if part.startswith('<') and part.endswith('>'):
                expanded = self.expand_symbol(part, depth + 1)
                result_parts.append(expanded)
            else:
                result_parts.append(part)
        
        result = ''.join(result_parts).strip()
        
        # Cache results for non-terminals
        if symbol not in self.cache:
            self.cache[symbol] = []
        if result and result not in self.cache[symbol]:
            self.cache[symbol].append(result)
        
        return result
    
    def generate_statement(self):
        return self.expand_symbol(self.start_symbol)

class SQLFuzzer:
    def __init__(self, grammar_path):
        self.grammar_fuzzer = GrammarFuzzer(grammar_path)
        self.corpus = []
        self.unique_hashes = set()
        self.stats = {{'generated': 0, 'unique': 0, 'mutated': 0}}
        self.mutation_weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Weighted mutation strategies
        self.templates = self._get_templates()
        self.start_time = None
        self.time_budget = 60.0
        
    def _get_templates(self):
        return [
            "SELECT * FROM {table}",
            "SELECT {cols} FROM {table} WHERE {cond}",
            "INSERT INTO {table} VALUES ({vals})",
            "UPDATE {table} SET {set} WHERE {cond}",
            "DELETE FROM {table} WHERE {cond}",
            "CREATE TABLE {table} ({cols} {type})",
            "DROP TABLE {table}",
            "BEGIN; {stmt1}; {stmt2}; END;",
            "WITH {cte} AS (SELECT * FROM {table}) SELECT * FROM {cte}",
            "SELECT {cols} FROM {table} JOIN {table2} ON {cond}",
        ]
    
    def add_to_corpus(self, stmt):
        stmt_hash = hashlib.md5(stmt.encode()).hexdigest()
        if stmt_hash not in self.unique_hashes and len(stmt) < 1000:
            self.corpus.append(stmt)
            self.unique_hashes.add(stmt_hash)
            return True
        return False
    
    def generate_grammar_based(self, count):
        stmts = []
        for _ in range(count):
            stmt = self.grammar_fuzzer.generate_statement()
            if stmt and self.add_to_corpus(stmt):
                stmts.append(stmt)
        return stmts
    
    def generate_template_based(self, count):
        stmts = []
        templates = [
            "SELECT * FROM t{id}",
            "SELECT col{id} FROM t{id} WHERE col{id} > {val}",
            "INSERT INTO t{id} VALUES ({val1}, '{val2}')",
            "UPDATE t{id} SET col{id} = {val}",
            "DELETE FROM t{id} WHERE id = {val}",
            "CREATE TABLE t{id} (id INT, name TEXT)",
            "DROP TABLE IF EXISTS t{id}",
            "SELECT * FROM (SELECT * FROM t{id})",
            "SELECT * FROM t{id} UNION SELECT * FROM t{id2}",
            "SELECT * FROM t{id} ORDER BY col{id} DESC LIMIT {val}",
        ]
        
        for _ in range(count):
            template = random.choice(templates)
            stmt = template.format(
                id=random.randint(1, 100),
                id2=random.randint(1, 100),
                val=random.randint(0, 1000),
                val1=random.randint(0, 100),
                val2=''.join(random.choices(string.ascii_letters, k=5))
            )
            if self.add_to_corpus(stmt):
                stmts.append(stmt)
        return stmts
    
    def mutate_statement(self, stmt):
        if not stmt or len(stmt) < 3:
            return self.grammar_fuzzer.generate_statement()
        
        tokens = re.split(r'(\\s+|[,;()])', stmt)
        if len(tokens) < 2:
            return stmt + " WHERE 1=1"
        
        mutation_type = random.choices(
            ['insert', 'delete', 'replace', 'swap', 'duplicate'],
            weights=self.mutation_weights,
            k=1
        )[0]
        
        try:
            if mutation_type == 'insert' and len(tokens) < 50:
                pos = random.randint(0, len(tokens) - 1)
                insert_tokens = [' WHERE 1=1', ' AND 2=2', ' OR 3=3', ' LIMIT 1', ' OFFSET 0']
                tokens.insert(pos, random.choice(insert_tokens))
                
            elif mutation_type == 'delete' and len(tokens) > 3:
                pos = random.randint(0, len(tokens) - 1)
                if tokens[pos].strip() and len(tokens[pos].strip()) > 1:
                    tokens.pop(pos)
                    
            elif mutation_type == 'replace':
                pos = random.randint(0, len(tokens) - 1)
                replacements = ['NULL', '1', '0', "'X'", '*']
                tokens[pos] = random.choice(replacements)
                
            elif mutation_type == 'swap' and len(tokens) > 4:
                i, j = random.sample(range(len(tokens)), 2)
                tokens[i], tokens[j] = tokens[j], tokens[i]
                
            elif mutation_type == 'duplicate' and len(tokens) < 40:
                pos = random.randint(0, len(tokens) - 1)
                tokens.insert(pos, tokens[pos])
        
        except Exception:
            pass
        
        result = ''.join(tokens)
        return result if result.strip() else stmt
    
    def generate_mutated(self, count):
        stmts = []
        if not self.corpus:
            return self.generate_grammar_based(count)
        
        for _ in range(count * 2):  # Generate extra to account for duplicates
            parent = random.choice(self.corpus[-100:] if len(self.corpus) > 100 else self.corpus)
            mutated = self.mutate_statement(parent)
            if mutated and self.add_to_corpus(mutated):
                stmts.append(mutated)
                if len(stmts) >= count:
                    break
        return stmts
    
    def generate_edge_cases(self):
        edge_cases = [
            # Empty/whitespace
            "",
            "   ",
            ";",
            
            # Minimal valid
            "SELECT 1",
            "SELECT",
            "FROM",
            
            # Keywords as identifiers
            "SELECT SELECT FROM SELECT",
            "SELECT * FROM WHERE",
            
            # Special values
            "SELECT NULL",
            "SELECT TRUE, FALSE",
            
            # Nested
            "SELECT (SELECT 1)",
            "SELECT * FROM (SELECT 1)",
            
            # Complex expressions
            "SELECT 1 + 2 * 3",
            "SELECT 'a' || 'b'",
            
            # Odd punctuation
            "SELECT;;;;",
            "SELECT * FROM t,,,,,",
            
            # Very long identifier
            "SELECT " + "x" * 100,
            
            # Unicode
            "SELECT '🎉'",
            "SELECT * FROM müßtär",
            
            # Mixed case
            "SeLeCt * FrOm TaBlE",
            
            # Missing parts
            "SELECT FROM",
            "INSERT INTO",
            "UPDATE SET",
            
            # Trailing semicolons
            "SELECT 1;",
            "SELECT 1;;;",
            
            # Multiple statements
            "SELECT 1; SELECT 2",
            "CREATE TABLE t (a INT); DROP TABLE t",
            
            # Comments
            "SELECT 1 -- comment",
            "/* comment */ SELECT 1",
            
            # Wildcards
            "SELECT %",
            "SELECT _",
        ]
        
        new_stmts = []
        for stmt in edge_cases:
            if self.add_to_corpus(stmt):
                new_stmts.append(stmt)
        return new_stmts
    
    def generate_batch(self, phase, batch_size):
        if phase == 0:  # Initial: grammar + templates
            stmts = []
            stmts.extend(self.generate_grammar_based(batch_size // 2))
            stmts.extend(self.generate_template_based(batch_size // 2))
            return stmts
            
        elif phase == 1:  # Exploration: edge cases + mutations
            stmts = []
            if random.random() < 0.3:
                stmts.extend(self.generate_edge_cases())
            stmts.extend(self.generate_mutated(max(1, batch_size - len(stmts))))
            return stmts
            
        else:  # Exploitation: mostly mutations with some grammar
            stmts = []
            if random.random() < 0.2:
                stmts.extend(self.generate_grammar_based(max(1, batch_size // 4)))
            stmts.extend(self.generate_mutated(max(1, batch_size - len(stmts))))
            return stmts
    
    def run_fuzzing(self, parse_sql, time_budget=60.0):
        self.start_time = time.time()
        self.time_budget = time_budget
        
        # Initial corpus
        initial_stmts = []
        initial_stmts.extend(self.generate_grammar_based(20))
        initial_stmts.extend(self.generate_template_based(20))
        initial_stmts.extend(self.generate_edge_cases())
        
        for stmt in initial_stmts:
            self.add_to_corpus(stmt)
        
        phase = 0
        last_coverage_time = self.start_time
        
        while True:
            elapsed = time.time() - self.start_time
            if elapsed >= self.time_budget * 0.95:
                return False
            
            # Dynamic batch sizing based on remaining time
            remaining = self.time_budget - elapsed
            if remaining < 5.0:
                batch_size = 5
            elif remaining < 15.0:
                batch_size = 10
            elif remaining < 30.0:
                batch_size = 15
            else:
                batch_size = 20
            
            # Phase transition
            if elapsed > 45.0:
                phase = 2  # Final exploitation
            elif elapsed > 20.0:
                phase = 1  # Exploration
            
            # Generate batch
            stmts = self.generate_batch(phase, batch_size)
            if not stmts:
                stmts = [self.grammar_fuzzer.generate_statement()]
            
            # Execute through parser
            try:
                parse_sql(stmts)
            except Exception:
                pass
            
            # Check if we should continue
            if elapsed >= self.time_budget * 0.98:
                return False
            
            # Yield to evaluator
            return True

# Global fuzzer instance
_fuzzer = None

def fuzz(parse_sql):
    """
    Main fuzzing function called by evaluator.
    """
    global _fuzzer
    
    # Initialize on first call
    if _fuzzer is None:
        try:
            # Try to find grammar file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            grammar_path = os.path.join(base_dir, "sql_grammar.txt")
            
            if not os.path.exists(grammar_path):
                # Try resources path
                resources_dir = os.path.join(base_dir, "resources")
                if os.path.exists(resources_dir):
                    grammar_path = os.path.join(resources_dir, "sql_grammar.txt")
            
            _fuzzer = SQLFuzzer(grammar_path)
        except Exception:
            # Fallback
            _fuzzer = SQLFuzzer("")
    
    # Run one iteration
    return _fuzzer.run_fuzzing(parse_sql, time_budget=60.0)
'''
        return {"code": fuzzer_code}