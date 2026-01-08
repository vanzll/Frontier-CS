import random
import string
import math
import time
from typing import List, Tuple, Dict, Set, Optional, Any
import json
import os

class Solution:
    def solve(self, resources_path: str) -> dict:
        grammar_path = os.path.join(resources_path, "sql_grammar.txt")
        
        with open(grammar_path, 'r') as f:
            grammar_content = f.read()
        
        fuzzer_code = f'''
import random
import string
import math
import time
from typing import List, Tuple, Dict, Set, Optional
import re

# Grammar parser and generator
class SQLGrammar:
    def __init__(self, grammar_text: str):
        self.rules = {{}}
        self._parse_grammar(grammar_text)
        self._build_first_sets()
        
    def _parse_grammar(self, grammar_text: str):
        lines = grammar_text.strip().split('\\n')
        current_nt = None
        current_productions = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if line.startswith('<') and '::=' in line:
                if current_nt:
                    self.rules[current_nt] = current_productions
                parts = line.split('::=', 1)
                current_nt = parts[0].strip()
                current_productions = [parts[1].strip()]
            elif line.startswith('|'):
                if current_nt:
                    current_productions.append(line[1:].strip())
            else:
                if current_nt:
                    current_productions[-1] += ' ' + line
                    
        if current_nt:
            self.rules[current_nt] = current_productions
            
        # Clean up productions
        for nt in self.rules:
            cleaned = []
            for prod in self.rules[nt]:
                prod = prod.strip()
                if prod:
                    # Split into tokens
                    tokens = []
                    i = 0
                    while i < len(prod):
                        if prod[i] == '<':
                            end = prod.find('>', i)
                            if end != -1:
                                tokens.append(prod[i:end+1])
                                i = end + 1
                            else:
                                tokens.append(prod[i])
                                i += 1
                        elif prod[i] == "'":
                            end = prod.find("'", i + 1)
                            if end != -1:
                                tokens.append(prod[i:end+1])
                                i = end + 1
                            else:
                                tokens.append(prod[i])
                                i += 1
                        elif prod[i] == '"':
                            end = prod.find('"', i + 1)
                            if end != -1:
                                tokens.append(prod[i:end+1])
                                i = end + 1
                            else:
                                tokens.append(prod[i])
                                i += 1
                        elif prod[i].isspace():
                            i += 1
                        else:
                            # Find end of token
                            start = i
                            while i < len(prod) and not prod[i].isspace() and prod[i] not in '<>\'"':
                                i += 1
                            token = prod[start:i]
                            if token:
                                tokens.append(token)
                    cleaned.append(tokens)
            self.rules[nt] = cleaned
            
    def _build_first_sets(self):
        self.first_sets = {{nt: set() for nt in self.rules}}
        changed = True
        while changed:
            changed = False
            for nt in self.rules:
                for production in self.rules[nt]:
                    if production:
                        first_token = production[0]
                        if first_token.startswith('<'):
                            old_len = len(self.first_sets[nt])
                            self.first_sets[nt].update(self.first_sets.get(first_token, set()))
                            if len(self.first_sets[nt]) > old_len:
                                changed = True
                        else:
                            old_len = len(self.first_sets[nt])
                            self.first_sets[nt].add(first_token.strip("'\\""))
                            if len(self.first_sets[nt]) > old_len:
                                changed = True
    
    def generate_from_nt(self, nt: str, depth: int = 0, max_depth: int = 8) -> str:
        if depth > max_depth:
            # Return shortest production
            shortest = None
            for prod in self.rules.get(nt, []):
                text = self._production_to_text(prod, depth, max_depth)
                if shortest is None or len(text) < len(shortest):
                    shortest = text
            return shortest or ''
            
        if nt not in self.rules:
            return nt.strip("'\\"")
            
        prods = self.rules[nt]
        # Weight productions based on complexity
        weights = []
        for prod in prods:
            complexity = self._estimate_complexity(prod)
            weight = 1.0 / (1.0 + complexity)
            weights.append(weight)
            
        if not weights:
            return ''
            
        prod = random.choices(prods, weights=weights, k=1)[0]
        return self._production_to_text(prod, depth + 1, max_depth)
    
    def _production_to_text(self, production: List[str], depth: int, max_depth: int) -> str:
        parts = []
        for token in production:
            if token.startswith('<'):
                parts.append(self.generate_from_nt(token, depth, max_depth))
            else:
                parts.append(token.strip("'\\""))
        return ' '.join(parts)
    
    def _estimate_complexity(self, production: List[str]) -> float:
        complexity = 0
        for token in production:
            if token.startswith('<'):
                complexity += 1
            elif token in ['SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP', 'ORDER', 'HAVING']:
                complexity += 0.5
        return complexity

# Mutation strategies
class SQLMutator:
    def __init__(self, grammar: SQLGrammar):
        self.grammar = grammar
        self.keywords = [
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
            'ON', 'AND', 'OR', 'NOT', 'IN', 'EXISTS', 'BETWEEN', 'LIKE', 'IS', 'NULL',
            'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT', 'OFFSET', 'DISTINCT',
            'INSERT', 'INTO', 'VALUES', 'UPDATE', 'SET', 'DELETE', 'CREATE', 'TABLE',
            'ALTER', 'DROP', 'TRUNCATE', 'INDEX', 'VIEW', 'UNION', 'INTERSECT', 'EXCEPT',
            'CASE', 'WHEN', 'THEN', 'ELSE', 'END'
        ]
        
        self.functions = [
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'COALESCE', 'NULLIF', 'CAST',
            'CONCAT', 'SUBSTR', 'UPPER', 'LOWER', 'TRIM', 'LENGTH', 'ROUND',
            'ABS', 'CEIL', 'FLOOR', 'MOD', 'POWER', 'SQRT', 'RANDOM',
            'CURRENT_DATE', 'CURRENT_TIME', 'CURRENT_TIMESTAMP', 'EXTRACT',
            'DATE_ADD', 'DATE_SUB', 'DATE_DIFF'
        ]
        
        self.operators = ['=', '<>', '!=', '<', '>', '<=', '>=', '+', '-', '*', '/', '%', '||']
        
    def mutate(self, sql: str) -> str:
        mutations = [
            self._mutate_keyword,
            self._mutate_operator,
            self._mutate_literal,
            self._mutate_whitespace,
            self._mutate_case,
            self._mutate_add_subexpression,
            self._mutate_remove_part,
            self._mutate_swap_parts
        ]
        
        # Apply 1-3 mutations
        num_mutations = random.randint(1, 3)
        result = sql
        
        for _ in range(num_mutations):
            if not result.strip():
                result = self.grammar.generate_from_nt('<start>')
                continue
                
            mutator = random.choice(mutations)
            try:
                mutated = mutator(result)
                if mutated and mutated != result:
                    result = mutated
            except:
                pass
                
        return result
    
    def _mutate_keyword(self, sql: str) -> str:
        words = sql.upper().split()
        if not words:
            return sql
            
        # Find keyword positions
        keyword_positions = []
        for i, word in enumerate(words):
            clean_word = word.strip(string.punctuation + string.whitespace)
            if clean_word in self.keywords:
                keyword_positions.append(i)
                
        if not keyword_positions:
            return sql
            
        pos = random.choice(keyword_positions)
        current = words[pos]
        
        # Find similar keywords
        similar = [k for k in self.keywords 
                  if k.startswith(current[:2]) or current[:2] in k]
        if not similar:
            similar = self.keywords
            
        replacement = random.choice(similar)
        
        # Reconstruct SQL preserving original case pattern
        original_words = sql.split()
        if pos < len(original_words):
            original_word = original_words[pos]
            if original_word.isupper():
                new_word = replacement
            elif original_word[0].isupper():
                new_word = replacement.lower().capitalize()
            else:
                new_word = replacement.lower()
                
            original_words[pos] = new_word
            
        return ' '.join(original_words)
    
    def _mutate_operator(self, sql: str) -> str:
        operators_in_sql = []
        for op in self.operators:
            if op in sql:
                operators_in_sql.append(op)
                
        if not operators_in_sql:
            return sql
            
        old_op = random.choice(operators_in_sql)
        new_op = random.choice([o for o in self.operators if o != old_op])
        
        # Replace carefully to avoid replacing within larger strings
        parts = re.split(f'(\\\\b{re.escape(old_op)}\\\\b)', sql)
        result_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 1:  # This is the operator
                result_parts.append(new_op)
            else:
                result_parts.append(part)
                
        return ''.join(result_parts)
    
    def _mutate_literal(self, sql: str) -> str:
        # Find numeric literals
        nums = re.findall(r'\\b\\d+\\.?\\d*\\b', sql)
        if nums:
            num = random.choice(nums)
            replacements = [
                '0', '1', 'NULL', '999', str(random.randint(-100, 100)),
                str(random.uniform(-100, 100))
            ]
            new_num = random.choice(replacements)
            return sql.replace(num, new_num, 1)
            
        # Find string literals
        str_pattern = r"('[^']*')|(\"[^\"]*\")"
        strings = re.findall(str_pattern, sql)
        flat_strings = [s[0] or s[1] for s in strings if s[0] or s[1]]
        
        if flat_strings:
            old_str = random.choice(flat_strings)
            quotes = old_str[0]
            replacements = [
                quotes + quotes,  # Empty string
                quotes + 'test' + quotes,
                quotes + ''.join(random.choices(string.ascii_letters, k=5)) + quotes,
                quotes + ''.join(random.choices(string.digits, k=3)) + quotes,
                'NULL'
            ]
            new_str = random.choice(replacements)
            return sql.replace(old_str, new_str, 1)
            
        return sql
    
    def _mutate_whitespace(self, sql: str) -> str:
        # Add, remove, or change whitespace
        lines = sql.split('\\n')
        if len(lines) > 1 and random.random() < 0.3:
            # Change indentation
            indent_chars = ['    ', '  ', '\\t', '']
            new_indent = random.choice(indent_chars)
            return '\\n'.join(new_indent + line.lstrip() for line in lines)
        else:
            # Add/remove spaces
            words = sql.split()
            if len(words) > 2:
                # Remove random space
                if random.random() < 0.5:
                    pos = random.randint(0, len(words) - 2)
                    words[pos] = words[pos] + words[pos + 1]
                    del words[pos + 1]
                else:
                    # Add extra space
                    pos = random.randint(0, len(words) - 1)
                    words.insert(pos, '')
            return ' '.join(words)
    
    def _mutate_case(self, sql: str) -> str:
        if random.random() < 0.5:
            return sql.upper()
        else:
            return sql.lower()
    
    def _mutate_add_subexpression(self, sql: str) -> str:
        insert_points = [' WHERE ', ' AND ', ' OR ', ',', ' GROUP BY ', ' ORDER BY ']
        
        for point in insert_points:
            if point in sql.upper():
                parts = re.split(f'(?i){re.escape(point)}', sql, 1)
                if len(parts) > 1:
                    subexprs = [
                        '1=1',
                        'NULL IS NULL',
                        'col IS NOT NULL',
                        'x > 0',
                        'name LIKE \\'%test%\\'',
                        'id IN (1, 2, 3)',
                        'EXISTS (SELECT 1)'
                    ]
                    subexpr = random.choice(subexprs)
                    return parts[0] + point + subexpr + ' AND ' + parts[1]
                    
        return sql
    
    def _mutate_remove_part(self, sql: str) -> str:
        upper_sql = sql.upper()
        patterns = [
            (r'WHERE.*?(?=(?:GROUP BY|ORDER BY|HAVING|LIMIT|$))', ''),
            (r'GROUP BY.*?(?=(?:ORDER BY|HAVING|LIMIT|$))', ''),
            (r'ORDER BY.*?(?=(?:LIMIT|$))', ''),
            (r'LIMIT \\d+', ''),
            (r'OFFSET \\d+', ''),
            (r',\\s*[\\w\\.]+', '')
        ]
        
        for pattern, replacement in patterns:
            match = re.search(pattern, upper_sql, re.IGNORECASE | re.DOTALL)
            if match:
                start, end = match.span()
                return sql[:start] + replacement + sql[end:]
                
        return sql
    
    def _mutate_swap_parts(self, sql: str) -> str:
        words = sql.split()
        if len(words) > 3:
            i = random.randint(0, len(words) - 3)
            j = random.randint(i + 1, min(i + 3, len(words) - 1))
            words[i], words[j] = words[j], words[i]
            return ' '.join(words)
        return sql

# Coverage-guided fuzzer
class CoverageGuidedFuzzer:
    def __init__(self, grammar: SQLGrammar):
        self.grammar = grammar
        self.mutator = SQLMutator(grammar)
        self.corpus = []
        self.energy = {}
        self.mutation_count = 0
        self.generation_count = 0
        self.last_new_input_time = time.time()
        
        # Initialize with diverse seeds
        self._initialize_corpus()
        
    def _initialize_corpus(self):
        seed_templates = [
            '<start>',
            '<query_specification>',
            '<insert_statement>',
            '<update_statement>',
            '<delete_statement>',
            '<create_table_statement>',
            '<subquery>',
            '<case_expression>'
        ]
        
        for template in seed_templates:
            for _ in range(3):
                try:
                    sql = self.grammar.generate_from_nt(template, max_depth=6)
                    if sql and sql not in self.corpus:
                        self.corpus.append(sql)
                        self.energy[sql] = 10.0
                except:
                    pass
                    
        # Add some edge cases
        edge_cases = [
            "SELECT NULL",
            "SELECT 1",
            "SELECT *",
            "SELECT",
            "",
            ";",
            "SELECT * FROM (SELECT 1)",
            "SELECT * FROM t WHERE 1=1",
            "SELECT * FROM t WHERE x IS NULL",
            "SELECT * FROM t WHERE x = 'test\\' OR '1'='1'",
            "SELECT * FROM t a, t b, t c, t d, t e",
            "SELECT " + ",".join([f"col{i}" for i in range(100)]),
            "SELECT * FROM t" + " JOIN t" * 10,
            "SELECT * FROM t WHERE x = 1 AND y = 2 AND z = 3 AND a = 4 AND b = 5"
        ]
        
        for sql in edge_cases:
            if sql not in self.corpus:
                self.corpus.append(sql)
                self.energy[sql] = 15.0
                
    def generate_batch(self, size: int = 50) -> List[str]:
        batch = []
        
        # Strategy distribution
        if len(self.corpus) < 20 or random.random() < 0.3:
            # Generate new from grammar
            for _ in range(size // 3):
                try:
                    nt = random.choice(list(self.grammar.rules.keys()))
                    max_depth = random.randint(4, 10)
                    sql = self.grammar.generate_from_nt(nt, max_depth=max_depth)
                    if sql and sql not in self.corpus:
                        self.corpus.append(sql)
                        self.energy[sql] = 10.0
                        batch.append(sql)
                        self.generation_count += 1
                except:
                    pass
                    
        # Mutation-based generation
        mutation_targets = []
        if self.corpus:
            # Select based on energy (priority to less explored inputs)
            total_energy = sum(self.energy.values())
            if total_energy > 0:
                for sql in self.corpus:
                    prob = self.energy[sql] / total_energy
                    mutation_targets.extend([sql] * max(1, int(prob * 100)))
        
        if mutation_targets:
            for _ in range(size - len(batch)):
                parent = random.choice(mutation_targets)
                mutated = self.mutator.mutate(parent)
                
                if mutated and mutated != parent:
                    # Keep mutated version in corpus temporarily
                    if random.random() < 0.1:  # 10% chance to add to permanent corpus
                        if mutated not in self.corpus:
                            self.corpus.append(mutated)
                            self.energy[mutated] = 10.0
                    
                    batch.append(mutated)
                    self.mutation_count += 1
                    
                    # Reduce parent energy to explore other paths
                    self.energy[parent] = max(1.0, self.energy[parent] * 0.95)
        
        # If we still need more, generate simple queries
        while len(batch) < size:
            simple_queries = [
                f"SELECT col{random.randint(1, 10)} FROM table{random.randint(1, 5)}",
                f"INSERT INTO t VALUES ({random.randint(1, 100)})",
                f"UPDATE t SET x = {random.randint(1, 100)} WHERE id = {random.randint(1, 100)}",
                "DELETE FROM t WHERE id > 0",
                f"SELECT * FROM t WHERE x {random.choice(['>', '<', '=', '<>'])} {random.randint(1, 100)}"
            ]
            batch.append(random.choice(simple_queries))
            
        # Shuffle to mix strategies
        random.shuffle(batch)
        return batch[:size]
    
    def feedback(self, successful: bool):
        # Simple feedback mechanism - increase energy for recent mutations if successful
        if successful and time.time() - self.last_new_input_time > 5:
            # Boost energy for all corpus items
            for sql in self.corpus:
                self.energy[sql] = min(50.0, self.energy[sql] * 1.1)
            self.last_new_input_time = time.time()

# Main fuzzer state
_fuzzer_instance = None
_start_time = None
_timeout = 58  # Stop 2 seconds before timeout

def fuzz(parse_sql):
    global _fuzzer_instance, _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    if _fuzzer_instance is None:
        # Parse grammar from embedded data
        grammar_text = """{grammar_content}"""
        grammar = SQLGrammar(grammar_text)
        _fuzzer_instance = CoverageGuidedFuzzer(grammar)
    
    # Check timeout
    elapsed = time.time() - _start_time
    if elapsed >= _timeout:
        return False
    
    # Generate batch with adaptive size based on remaining time
    remaining = _timeout - elapsed
    if remaining < 5:
        batch_size = 20
    elif remaining < 15:
        batch_size = 40
    else:
        batch_size = 50
    
    # Generate and execute
    batch = _fuzzer_instance.generate_batch(batch_size)
    
    if batch:
        parse_sql(batch)
        # Assume success if we got here without exception
        _fuzzer_instance.feedback(True)
    
    return True
'''
        
        return {"code": fuzzer_code}