import os
import re
import random
import collections

class Solution:
    def solve(self, resources_path: str) -> list[str]:
        grammar_path = os.path.join(resources_path, 'sql_grammar.txt')
        
        # 1. Parse Grammar
        grammar, root_rule = self._parse_grammar(grammar_path)
        
        # Fallback if grammar is empty or broken
        if not grammar or not root_rule:
            return self._fallback_queries()
            
        # 2. Generate Statements
        generator = GrammarGenerator(grammar, root_rule)
        statements = []
        unique_stmts = set()
        
        # Target: ~40 statements to balance coverage vs efficiency
        # We assume 40 covers most complex grammars if we guide it well.
        # Efficiency bonus drops significantly after 50.
        max_stmts = 40
        attempts = 0
        
        while len(unique_stmts) < max_stmts and attempts < 200:
            attempts += 1
            stmt = generator.generate()
            if stmt and stmt not in unique_stmts:
                unique_stmts.add(stmt)
                statements.append(stmt)
                
            # If we have fully saturated the grammar (visited every branch) and have a decent sample size,
            # stop early to maximize efficiency bonus.
            if len(unique_stmts) > 25 and generator.is_saturated():
                break
                
        return statements

    def _parse_grammar(self, path):
        grammar = {}
        root = None
        if not os.path.exists(path):
            return grammar, root
            
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        # Robust BNF Parsing
        # Regex to identify rule definition start: <rule> ::= ... or rule ::= ... or rule : ...
        # Group 1: Rule Name
        # Group 2: Separator
        # Group 3: Rest of line
        start_re = re.compile(r'^\s*(?:<)?([a-zA-Z0-9_]+)(?:>)?\s*(::?=|:)\s*(.*)$')
        
        current_rule = None
        buffer = ""
        
        for line in lines:
            line = line.split('#')[0].strip() # Remove comments
            if not line: continue
            
            match = start_re.match(line)
            if match:
                # Store previous rule if exists
                if current_rule:
                    grammar[current_rule] = self._parse_rhs(buffer)
                
                current_rule = match.group(1)
                buffer = match.group(3)
                if not root: root = current_rule
            else:
                # Continuation of previous rule
                if current_rule:
                    buffer += " " + line
                    
        # Store last rule
        if current_rule:
            grammar[current_rule] = self._parse_rhs(buffer)
            
        return grammar, root

    def _parse_rhs(self, rhs):
        # Alternatives separated by |
        alts = [x.strip() for x in rhs.split('|')]
        parsed_alts = []
        
        for alt in alts:
            # Tokenize by whitespace
            raw_tokens = alt.split()
            parsed_row = []
            
            for t in raw_tokens:
                # Check for Non-Terminal: <name>
                if t.startswith('<') and t.endswith('>'):
                    parsed_row.append({'type': 'NONTERM', 'val': t.strip('<>')})
                elif t.upper() == 'EPSILON' or t == "''" or t == '""':
                    continue
                else:
                    # Terminal: remove quotes
                    parsed_row.append({'type': 'TERM', 'val': t.strip("'\"")})
            
            parsed_alts.append(parsed_row)
            
        return parsed_alts

    def _fallback_queries(self):
        return [
            "SELECT * FROM table1",
            "SELECT col1, col2 FROM table1 WHERE col1 > 10",
            "INSERT INTO table1 (col1, col2) VALUES (1, 'test')",
            "UPDATE table1 SET col1 = 5 WHERE col2 = 'test'",
            "DELETE FROM table1 WHERE col1 IS NULL",
            "CREATE TABLE table2 (id INT, name TEXT)",
            "SELECT count(*) FROM table1 GROUP BY col1 HAVING count(*) > 5",
            "SELECT * FROM t1 JOIN t2 ON t1.id = t2.id",
            "SELECT * FROM t1 LEFT JOIN t2 ON t1.id = t2.id",
            "SELECT (1 + 2) * 3 FROM t1"
        ]

class GrammarGenerator:
    def __init__(self, grammar, root):
        self.grammar = grammar
        self.root = root
        # Usage map: (rule_name, alt_index) -> count
        self.usage = collections.defaultdict(int)
        
    def generate(self):
        # Generate with depth limit
        return self._expand(self.root, 0)
        
    def _expand(self, symbol, depth):
        # Check if symbol is a defined rule
        if symbol not in self.grammar:
            # If it's a non-terminal <foo> but not defined, treat as primitive
            return self._generate_primitive(symbol)
            
        alts = self.grammar[symbol]
        
        # Select Alternative
        # 1. Filter based on depth to avoid infinite recursion
        valid_indices = list(range(len(alts)))
        
        # Heuristic to stop recursion
        if depth > 10:
            # Prefer alternatives that are purely terminals or non-recursive
            non_rec = []
            for i in valid_indices:
                row = alts[i]
                # Check if this alternative has Non-Terminals
                has_nt = any(t['type'] == 'NONTERM' for t in row)
                if not has_nt:
                    non_rec.append(i)
            
            if non_rec:
                valid_indices = non_rec
            else:
                # If all recursive, pick the shortest one (fewest symbols)
                valid_indices.sort(key=lambda i: len(alts[i]))
                valid_indices = valid_indices[:1]
        
        # 2. Coverage-guided selection (Least Used)
        # Shuffle valid indices first to ensure variety if counts are equal
        random.shuffle(valid_indices)
        best_idx = min(valid_indices, key=lambda i: self.usage[(symbol, i)])
        
        # Increment usage
        self.usage[(symbol, best_idx)] += 1
        
        # Expand chosen alternative
        parts = []
        for token in alts[best_idx]:
            if token['type'] == 'TERM':
                parts.append(token['val'])
            else:
                parts.append(self._expand(token['val'], depth + 1))
                
        return " ".join(parts)

    def _generate_primitive(self, symbol):
        # Heuristics for common SQL undefined non-terminals
        s = symbol.lower()
        if 'ident' in s or 'name' in s:
            return f"id_{random.randint(1,100)}"
        if 'table' in s:
            return f"table_{random.randint(1,10)}"
        if 'col' in s:
            return f"col_{random.randint(1,10)}"
        if 'int' in s or 'num' in s:
            return str(random.randint(0, 100))
        if 'str' in s or 'text' in s or 'literal' in s:
            return "'test_val'"
        if 'bool' in s:
            return random.choice(['TRUE', 'FALSE'])
        if 'op' in s:
            return random.choice(['=', '!=', '>', '<', '>=', '<='])
        # Catch-all
        return "x"

    def is_saturated(self):
        # Check if every alternative of every rule has been used at least once
        for rule, alts in self.grammar.items():
            for i in range(len(alts)):
                if self.usage[(rule, i)] == 0:
                    return False
        return True