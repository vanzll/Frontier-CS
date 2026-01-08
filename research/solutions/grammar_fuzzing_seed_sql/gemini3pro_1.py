import sys
import os
import re
import random

class Solution:
    def solve(self, resources_path: str) -> list[str]:
        # Ensure we can import from the resources if needed (though we mostly parse text)
        if resources_path not in sys.path:
            sys.path.append(resources_path)
            
        grammar_path = os.path.join(resources_path, 'sql_grammar.txt')
        grammar = Grammar(grammar_path)
        
        # Configure generators for leaf nodes/values to ensure valid SQL
        # We define a few variations of common names used in BNFs
        id_gen = lambda: f"x{random.randint(1, 100)}"
        grammar.set_generator('identifier', id_gen)
        grammar.set_generator('ident', id_gen)
        grammar.set_generator('id', id_gen)
        grammar.set_generator('name', id_gen)
        grammar.set_generator('table_name', id_gen)
        grammar.set_generator('column_name', id_gen)
        grammar.set_generator('alias', id_gen)
        
        int_gen = lambda: str(random.randint(0, 100))
        grammar.set_generator('integer', int_gen)
        grammar.set_generator('int', int_gen)
        grammar.set_generator('number', int_gen)
        
        float_gen = lambda: f"{random.randint(0,100)}.{random.randint(0,99)}"
        grammar.set_generator('float', float_gen)
        grammar.set_generator('decimal', float_gen)
        
        str_gen = lambda: f"'{random.choice(['foo', 'bar', 'baz'])}'"
        grammar.set_generator('string', str_gen)
        grammar.set_generator('string_literal', str_gen)
        
        cases = set()
        start = grammar.start_symbol
        
        # If grammar failed to load, return a fallback
        if not start:
            return ["SELECT * FROM table1"]
        
        # Generate test cases targeting coverage
        # We run enough iterations to likely hit all accessible grammar branches
        # Deduping ensures we don't submit redundant tests
        for _ in range(75):
            sql = grammar.generate(start, mode='coverage')
            if sql:
                # Clean up whitespace
                sql = re.sub(r'\s+', ' ', sql).strip()
                cases.add(sql)
                
        # Add a few deep random cases to test recursion limits/complexity
        for _ in range(15):
            sql = grammar.generate(start, mode='random', depth_limit=15)
            if sql:
                sql = re.sub(r'\s+', ' ', sql).strip()
                cases.add(sql)
                
        return list(cases)

class Grammar:
    def __init__(self, path):
        self.rules = {}      # Map[lhs, list of alternatives]
        self.coverage = {}   # Map[lhs, set of visited alternative indices]
        self.gens = {}       # Custom generator functions
        self.start_symbol = None
        self.load(path)
        
    def set_generator(self, key, func):
        self.gens[key] = func
        self.gens[f"<{key}>"] = func
        self.gens[key.upper()] = func
        
    def load(self, path):
        try:
            with open(path, 'r') as f:
                data = f.read()
        except:
            return
            
        # Strip comments
        data = re.sub(r'#.*', '', data)
        
        # Identify rules. Assumption: standard BNF-like "<rule> ::= ..." or "rule ::= ..."
        # We prepend a newline to make the regex matching cleaner for the first rule
        data = "\n" + data
        
        # Regex finds the start of a rule definition
        rule_start_pattern = re.compile(r'\n\s*(<[\w-]+>|[\w]+)\s*::=')
        
        parts = rule_start_pattern.split(data)
        # parts[0] is pre-grammar text, then parts[1]=lhs, parts[2]=rhs, parts[3]=lhs...
        
        for i in range(1, len(parts), 2):
            lhs = parts[i].strip()
            rhs = parts[i+1].strip()
            
            if not self.start_symbol:
                self.start_symbol = lhs
                
            self.rules[lhs] = self.parse_rhs(rhs)
            self.coverage[lhs] = set()
            
    def parse_rhs(self, rhs):
        # Tokenize the RHS. We preserve quoted strings, brackets, and words.
        # Regex captures: quoted strings, brackets/pipes, identifiers in <> or plain words
        tokens = re.findall(r"""('[^']*'|"[^"]*"|[\[\]\{\}\(\)\|]|<[^>]+>|[\w]+)""", rhs)
        return self.parse_alts(tokens)
        
    def parse_alts(self, tokens):
        # Parses "A B | C D" into [[A, B], [C, D]]
        alts = []
        chunk = []
        depth = 0
        for t in tokens:
            if t == '|' and depth == 0:
                alts.append(self.parse_seq(chunk))
                chunk = []
            else:
                if t in '[{(': depth += 1
                elif t in ']})': depth -= 1
                chunk.append(t)
        if chunk:
            alts.append(self.parse_seq(chunk))
        return alts
        
    def parse_seq(self, tokens):
        # Parses a sequence of tokens, handling nested blocks [ ], { }, ( )
        seq = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t in '[{(':
                closer = {'[':']', '{':'}', '(':')'}[t]
                depth = 1
                j = i + 1
                while j < len(tokens) and depth > 0:
                    if tokens[j] == t: depth += 1
                    elif tokens[j] == closer: depth -= 1
                    j += 1
                
                # Recursive parse of the inner block
                block_tokens = tokens[i+1:j-1]
                block_alts = self.parse_alts(block_tokens)
                
                type_map = {'[': 'OPT', '{': 'REP', '(': 'GRP'}
                seq.append((type_map[t], block_alts))
                
                i = j - 1
            else:
                # Terminal or NonTerminal
                if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
                    seq.append(('TERM', t[1:-1]))
                elif t.startswith('<'):
                    seq.append(('NON', t))
                else:
                    # Heuristic: Uppercase usually keywords, mixed/lower usually rules
                    if t.isupper():
                        seq.append(('TERM', t))
                    else:
                        seq.append(('NON', t))
            i += 1
        return seq

    def generate(self, symbol, mode='random', depth_limit=20, stack=()):
        if len(stack) > depth_limit:
            return ""
        
        # Normalize symbol key
        key = symbol
        if key not in self.rules:
            if f"<{key}>" in self.rules: key = f"<{key}>"
            elif key.startswith('<') and key[1:-1] in self.rules: key = key[1:-1]
            
        # Check custom generators
        if key in self.gens:
            return self.gens[key]()
            
        # If not a rule, treat as literal
        if key not in self.rules:
            if (key.startswith("'") and key.endswith("'")) or (key.startswith('"') and key.endswith('"')):
                return key[1:-1]
            return key
            
        alts = self.rules[key]
        if not alts: return ""
        
        # Select alternative
        indices = list(range(len(alts)))
        if mode == 'coverage':
            unvisited = [idx for idx in indices if idx not in self.coverage[key]]
            if unvisited:
                idx = random.choice(unvisited)
                self.coverage[key].add(idx)
            else:
                idx = random.choice(indices)
        else:
            idx = random.choice(indices)
            
        return self.expand(alts[idx], mode, depth_limit, stack + (key,))
        
    def expand(self, seq, mode, depth_limit, stack):
        out = []
        for type_, val in seq:
            if type_ == 'TERM':
                out.append(val)
            elif type_ == 'NON':
                out.append(self.generate(val, mode, depth_limit, stack))
            elif type_ == 'GRP':
                # val is list of alternatives
                if val:
                    alt = random.choice(val)
                    out.append(self.expand(alt, mode, depth_limit, stack))
            elif type_ == 'OPT':
                # Bias towards inclusion (60%) to exercise code
                if val and random.random() < 0.6:
                    alt = random.choice(val)
                    out.append(self.expand(alt, mode, depth_limit, stack))
            elif type_ == 'REP':
                # Repeat 1-3 times
                if val:
                    count = random.randint(1, 3)
                    for _ in range(count):
                        alt = random.choice(val)
                        out.append(self.expand(alt, mode, depth_limit, stack))
        return " ".join([s for s in out if s])