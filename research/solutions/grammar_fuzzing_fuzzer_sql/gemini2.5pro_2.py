import os
import re

class Solution:
  def solve(self, resources_path: str) -> dict:
    grammar_path = os.path.join(resources_path, "sql_grammar.txt")
    with open(grammar_path, 'r') as f:
        grammar_content = f.read()

    fuzzer_code = f"""
import random
import re
import string
import sys

# A higher recursion limit might be needed for deeply nested grammar rules.
# This is safe because the generator has its own depth checks.
sys.setrecursionlimit(2000)

# --- Embedded Grammar ---
# The SQL grammar is embedded directly into the script by the Solution class.
GRAMMAR_TEXT = {repr(grammar_content)}

# --- Fuzzer Implementation ---

class GrammarFuzzer:
    \"\"\"
    A grammar-based fuzzer designed to maximize SQL parser coverage.

    Key Features:
    - Systematic Phase: Ensures all top-level statement types are generated,
      quickly covering major parser branches.
    - Random Phase: Explores complex and nested syntactic structures through
      randomized grammar traversal.
    - Mutation Engine: Creates near-miss invalid inputs to test error handling.
    - Malformed Generation: Targets the tokenizer and low-level parsing logic.
    - High-Efficiency Batching: Generates large batches to optimize the efficiency
      score.
    \"\"\"

    def __init__(self, grammar_text):
        self.grammar = self._parse_grammar(grammar_text)
        
        self.root_symbol = "<sql_statement>"
        if self.root_symbol not in self.grammar:
            # Fallback to find a likely root symbol
            for key in self.grammar:
                if "statement" in key or "stmt" in key:
                    self.root_symbol = key
                    break
            else:
                self.root_symbol = next(iter(self.grammar.keys()), None)

        self.max_depth = 8
        self.batch_size = 5000  # Large batch for efficiency
        
        self.phase = 'systematic'
        self.systematic_phase_step = 0
        self.root_rules = self.grammar.get(self.root_symbol, []) if self.root_symbol else []
        
        self.keywords = self._extract_keywords()

    def _parse_grammar(self, grammar_text):
        grammar = {{}}
        current_nt = None
        for line in grammar_text.splitlines():
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            
            match = re.match(r'^(<[^>]+>)\\s*::=', line)
            if match:
                current_nt = match.group(1)
                body = line[match.end():]
                if current_nt not in grammar:
                    grammar[current_nt] = []
            elif current_nt and (line.startswith('|') or line.startswith('::=')):
                body = re.sub(r'^(\\|::=)\\s*', '', line)
            elif current_nt:
                body = line
            else:
                continue

            alternatives = body.split('|')
            for alt in alternatives:
                alt = alt.strip()
                if not alt: continue
                tokens = re.findall(r'<[^>]+>|"[^"]+"|\'[^\']+\'|[A-Z_][A-Z0-9_]*', alt)
                rule = [('NT', t) if t.startswith('<') else ('T', t.strip('"\\'')) for t in tokens]
                if rule:
                    grammar[current_nt].append(rule)
        return grammar

    def _extract_keywords(self):
        keywords = set()
        for rules in self.grammar.values():
            for rule in rules:
                for part_type, part_value in rule:
                    if part_type == 'T' and part_value.isalpha() and part_value.upper() == part_value:
                        keywords.add(part_value)
        return list(keywords) if keywords else ["SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES"]

    def _generate_from_rule(self, rule, depth, max_depth):
        parts = []
        for part_type, part_value in rule:
            if part_type == 'NT':
                parts.append(self.generate_for_symbol(part_value, depth + 1, max_depth))
            else:
                parts.append(part_value)
        return " ".join(filter(None, parts))

    def generate_for_symbol(self, symbol, depth=0, max_depth=None):
        max_depth = max_depth if max_depth is not None else self.max_depth

        # Specialized generators for common lexical tokens
        if symbol == "<identifier>": return self.generate_identifier()
        if symbol == "<string_literal>": return self.generate_string_literal()
        if symbol == "<numeric_literal>": return self.generate_numeric_literal()
        if symbol == "<boolean_literal>": return random.choice(["TRUE", "FALSE"])

        if depth > max_depth:
            # At max depth, find a non-recursive, preferably short, rule to terminate
            if symbol in self.grammar:
                terminating_rules = [r for r in self.grammar[symbol] if not any(pt == 'NT' and pv == symbol for pt, pv in r)]
                if terminating_rules:
                    chosen_rule = min(terminating_rules, key=len)
                    return self._generate_from_rule(chosen_rule, depth, max_depth + 5)
            return "" # Terminate if no other option

        if symbol not in self.grammar:
            return symbol # A bareword terminal like SEMICOLON

        rules = self.grammar.get(symbol, [])
        if not rules:
            return ""

        return self._generate_from_rule(random.choice(rules), depth, max_depth)

    def generate_identifier(self):
        length = random.choices([1, 4, 8, 16, 32], weights=[10, 20, 40, 20, 10], k=1)[0]
        name = ''.join(random.choice(string.ascii_lowercase + "_") for _ in range(length))
        
        choice = random.random()
        if choice < 0.1: return f'"{{name}}"'
        if choice < 0.15 and self.keywords: return random.choice(self.keywords)
        if choice < 0.2: return name + str(random.randint(0, 99))
        return name

    def generate_string_literal(self):
        length = random.choices([0, 1, 10, 50], weights=[20, 20, 40, 20], k=1)[0]
        content = ''.join(random.choice(string.printable) for _ in range(length))
        content = content.replace("'", "''").replace("\\\\", "\\\\\\\\")
        return f"'{{content}}'"

    def generate_numeric_literal(self):
        choice = random.random()
        if choice < 0.5: return str(random.randint(-1000, 1000))
        if choice < 0.8: return str(random.uniform(-1e5, 1e5))
        if choice < 0.9: return str(random.choice([0, 1, -1, 100, 10000000000000000]))
        else: return f"{{random.uniform(-1,1)}}E{{random.randint(-20, 20)}}"

    def mutate(self, statement):
        if not statement: return ""
        tokens = re.split(r'(\\s+)', statement)
        if len(tokens) < 2: return statement
        
        mutation_type = random.randint(0, 3)
        idx = random.randrange(len(tokens))

        if mutation_type == 0:
            tokens.pop(idx)
        elif mutation_type == 1:
            token = random.choice(self.keywords + ["'", '"', '(', ')', ';', ',', '123', self.generate_identifier()])
            tokens.insert(idx, token)
        elif mutation_type == 2 and len(tokens) > 1:
            idx2 = random.randrange(len(tokens))
            tokens[idx], tokens[idx2] = tokens[idx2], tokens[idx]
        else:
            tokens.insert(idx, tokens[idx])
            
        return "".join(tokens)

    def generate_malformed(self):
        options = ["'unmatched quote", "(unmatched paren", "1.2.3.4", ";;"]
        if self.keywords:
            options.append(f"{{random.choice(self.keywords)}} {{random.choice(self.keywords)}}")
        options.append(''.join(random.choice(string.printable) for _ in range(random.randint(10, 50))))
        return random.choice(options)

    def generate_batch(self):
        statements = []
        
        if self.phase == 'systematic':
            if self.systematic_phase_step < len(self.root_rules):
                target_rule = self.root_rules[self.systematic_phase_step]
                for _ in range(self.batch_size):
                    statements.append(self._generate_from_rule(target_rule, 0, random.randint(5, 15)))
                self.systematic_phase_step += 1
            else:
                self.phase = 'random'
        
        if self.phase == 'random':
            if not self.root_symbol: return [] # Cannot generate without a root
            for _ in range(self.batch_size):
                statements.append(self.generate_for_symbol(self.root_symbol, 0, random.randint(4, 12)))
        
        num_mutations = int(len(statements) * 0.15)
        if statements:
            base_for_mutation = random.sample(statements, min(num_mutations, len(statements)))
            statements.extend(self.mutate(s) for s in base_for_mutation)
            
        num_malformed = int(self.batch_size * 0.1)
        statements.extend(self.generate_malformed() for _ in range(num_malformed))

        random.shuffle(statements)
        return statements

# --- Fuzzer Entrypoint ---

fuzzer_instance = None

def fuzz(parse_sql):
    global fuzzer_instance
    
    if fuzzer_instance is None:
        fuzzer_instance = GrammarFuzzer(GRAMMAR_TEXT)

    statements = fuzzer_instance.generate_batch()
    if not statements:
        # Stop if generation fails (e.g., empty grammar)
        return False

    parse_sql(statements)

    return True
"""
    return {"code": fuzzer_code}