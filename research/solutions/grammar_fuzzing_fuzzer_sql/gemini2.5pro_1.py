import os
import re

class Solution:
    def solve(self, resources_path: str) -> dict:
        
        def _parse_grammar(grammar_path: str) -> dict:
            """
            Parses the BNF-style grammar file into a dictionary representation.
            """
            grammar = {}
            try:
                with open(grammar_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        if '::=' not in line:
                            continue

                        name, rules_str = line.split('::=', 1)
                        name = name.strip()
                        
                        productions = []
                        rules = rules_str.strip().split('|')
                        
                        for rule in rules:
                            rule = rule.strip()
                            if not rule:
                                productions.append([])
                                continue
                            
                            tokens = re.findall(r"<[^>]+>|'[^']+'|\"[^\"]+\"|[^'\"<> ]+", rule)
                            productions.append([t.strip() for t in tokens])
                        
                        grammar[name] = productions
            except FileNotFoundError:
                return {}
            return grammar

        grammar_file_path = os.path.join(resources_path, 'sql_grammar.txt')
        grammar_dict = _parse_grammar(grammar_file_path)
        grammar_repr = repr(grammar_dict)

        # The fuzzer code is defined in a template string. The parsed grammar
        # will be embedded into this string. This makes the fuzzer self-contained.
        fuzzer_code_template = """
import random
import re
import string
import time

# --- Grammar is embedded directly by the Solution class ---
GRAMMAR = {GRAMMAR_PLACEHOLDER}

class Fuzzer:
    _instance = None

    @staticmethod
    def get_instance():
        if Fuzzer._instance is None:
            Fuzzer._instance = Fuzzer()
        return Fuzzer._instance

    def __init__(self):
        # Seed with time to ensure varied outputs across runs
        random.seed(int(time.time() * 1000))

        self.grammar = GRAMMAR
        if not self.grammar:
            raise RuntimeError("Grammar not loaded, cannot fuzz.")
            
        self.start_symbol = "<sql_statement>"
        self.max_depth = 8
        self.generation_count = 0
        self.batch_size = 200

        self.terminal_generators = {
            '<identifier>': self._gen_identifier,
            '<string_literal>': self._gen_string_literal,
            '<numeric_literal>': self._gen_numeric_literal,
            '<boolean_literal>': lambda: random.choice(['TRUE', 'FALSE']),
        }

        self.keywords = self._extract_keywords()
        self.common_identifiers = ['t1', 't2', 'c1', 'c2', 'id', 'name', 'value', 'data', 'users', 'products', 'orders']
        self.malicious_inputs = ['\\0', ';', '`', '#', '--', "' OR 1=1 --", '(', ')', '%', '_']

    def _extract_keywords(self):
        keywords = set()
        for productions in self.grammar.values():
            for production in productions:
                for token in production:
                    if not token.startswith('<') and not token.startswith("'") and not token.startswith('"') and token.isalpha():
                        keywords.add(token.upper())
        return list(keywords)
    
    def _gen_identifier(self):
        if random.random() < 0.7:
            return random.choice(self.common_identifiers)
        
        length = random.randint(1, 24)
        chars = string.ascii_letters + string.digits + '_'
        start_char = random.choice(string.ascii_letters + '_')
        rest_chars = ''.join(random.choice(chars) for _ in range(length - 1))
        ident = start_char + rest_chars
        
        if random.random() < 0.1:
            return f'"{{ident}}"'
        return ident

    def _gen_string_literal(self):
        r = random.random()
        if r < 0.8:
            length = random.randint(0, 40)
            chars = string.printable.replace("'", "") 
            s = ''.join(random.choice(chars) for _ in range(length))
            if random.random() < 0.3:
                s = s.replace('s', "''")
            return f"'{{s}}'"
        elif r < 0.95:
            payload = random.choice(self.malicious_inputs)
            escaped_payload = payload.replace("'", "''")
            return f"'{{escaped_payload}}'"
        else:
            return "''"

    def _gen_numeric_literal(self):
        r = random.random()
        if r < 0.5:
            return str(random.randint(-10000, 10000))
        elif r < 0.8:
            return f"{{random.uniform(-10000.0, 10000.0):.4f}}"
        elif r < 0.95:
            return f"{{random.randint(1, 9)}}E{{random.randint(-10, 10)}}"
        else:
            return random.choice(['0', '1', '-1', str(2**63 - 1), str(-2**63)])

    def _mutate_keyword(self, keyword):
        r = random.random()
        if r < 0.7: return keyword
        if r < 0.85: return keyword.lower()
        if r < 0.95: return keyword.upper()
        return ''.join(random.choice([c.upper(), c.lower()]) for c in keyword)

    def generate(self, symbol, depth):
        if depth > self.max_depth:
            return self._gen_identifier() if random.random() < 0.5 else self._gen_numeric_literal()

        if symbol not in self.grammar:
            # Terminal symbol
            if symbol in self.terminal_generators:
                return self.terminal_generators[symbol]()

            if (symbol.startswith("'") and symbol.endswith("'")) or \\
               (symbol.startswith('"') and symbol.endswith('"')):
                content = symbol[1:-1]
                if content.isalpha():
                    return self._mutate_keyword(content)
                return content

            if symbol.isalpha():
                return self._mutate_keyword(symbol)

            return symbol

        # Non-terminal symbol
        productions = self.grammar[symbol]
        
        if depth > self.max_depth / 2:
            productions.sort(key=len)
            idx = int(abs(random.gammavariate(2.0, 1.0)))
            chosen_production = productions[min(idx, len(productions) - 1)]
        else:
            chosen_production = random.choice(productions)

        if random.random() < 0.005:
            random_symbol = random.choice(list(self.grammar.keys()))
            if self.grammar.get(random_symbol):
                chosen_production = random.choice(self.grammar[random_symbol])

        parts = [self.generate(part, depth + 1) for part in chosen_production]

        if random.random() < 0.01 and len(parts) > 1:
            parts.pop(random.randrange(len(parts)))
        if random.random() < 0.005:
            garbage = random.choice(self.keywords + self.malicious_inputs)
            parts.insert(random.randrange(len(parts) + 1), garbage)

        return " ".join(filter(None, parts))

    def generate_statement(self):
        self.generation_count += 1
        
        current_max_depth = min(4 + self.generation_count // 1000, self.max_depth)
        
        start = self.start_symbol
        
        if random.random() < 0.05:
            major_non_terminals = [s for s in self.grammar if 'expr' in s or 'clause' in s or 'statement' in s]
            if major_non_terminals:
                start = random.choice(major_non_terminals)

        statement = self.generate(start, 0)
        
        if random.random() < 0.05:
            statement += random.choice([';', ' ', ' -- comment'])
        
        return statement.strip()

def fuzz(parse_sql):
    """
    Generate SQL statements using a grammar-based fuzzer and execute them.
    """
    try:
        fuzzer = Fuzzer.get_instance()
        
        statements = [fuzzer.generate_statement() for _ in range(fuzzer.batch_size)]
        
        parse_sql(statements)
    except Exception:
        # If the fuzzer itself has an error, stop to avoid flooding logs.
        return False
    
    return True
"""
        
        final_code = fuzzer_code_template.replace("{GRAMMAR_PLACEHOLDER}", grammar_repr, 1)

        return {"code": final_code}