import sys
import os

class Solution:
  def solve(self, resources_path: str) -> dict:
    fuzzer_code = """
import random
import re
from typing import List, Dict, Set, Optional

# --- CONSTANTS ---
MAX_DEPTH = 7
BATCH_SIZE = 2000
FUZZ_PROBABILITY = 0.1

# --- FUZZER IMPLEMENTATION ---

class GrammarFuzzer:
    \"\"\"
    A grammar-based fuzzer that generates SQL statements from a BNF-style grammar.
    It combines valid generation with random mutations to explore both valid syntax
    and error-handling paths of the parser.
    \"\"\"
    def __init__(self):
        self.grammar: Dict[str, List[List[str]]] = {}
        self.terminals: Set[str] = set()
        self.start_symbol: Optional[str] = None
        self.initialized: bool = False

    def initialize(self, grammar_path: str = './resources/sql_grammar.txt'):
        \"\"\"Parses the grammar file once and sets up the fuzzer state.\"\"\"
        if self.initialized:
            return
        self._parse_grammar(grammar_path)
        self.initialized = True

    def _parse_grammar(self, file_path: str):
        \"\"\"
        Parses a BNF-style grammar file. This parser is designed to be robust
        against various formatting styles, including multiline rules.
        \"\"\"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            content = re.sub(r'#.*', '', content)
            
            # Split grammar into definitions. A definition starts with <symbol> ::=
            definitions = re.split(r'\\s*(?=<[^>]+>\\s*::=)', content)
            definitions = [d.strip() for d in definitions if d.strip()]

            if not definitions: return

            match = re.match(r'<[^>]+>', definitions[0])
            if match:
                self.start_symbol = match.group(0)

            for definition in definitions:
                if '::=' not in definition: continue
                symbol, productions_str = definition.split('::=', 1)
                symbol = symbol.strip()
                
                if symbol not in self.grammar:
                    self.grammar[symbol] = []
                
                productions_str = productions_str.replace('\\n', ' ')
                productions = productions_str.split('|')

                for prod in productions:
                    prod = prod.strip()
                    if not prod: continue
                    
                    tokens = re.findall(r'<[^>]+>|\\S+', prod)
                    self.grammar[symbol].append(tokens)
                    for token in tokens:
                        if not token.startswith('<'):
                            self.terminals.add(token)
        except (FileNotFoundError, IndexError, ValueError):
            self.grammar = {} # parsing failed, will use fallback

    def generate_batch(self) -> List[str]:
        \"\"\"Generates a batch of SQL statements.\"\"\"
        if not self.grammar or not self.start_symbol:
            return self._generate_fallback_batch()
        return [self._generate_from_symbol(self.start_symbol) for _ in range(BATCH_SIZE)]

    def _generate_from_symbol(self, symbol: str, depth: int = 0) -> str:
        \"\"\"Recursively generates a string from a grammar symbol.\"\"\"
        if depth > MAX_DEPTH:
            return random.choice(['1', 'x', "''"])

        # Handle special primitive types defined in the grammar
        if symbol == '<identifier>': return self._gen_identifier()
        if symbol == '<literal_value>': return self._gen_literal_value()
        if symbol == '<string_literal>': return self._gen_string_literal()
        if symbol == '<numeric_literal>': return self._gen_numeric_literal()
        if symbol == '<boolean_literal>': return random.choice(['TRUE', 'FALSE'])

        # Handle non-terminals from the grammar rules
        if symbol in self.grammar:
            expansions = self.grammar.get(symbol, [])
            if not expansions:
                return ''

            if depth > MAX_DEPTH / 2: # Prefer shorter rules at high depth
                expansions = sorted(expansions, key=len)
                expansion = random.choice(expansions[:len(expansions)//2 + 1])
            else:
                expansion = random.choice(expansions)

            if random.random() < FUZZ_PROBABILITY:
                expansion = self._mutate_structure(expansion)
            
            return " ".join(self._generate_from_symbol(token, depth + 1) for token in expansion)

        # Handle terminals (keywords, operators)
        if not symbol.startswith('<'):
            if random.random() < FUZZ_PROBABILITY:
                return self._mutate_terminal(symbol)
            return symbol

        # Fallback for unknown <...> symbols: treat as a literal
        return symbol[1:-1]

    def _mutate_terminal(self, terminal: str) -> str:
        \"\"\"Applies a random mutation to a terminal symbol.\"\"\"
        actions = [
            lambda t: random.choice(list(self.terminals)) if self.terminals else 'FUZZ',
            lambda t: t + t,
            lambda t: '',
            lambda t: random.choice([';', '(', ')', '%', '`', "'", '"']),
        ]
        return random.choice(actions)(terminal)

    def _mutate_structure(self, expansion: List[str]) -> List[str]:
        \"\"\"Applies a random mutation to a list of grammar tokens.\"\"\"
        if not expansion: return []
        new_expansion = expansion[:]
        action = random.randint(0, 2)
        
        if action == 0 and len(new_expansion) > 1: # Swap
            i, j = random.sample(range(len(new_expansion)), 2)
            new_expansion[i], new_expansion[j] = new_expansion[j], new_expansion[i]
        elif action == 1 and len(new_expansion) > 0: # Duplicate
            idx = random.randrange(len(new_expansion))
            new_expansion.insert(idx, new_expansion[idx])
        elif action == 2 and len(new_expansion) > 0: # Delete
            idx = random.randrange(len(new_expansion))
            new_expansion.pop(idx)
        return new_expansion

    def _gen_identifier(self) -> str:
        return random.choice([
            't1', 'c1', 'users', 'id', 'name', '`a b`', '"c d"',
            'a' * 100, '_', '`' * 50, 'SELECT'
        ])

    def _gen_literal_value(self) -> str:
        return random.choice([
            self._gen_string_literal, self._gen_numeric_literal,
            lambda: 'NULL', lambda: random.choice(['TRUE', 'FALSE']),
        ])()

    def _gen_string_literal(self) -> str:
        return random.choice([
            "''", "'hello'", "'it''s a string'", '"double-quoted"',
            "'malformed", "'\\\\'", "'你好'", "'" + "a"*100 + "'",
            "'' -- comment injection"
        ])

    def _gen_numeric_literal(self) -> str:
        return random.choice([
            '0', '1', '-1', '1.0', '-1.5', '1e10', '1.2e-5', str(2**63-1),
            '9' * 50, '.1', '1.'
        ])

    def _generate_fallback_batch(self) -> List[str]:
        \"\"\"A simple keyword-based fuzzer for when grammar parsing fails.\"\"\"
        keywords = ['SELECT', 'FROM', 'WHERE', 'INSERT', 'INTO', 'VALUES', '*', '1', "'abc'"]
        batch = []
        for _ in range(BATCH_SIZE):
            stmt = " ".join(random.choices(keywords, k=random.randint(2, 8)))
            batch.append(stmt)
        return batch

# --- Singleton Fuzzer Instance ---
_FUZZER = GrammarFuzzer()

# --- Fuzzer Entrypoint ---
def fuzz(parse_sql):
    \"\"\"
    Main fuzzing loop function called by the evaluator.
    Generates and executes a large batch of SQL statements in each call.
    \"\"\"
    global _FUZZER
    if not _FUZZER.initialized:
        _FUZZER.initialize()
    
    statements = _FUZZER.generate_batch()
    parse_sql(statements)
    
    return True

"""
    return {"code": fuzzer_code}