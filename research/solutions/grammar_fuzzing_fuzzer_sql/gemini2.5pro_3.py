import os
import re

class Solution:
    def solve(self, resources_path: str) -> dict:
        """
        Reads and parses the SQL grammar, then injects it into the fuzzer code string.
        """
        
        def _parse_grammar(file_path: str) -> dict:
            """
            Parses a BNF-style grammar file into a dictionary.
            Handles multi-line rules and recognizes non-terminals, terminals, and literals.
            """
            grammar = {}
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
            except FileNotFoundError:
                # Fallback to an empty grammar if file not found
                return {}

            # Remove comments and normalize line endings
            content = re.sub(r'#.*', '', content)
            content = content.replace('\r\n', '\n').replace('\r', '\n')
            
            # Combine multi-line rules into a single logical line
            content = re.sub(r'\n\s*\|', ' |', content)
            
            # Split into individual rule definitions
            rule_defs = re.split(r'\n(?=\s*<)', content)

            for rule_def in rule_defs:
                rule_def = rule_def.strip()
                if not rule_def or '::=' not in rule_def:
                    continue
                
                parts = rule_def.split('::=', 1)
                non_terminal, productions_str = parts
                non_terminal = non_terminal.strip()
                
                if non_terminal not in grammar:
                    grammar[non_terminal] = []
                
                # Split productions by '|'
                productions = [p.strip() for p in productions_str.split('|')]
                for prod in productions:
                    if not prod: # Handle empty productions like <A> ::= <B> |
                        grammar[non_terminal].append(['<empty>'])
                        continue

                    # Tokenize the production rule, handling terminals, non-terminals, and literals
                    tokens = re.findall(r"<[^>]+>|'[^']+'|[^<'\s]+", prod)
                    if tokens:
                        grammar[non_terminal].append(tokens)

            # Ensure <empty> rule is defined if used, to produce nothing
            if any('<empty>' in rule for prod_list in grammar.values() for rule in prod_list):
                if '<empty>' not in grammar:
                    grammar['<empty>'] = [[]]
            
            return grammar

        grammar_path = os.path.join(resources_path, "sql_grammar.txt")
        grammar_dict = _parse_grammar(grammar_path)
        grammar_repr = repr(grammar_dict)

        # The fuzzer code is embedded in a string, with the parsed grammar injected.
        fuzzer_code = f"""
import random
import re
import string
import sys

# Set a higher recursion limit for the grammar generator if possible
try:
    sys.setrecursionlimit(2000)
except (ValueError, AttributeError):
    pass

class Fuzzer:
    """
    A hybrid fuzzer that combines grammar-based generation for seeding
    with mutation-based strategies for deep exploration.
    """
    def __init__(self, grammar):
        self.initialized = False
        self.grammar = grammar
        self.corpus = []
        
        # Fuzzing parameters
        self.batch_size = 500
        self.seeding_rounds = 10
        self.max_corpus_size = 2000
        self.min_corpus_size = 1000

        self.terminals = set()
        if self.grammar:
            self.start_symbols = [s for s in self.grammar if 'statement' in s or 'command' in s]
            if not self.start_symbols:
                # Fallback if no 'statement' symbol is found
                self.start_symbols = sorted([k for k in self.grammar.keys() if self.grammar[k]])[:5] or ['<sql_statement>']
                
            for productions in self.grammar.values():
                for rule in productions:
                    for symbol in rule:
                        if not symbol.startswith('<'):
                            self.terminals.add(symbol)
            self.terminals_list = list(self.terminals)
        else: # Minimal fallback if grammar parsing failed
            self.start_symbols = ['<sql_statement>']
            self.terminals_list = ['SELECT', 'FROM', 'WHERE', 'INSERT', 'INTO', 'VALUES', '1', "'a'"]

        # Interesting values for generation and mutation
        self.interesting_values = {{
            'int': ['0', '1', '-1', '2147483647', '-2147483648', '65535', '65536', '4294967295'],
            'string': ["''", "'test'", "'\"'", "'\\''", "'--'", "' OR 1=1 --'", "'%s%s'", "'\\n'", "NULL"],
            'identifier': ['t1', 'c1', 'a', '_', 'table' * 5]
        }}

    def generate_for_terminal(self, terminal: str) -> str:
        """Generates a concrete value for a terminal symbol."""
        if terminal == '<identifier>':
            if random.random() < 0.3:
                return random.choice(self.interesting_values['identifier'])
            return ''.join(random.choices(string.ascii_lowercase + '_', k=random.randint(1, 10)))
        if terminal == '<string_literal>':
            if random.random() < 0.3:
                return random.choice(self.interesting_values['string'])
            content = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=random.randint(0, 15)))
            return f"'{content.replace('\'', '\'\'')}'"
        if terminal in ('<integer>', '<numeric_literal>', '<signed_number>'):
            if random.random() < 0.3:
                return random.choice(self.interesting_values['int'])
            return str(random.randint(-10000, 10000))
        if terminal == '<empty>':
            return ''
        return '1' # Fallback for unknown non-terminals

    def generate_from_grammar(self, symbol: str, depth: int = 0, max_depth: int = 8) -> str:
        """Recursively generates a string from a grammar symbol."""
        if depth > max_depth or symbol not in self.grammar:
            if symbol.startswith('<'):
                return self.generate_for_terminal(symbol)
            return symbol

        productions = self.grammar.get(symbol)
        if not productions:
            return self.generate_for_terminal(symbol) if symbol.startswith('<') else symbol

        chosen_rule = random.choice(productions)
        
        result_parts = []
        for part in chosen_rule:
            try:
                res = self.generate_from_grammar(part, depth + 1, max_depth)
                result_parts.append(res)
            except RecursionError:
                result_parts.append(self.generate_for_terminal(part) if part.startswith('<') else part)

        return ' '.join(filter(None, result_parts))

    def get_tokens(self, statement: str) -> list[str]:
        """A simple, fast tokenizer for mutation purposes."""
        return [t for t in re.split(r'([,;()\s\\'.="<>!+-/*`])', statement) if t]

    def mutate(self, statement: str) -> str:
        """Applies a random mutation to a statement."""
        if not statement:
            return ""

        mutators = [
            self._delete_token, self._insert_token, self._replace_token,
            self._duplicate_token, self._swap_tokens, self._char_mutate
        ]
        mutator = random.choice(mutators)
        try:
            return mutator(statement)
        except Exception:
            return statement # Return original on any mutation error

    def _delete_token(self, statement: str) -> str:
        tokens = self.get_tokens(statement)
        if len(tokens) < 2: return statement
        del_idx = random.randrange(len(tokens))
        return "".join(tokens[:del_idx] + tokens[del_idx+1:])

    def _insert_token(self, statement: str) -> str:
        tokens = self.get_tokens(statement)
        ins_idx = random.randrange(len(tokens) + 1)
        new_token = random.choice(self.terminals_list + self.interesting_values['int'] + self.interesting_values['string'])
        return "".join(tokens[:ins_idx]) + new_token + "".join(tokens[ins_idx:])

    def _replace_token(self, statement: str) -> str:
        tokens = self.get_tokens(statement)
        if not tokens: return statement
        rep_idx = random.randrange(len(tokens))
        new_token = random.choice(self.terminals_list + self.interesting_values['int'] + self.interesting_values['string'])
        tokens[rep_idx] = new_token
        return "".join(tokens)

    def _duplicate_token(self, statement: str) -> str:
        tokens = self.get_tokens(statement)
        if not tokens: return statement
        dup_idx = random.randrange(len(tokens))
        return "".join(tokens[:dup_idx+1]) + tokens[dup_idx] + "".join(tokens[dup_idx+1:])

    def _swap_tokens(self, statement: str) -> str:
        tokens = self.get_tokens(statement)
        if len(tokens) < 2: return statement
        idx1, idx2 = random.sample(range(len(tokens)), 2)
        tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]
        return "".join(tokens)
    
    def _char_mutate(self, statement: str) -> str:
        if not statement: return statement
        pos = random.randrange(len(statement))
        mutation_type = random.random()
        if mutation_type < 0.33: # Replace
            new_char = random.choice(string.printable)
            return statement[:pos] + new_char + statement[pos+1:]
        elif mutation_type < 0.66: # Insert
            new_char = random.choice(string.printable)
            return statement[:pos] + new_char + statement[pos:]
        else: # Delete
            return statement[:pos] + statement[pos+1:]

    def do_fuzz(self, parse_sql):
        """Main fuzzing loop, managing seeding and mutation stages."""
        if not self.initialized:
            # Seeding Phase: Generate initial corpus from grammar
            initial_statements = set()
            if self.grammar:
                for _ in range(self.seeding_rounds):
                    batch = []
                    for _ in range(self.batch_size):
                        try:
                            symbol = random.choice(self.start_symbols)
                            gen = self.generate_from_grammar(symbol)
                            if gen and len(gen) > 1:
                                batch.append(gen)
                        except (RecursionError, IndexError):
                            pass
                    if batch:
                        parse_sql(batch)
                        initial_statements.update(batch)
            
            self.corpus = list(initial_statements)
            # Fallback to a hardcoded corpus if generation fails
            if not self.corpus:
                self.corpus = ["SELECT 1;", "INSERT INTO t VALUES (1, 'a');", "CREATE TABLE t (a INT);"]
            
            self.initialized = True
            return True

        # Mutation Phase
        batch = []
        for _ in range(self.batch_size):
            # 95% chance to mutate an existing entry, 5% to generate a new one
            if self.corpus and random.random() < 0.95:
                seed = random.choice(self.corpus)
                mutated = self.mutate(seed)
                batch.append(mutated)
            elif self.grammar:
                try:
                    symbol = random.choice(self.start_symbols)
                    gen = self.generate_from_grammar(symbol)
                    if gen and len(gen) > 1:
                        batch.append(gen)
                except (RecursionError, IndexError):
                    pass
        
        if batch:
            # Parse unique statements to avoid redundant work and reduce load
            parse_sql(list(set(batch)))

            # Probabilistically update the corpus with new inputs
            self.corpus.extend(random.sample(batch, min(len(batch), 20)))
            if len(self.corpus) > self.max_corpus_size:
                self.corpus = random.sample(self.corpus, self.min_corpus_size)
        
        return True

# Create a single global instance of the fuzzer.
# The grammar is injected here by the Solution.solve method.
fuzzer_instance = Fuzzer({grammar_repr})

def fuzz(parse_sql):
    """
    The main entry point for the fuzzer, called repeatedly by the evaluator.
    Delegates to the Fuzzer instance to perform the work.
    """
    return fuzzer_instance.do_fuzz(parse_sql)
"""
        return {"code": fuzzer_code}