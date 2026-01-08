import os
import re
import random
from collections import defaultdict

class Node:
    pass

class Literal(Node):
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return f"'{self.value}'"

class NonTerminal(Node):
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return self.name

class Sequence(Node):
    def __init__(self, items):
        self.items = items
    def __repr__(self):
        return f"Sequence({self.items})"

class Alternative(Node):
    def __init__(self, options):
        self.options = options
    def __repr__(self):
        return f"Alternative({self.options})"

class Optional(Node):
    def __init__(self, item):
        self.item = item
    def __repr__(self):
        return f"Optional({self.item})"

class Repetition(Node):
    def __init__(self, item):
        self.item = item
    def __repr__(self):
        return f"Repetition({self.item})"

class Grammar:
    def __init__(self, text):
        self.rules = {}
        self.start_symbol = None
        self.parse(text)

    def parse(self, text):
        # Remove comments and normalize whitespace
        lines = [line.split('#')[0].strip() for line in text.split('\n')]
        full_text = " ".join(l for l in lines if l)
        
        # Split rules by ::=
        # Pattern captures the rule name
        parts = re.split(r'(<[\w-]+>)\s*::=', full_text)
        
        # parts[0] is preamble/empty, then alternating name, definition
        for i in range(1, len(parts), 2):
            name = parts[i].strip()
            definition = parts[i+1].strip()
            if not self.start_symbol:
                self.start_symbol = name
            self.rules[name] = self.parse_definition(definition)

    def parse_definition(self, text):
        # Tokenizer for grammar definition
        # Matches: <ref>, "lit", 'lit', bare_word, control_chars
        token_pattern = re.compile(r'(<[\w-]+>)|"([^"]+)"|\'([^\']+)\'|([a-zA-Z0-9_]+)|([|\[\]{}()])')
        tokens = []
        for m in token_pattern.finditer(text):
            if m.group(1): tokens.append(('REF', m.group(1)))
            elif m.group(2): tokens.append(('LIT', m.group(2)))
            elif m.group(3): tokens.append(('LIT', m.group(3)))
            elif m.group(4): tokens.append(('LIT', m.group(4)))
            elif m.group(5): tokens.append(('CTL', m.group(5)))
            
        self.tokens = tokens
        self.pos = 0
        return self.parse_alternatives()

    def parse_alternatives(self):
        options = []
        current_seq = []
        
        while self.pos < len(self.tokens):
            typ, val = self.tokens[self.pos]
            if typ == 'CTL':
                if val == '|':
                    options.append(self._seq_or_single(current_seq))
                    current_seq = []
                    self.pos += 1
                elif val in ']})':
                    # End of group/optional/repetition
                    break
                elif val == '[':
                    self.pos += 1
                    current_seq.append(Optional(self.parse_alternatives()))
                    if self.pos < len(self.tokens) and self.tokens[self.pos][1] == ']':
                        self.pos += 1
                elif val == '{':
                    self.pos += 1
                    current_seq.append(Repetition(self.parse_alternatives()))
                    if self.pos < len(self.tokens) and self.tokens[self.pos][1] == '}':
                        self.pos += 1
                elif val == '(':
                    self.pos += 1
                    current_seq.append(self.parse_alternatives())
                    if self.pos < len(self.tokens) and self.tokens[self.pos][1] == ')':
                        self.pos += 1
                else:
                    self.pos += 1
            else:
                if typ == 'REF':
                    current_seq.append(NonTerminal(val))
                else:
                    current_seq.append(Literal(val))
                self.pos += 1
        
        if current_seq:
            options.append(self._seq_or_single(current_seq))
        elif not options:
            # Empty
            pass
            
        if not options: return Sequence([])
        if len(options) == 1: return options[0]
        return Alternative(options)

    def _seq_or_single(self, seq):
        if not seq: return Sequence([])
        if len(seq) == 1: return seq[0]
        return Sequence(seq)

class Generator:
    def __init__(self, grammar):
        self.grammar = grammar
        self.counts = defaultdict(int) # Tracks (node_id, alternative_index) usage
        self.depths = defaultdict(int) # Tracks recursion depth per rule

    def generate(self, node, depth=0):
        # Global depth limit to prevent excessively long queries
        if depth > 15:
            return self.generate_minimal(node)
            
        if isinstance(node, Literal):
            return node.value
            
        elif isinstance(node, NonTerminal):
            # Per-rule recursion limit
            if self.depths[node.name] > 3:
                return self.generate_minimal(node)
            
            rule = self.grammar.rules.get(node.name)
            if not rule:
                return self.fallback(node.name)
            
            self.depths[node.name] += 1
            res = self.generate(rule, depth + 1)
            self.depths[node.name] -= 1
            return res
            
        elif isinstance(node, Sequence):
            return " ".join(self.generate(item, depth) for item in node.items)
            
        elif isinstance(node, Alternative):
            # Prioritize least covered alternatives
            # We use (id(node), index) as a key
            counts = [self.counts[(id(node), i)] for i in range(len(node.options))]
            min_count = min(counts)
            candidates = [i for i, c in enumerate(counts) if c == min_count]
            
            choice_idx = random.choice(candidates)
            self.counts[(id(node), choice_idx)] += 1
            
            return self.generate(node.options[choice_idx], depth)
            
        elif isinstance(node, Optional):
            # Bias towards including optional elements for better coverage
            if random.random() < 0.7:
                return self.generate(node.item, depth)
            return ""
            
        elif isinstance(node, Repetition):
            # Generate 1 to 3 items
            count = random.randint(1, 3)
            return " ".join(self.generate(node.item, depth) for _ in range(count))
            
        return ""

    def generate_minimal(self, node):
        """Generate the shortest/simplest possible string to terminate recursion."""
        if isinstance(node, Literal):
            return node.value
        elif isinstance(node, NonTerminal):
            if self.depths[node.name] > 5: return self.fallback(node.name)
            rule = self.grammar.rules.get(node.name)
            if not rule: return self.fallback(node.name)
            self.depths[node.name] += 1
            res = self.generate_minimal(rule)
            self.depths[node.name] -= 1
            return res
        elif isinstance(node, Sequence):
            return " ".join(self.generate_minimal(item) for item in node.items)
        elif isinstance(node, Alternative):
            # Always pick the first option as a heuristic for "simplest"
            return self.generate_minimal(node.options[0])
        elif isinstance(node, Optional):
            return ""
        elif isinstance(node, Repetition):
            return self.generate_minimal(node.item)
        return ""

    def fallback(self, name):
        """Provide defaults for undefined non-terminals."""
        u_name = name.upper()
        if 'IDENT' in u_name or 'NAME' in u_name or 'TABLE' in u_name or 'COL' in u_name:
            return "t1"
        if 'NUM' in u_name or 'INT' in u_name:
            return "1"
        if 'STR' in u_name or 'TEXT' in u_name:
            return "'val'"
        if 'BOOL' in u_name:
            return "TRUE"
        return "x"

class Solution:
    def solve(self, resources_path: str) -> list[str]:
        grammar_path = os.path.join(resources_path, 'sql_grammar.txt')
        with open(grammar_path, 'r') as f:
            grammar_text = f.read()
            
        grammar = Grammar(grammar_text)
        generator = Generator(grammar)
        
        test_cases = []
        # Generate 40 test cases.
        # This number balances the efficiency bonus (decreases with N)
        # against the need for high coverage (increases with N).
        # Since the generator prioritizes uncovered paths, 40 should be sufficient.
        
        if grammar.start_symbol:
            start_node = NonTerminal(grammar.start_symbol)
            for _ in range(40):
                # We reset recursion depths per query but keep coverage counts
                # to encourage diversity across queries.
                generator.depths.clear()
                sql = generator.generate(start_node)
                # Simple cleanup for formatting
                sql = re.sub(r'\s+', ' ', sql).strip()
                test_cases.append(sql)
                
        return test_cases