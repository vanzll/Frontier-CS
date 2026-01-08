import collections
import os
import random
import re
import string
import time


class Solution:
    """
    Solution class to generate the fuzzer code.
    """
    def solve(self, resources_path: str) -> dict:
        """
        Reads the SQL grammar and embeds it into a self-contained fuzzer code string.

        Args:
            resources_path: Path to the directory containing `sql_grammar.txt`.

        Returns:
            A dictionary containing the fuzzer code as a string.
        """
        grammar_path = os.path.join(resources_path, 'sql_grammar.txt')
        try:
            with open(grammar_path, 'r', encoding='utf-8') as f:
                grammar_content = f.read()
        except FileNotFoundError:
            # Provide a minimal fallback grammar if the file is unexpectedly missing.
            grammar_content = "sql_stmt ::= 'SELECT' '1' ';' "

        fuzzer_code = f"""
# Fuzzer implementation starts here
import collections
import random
import re
import string
import time

# --- Embedded Grammar ---
# The SQL grammar is embedded directly to avoid file I/O in the fuzzer environment.
GRAMMAR_CONTENT = {repr(grammar_content)}

# --- Global State ---
# This dictionary holds state across multiple calls to the fuzz() function.
FUZZER_STATE = {{
    'grammar': None,
    'start_time': None,
    'rule_chooser': None,
    'call_count': 0,
}}

# --- Core Fuzzer Components ---

class GrammarParser:
    \"\"\"Parses the provided BNF-style grammar into a dictionary.\"\"\"
    def __init__(self, grammar_text):
        self.grammar = self._parse(grammar_text)

    def _parse(self, text):
        grammar = collections.defaultdict(list)
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('#'):
                i += 1
                continue

            if '::=' in line:
                name, expansion = line.split('::=', 1)
                name = name.strip()
                
                current_alt = []
                
                def add_tokens_to_alt(text_part, alt):
                    tokens = re.findall(r"'[^']+'|\\S+", text_part)
                    alt.extend(tokens)

                add_tokens_to_alt(expansion.strip(), current_alt)
                
                i += 1
                while i < len(lines):
                    line = lines[i].strip()
                    if not line or line.startswith('#'):
                        i += 1
                        continue
                    if '::=' in line:
                        break

                    if line.startswith('|'):
                        if current_alt:
                            grammar[name].append(current_alt)
                        current_alt = []
                        add_tokens_to_alt(line[1:].strip(), current_alt)
                    else:
                        add_tokens_to_alt(line, current_alt)
                    i += 1
                
                if current_alt:
                    grammar[name].append(current_alt)
            else:
                i += 1
                
        return dict(grammar)

class RuleChooser:
    \"\"\"
    A stateful rule chooser that guides the fuzzer to explore all grammar rules.
    It prioritizes less frequently chosen rules to increase exploration.
    \"\"\"
    def __init__(self, grammar):
        self.counts = {{
            key: [0] * len(rules) for key, rules in grammar.items()
        }}

    def choose(self, non_terminal):
        if non_terminal not in self.counts or not self.counts[non_terminal]:
            return 0

        rule_counts = self.counts[non_terminal]
        
        # Use inverse weighting: less used rules get higher probability
        weights = [1.0 / (c + 1) for c in rule_counts]
        
        try:
            chosen_index = random.choices(range(len(rule_counts)), weights=weights, k=1)[0]
        except (ValueError, IndexError):
            chosen_index = random.randint(0, len(rule_counts) - 1)

        self.counts[non_terminal][chosen_index] += 1
        return chosen_index


class GrammarFuzzer:
    \"\"\"Generates SQL statements based on the parsed grammar.\"\"\"
    def __init__(self, grammar, rule_chooser):
        self.grammar = grammar
        self.rule_chooser = rule_chooser
        self.memo = {{}}

    def generate(self, start_symbol='sql_stmt', max_depth=8):
        self.memo.clear()
        try:
            return self._expand(start_symbol, 0, max_depth)
        except RecursionError:
            return "SELECT 'recursion_error_fallback';"

    def _expand(self, symbol, depth, max_depth):
        memo_key = (symbol, depth)
        if memo_key in self.memo:
            return self.memo[memo_key]

        if depth > max_depth:
            res = self._generate_terminal('NUMERIC_LITERAL')
            self.memo[memo_key] = res
            return res

        if symbol not in self.grammar:
            res = self._generate_terminal(symbol)
            self.memo[memo_key] = res
            return res

        rules = self.grammar[symbol]
        if not rules:
            return ""
        
        rule_idx = self.rule_chooser.choose(symbol)
        chosen_rule = rules[rule_idx]

        parts = [self._expand(part, depth + 1, max_depth) for part in chosen_rule]
        
        res = " ".join(filter(None, parts))
        self.memo[memo_key] = res
        return res

    def _generate_terminal(self, terminal):
        if terminal.startswith("'") and terminal.endswith("'"):
            return terminal[1:-1]

        if terminal == 'IDENTIFIER':
            return random.choice(string.ascii_lowercase) + ''.join(random.choices(string.ascii_lowercase + string.digits + '_', k=random.randint(0, 9)))
        if terminal == 'STRING_LITERAL':
            content = ''.join(random.choices(string.ascii_letters + string.digits + ' _-', k=random.randint(0, 20)))
            return f"'{content.replace("'", "''")}'"
        if terminal == 'NUMERIC_LITERAL':
            if random.random() < 0.2:
                return str(random.choice([0, 1, -1, 100, -100]))
            return str(random.randint(-1000, 10000)) if random.random() > 0.3 else f"{{random.uniform(-1000.0, 1000.0):.4f}}"
        if terminal == 'BLOB_LITERAL':
            val = ''.join(random.choices('0123456789abcdef', k=random.randint(1, 20) * 2))
            return f"x'{{val}}'"
        
        return ""

# --- Mutation and Edge Case Generation ---

def mutate(s):
    \"\"\"Applies random mutations to a string to generate invalid/edge-case inputs.\"\"\"
    if not s or random.random() < 0.05:
        return s

    s_mutated = list(s)
    num_mutations = random.randint(1, max(2, len(s_mutated) // 8))
    ops = [';', ',', '(', ')', "'", '*', '`', '"', '-', '\\t', '\\n'] + list(string.ascii_letters + string.digits)

    for _ in range(num_mutations):
        if not s_mutated: break
        pos = random.randint(0, len(s_mutated))
        choice = random.random()

        if choice < 0.3 and len(s_mutated) > 0:
            s_mutated.pop(pos - 1)
        elif choice < 0.6:
            s_mutated.insert(pos, random.choice(ops))
        elif choice < 0.9 and len(s_mutated) > 0:
            s_mutated[pos - 1] = random.choice(ops)
        elif len(s_mutated) > 5:
            end_pos = min(len(s_mutated), pos + random.randint(1, 5))
            chunk = s_mutated[pos:end_pos]
            s_mutated = s_mutated[:pos] + chunk + s_mutated[pos:]
            
    return "".join(s_mutated)

def get_edge_cases():
    \"\"\"Returns a list of handcrafted edge cases and generated variants.\"\"\"
    base_cases = [
        "", ";", "SELECT;", "SELECT", "SELECT *", "SELECT * FROM", 
        "SELECT 1,;", "UPDATE t SET c = 1 WHERE ;",
        "INSERT INTO t VALUES ();", "INSERT INTO t () VALUES ();",
        "CREATE TABLE t ();", "CREATE TABLE t (c1 INT,);",
        "CREATE TABLE SELECT (INSERT TEXT);", "SELECT FROM FROM WHERE WHERE;",
        "SELECT 'üñîçødé' FROM t;", "SELECT * FROM `t-with-hyphen`;",
        "SELECT 9223372036854775807, -9223372036854775808;",
        "SELECT 1.79e+308, 2.22e-308;",
        "SELECT * FROM t WHERE (x=1", "SELECT 'hello",
    ]
    
    base_cases.append("SELECT " + ",".join(["c" + str(i) for i in range(random.randint(100, 200))]) + " FROM t;")
    base_cases.append("SELECT " + "( " * random.randint(50, 100) + "1" + ") " * random.randint(50, 100) + ";")
    base_cases.append("".join(random.choices(string.printable, k=random.randint(500, 1000))))
    return base_cases

# --- Main Fuzzer Entrypoint ---

def fuzz(parse_sql):
    \"\"\"
    The main fuzzing loop called by the evaluator.
    It generates and executes batches of SQL statements.
    \"\"\"
    if FUZZER_STATE['start_time'] is None:
        FUZZER_STATE['start_time'] = time.time()
        parser = GrammarParser(GRAMMAR_CONTENT)
        FUZZER_STATE['grammar'] = parser.grammar
        if not parser.grammar:
             FUZZER_STATE['grammar'] = {{'sql_stmt': [['SELECT', 'NUMERIC_LITERAL']]}}
        FUZZER_STATE['rule_chooser'] = RuleChooser(FUZZER_STATE['grammar'])

    if time.time() - FUZZER_STATE['start_time'] > 58.5:
        return False

    FUZZER_STATE['call_count'] += 1
    
    fuzzer = GrammarFuzzer(
        FUZZER_STATE['grammar'], 
        FUZZER_STATE['rule_chooser']
    )

    batch_size = 500
    statements = []
    
    strategy = FUZZER_STATE['call_count'] % 5

    try:
        if strategy == 0:
            for _ in range(batch_size):
                statements.append(fuzzer.generate(start_symbol='sql_stmt', max_depth=random.randint(3, 6)))
        
        elif strategy == 1:
            for _ in range(batch_size):
                statements.append(fuzzer.generate(start_symbol='sql_stmt', max_depth=random.randint(7, 12)))

        elif strategy == 2:
            half_batch = batch_size // 2
            base = [fuzzer.generate('sql_stmt', 7) for _ in range(half_batch)]
            mutated = [mutate(s) for s in base]
            statements.extend(base)
            statements.extend(mutated)
        
        elif strategy == 3:
            statements.extend(get_edge_cases())
            remaining = batch_size - len(statements)
            for _ in range(remaining):
                statements.append(fuzzer.generate('sql_stmt', 5))
        
        else:
            for _ in range(batch_size):
                depth = random.choice([4, 5, 6, 7, 8, 10])
                stmt = fuzzer.generate('sql_stmt', depth)
                if random.random() < 0.3:
                    stmt = mutate(stmt)
                statements.append(stmt)

    except Exception:
        pass

    final_statements = [s for s in statements if s and isinstance(s, str)]
    
    if final_statements:
        parse_sql(final_statements)

    return True
"""
        return {"code": fuzzer_code}