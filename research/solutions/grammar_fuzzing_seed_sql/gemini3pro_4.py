import os
import re
import random
import collections

class Solution:
    def solve(self, resources_path: str) -> list[str]:
        grammar_path = os.path.join(resources_path, "sql_grammar.txt")
        self.rules = {}
        self.rule_order = []
        
        # 1. Parse Grammar
        try:
            with open(grammar_path, 'r') as f:
                content = f.read()
            self.parse_grammar(content)
        except Exception:
            # Fallback if grammar is unreadable
            return ["SELECT * FROM t1;"]

        if not self.rules:
            return ["SELECT 1;"]

        # 2. Identify Start Symbol (Heuristic: First rule defined)
        self.start_symbol = self.rule_order[0]

        # 3. Pre-calculate Shortest Expansions for Termination
        self.shortest_expansions = {}
        self.calculate_shortest_expansions()

        # 4. Generate Test Cases
        generated = set()
        self.production_counts = collections.defaultdict(int)
        
        target_count = 60
        attempts = 0
        max_attempts = 400
        
        while len(generated) < target_count and attempts < max_attempts:
            attempts += 1
            # Randomize depth to exercise different paths
            max_depth = random.randint(5, 20)
            
            try:
                tokens = self.generate(self.start_symbol, 0, max_depth)
                sql = self.post_process(tokens)
                if sql and sql not in generated:
                    generated.add(sql)
            except (RecursionError, Exception):
                continue

        return list(generated)

    def parse_grammar(self, text):
        # Basic cleaning: remove comments
        lines = [line.split('#')[0].strip() for line in text.splitlines() if line.strip()]
        full_text = " ".join(lines)
        
        # Tokenizer
        # Preserves <rules>, ::=, |, [, ], {, }, (, ), and terminals (quoted or alphanumeric)
        token_re = re.compile(r'(<[\w-]+>)|(::=)|(\|)|(\[)|(\])|(\{)|(\})|(\()|(\))|(\'[^\']*\')|("[^"]*")|([a-zA-Z0-9_<>!=*]+)')
        tokens = [m.group(0) for m in token_re.finditer(full_text)]
        
        self.scan_rules(tokens)

    def scan_rules(self, tokens):
        i = 0
        current_rule = None
        rhs_buffer = []
        
        while i < len(tokens):
            t = tokens[i]
            # Check for rule definition: <rule> ::=
            if t.startswith('<') and i + 1 < len(tokens) and tokens[i+1] == '::=':
                if current_rule:
                    self.process_rule_body(current_rule, rhs_buffer)
                
                current_rule = t
                if current_rule not in self.rule_order:
                    self.rule_order.append(current_rule)
                # Reset rules entry to allow for accumulation or fresh start. 
                # Assuming standard BNF where rules are defined once or concatenated.
                if current_rule not in self.rules:
                    self.rules[current_rule] = []
                    
                rhs_buffer = []
                i += 2 # Skip name and ::=
            else:
                if current_rule:
                    rhs_buffer.append(t)
                i += 1
        
        if current_rule and rhs_buffer:
             self.process_rule_body(current_rule, rhs_buffer)

    def process_rule_body(self, rule_name, tokens):
        # Recursive scanner to handle nested groups [ ] { } ( )
        def scan(idx, tokens):
            alts = []
            current_alt = []
            i = idx
            while i < len(tokens):
                t = tokens[i]
                if t == '|':
                    alts.append(current_alt)
                    current_alt = []
                    i += 1
                elif t in ['[', '{', '(']:
                    closer = { '[':']', '{':'}', '(':')' }[t]
                    # Find matching closer
                    depth = 1
                    j = i + 1
                    while j < len(tokens):
                        if tokens[j] == t: depth += 1
                        elif tokens[j] == closer: depth -= 1
                        if depth == 0: break
                        j += 1
                    
                    if j >= len(tokens): # Unmatched
                        current_alt.append(t)
                        i += 1
                        continue

                    inner_content = tokens[i+1:j]
                    
                    # Create aux rule
                    aux_name = f"<{rule_name.strip('<>')}_aux_{random.randint(0, 1000000)}>"
                    if aux_name not in self.rules: self.rules[aux_name] = []
                    
                    # Recursively process inner content
                    self.process_rule_body(aux_name, inner_content)
                    
                    # Modify aux rule based on type
                    if t == '[':
                        # Optional: Add empty alt
                        self.rules[aux_name].append([])
                    elif t == '{':
                        # Repetition: Make recursive
                        # aux -> inner | inner aux
                        base_alts = list(self.rules[aux_name])
                        recursive_alts = []
                        for balt in base_alts:
                            recursive_alts.append(balt + [aux_name])
                        self.rules[aux_name].extend(recursive_alts)
                    
                    current_alt.append(aux_name)
                    i = j + 1
                elif t in [']', '}', ')']:
                    # Should be handled by finding matching closer, but if stray:
                    i += 1
                else:
                    current_alt.append(t)
                    i += 1
            
            if current_alt: alts.append(current_alt)
            elif tokens and tokens[-1] == '|': alts.append([])
            
            return alts

        new_alts = scan(0, tokens)
        if rule_name not in self.rules: self.rules[rule_name] = []
        self.rules[rule_name].extend(new_alts)

    def calculate_shortest_expansions(self):
        # Calculate shortest path to terminals for every non-terminal
        # to ensure we can terminate recursion
        for r in self.rules:
            self.shortest_expansions[r] = None
        
        changed = True
        while changed:
            changed = False
            for r, alts in self.rules.items():
                best_exp = self.shortest_expansions[r]
                best_len = len(best_exp) if best_exp is not None else float('inf')
                
                for alt in alts:
                    current_exp = []
                    possible = True
                    current_len = 0
                    for token in alt:
                        if token.startswith('<'): 
                            if token in self.rules:
                                s = self.shortest_expansions.get(token)
                                if s is None:
                                    possible = False; break
                                current_exp.extend(s)
                                current_len += len(s)
                            else:
                                # Unknown non-terminal, assume size 1 dummy
                                current_exp.append(self.dummy_terminal(token))
                                current_len += 1
                        else:
                            current_exp.append(self.clean_token(token))
                            current_len += 1
                    
                    if possible:
                        if current_len < best_len:
                            best_len = current_len
                            best_exp = current_exp
                            self.shortest_expansions[r] = best_exp
                            changed = True

    def generate(self, symbol, depth, max_depth):
        # 1. Terminal case
        if not symbol.startswith('<'):
            return [self.clean_token(symbol)]
            
        # 2. Depth limit / Fallback
        if depth > max_depth:
            if symbol in self.shortest_expansions and self.shortest_expansions[symbol] is not None:
                return self.shortest_expansions[symbol]
            return [self.dummy_terminal(symbol)]
            
        # 3. Expansion
        if symbol not in self.rules:
            return [self.dummy_terminal(symbol)]
            
        alts = self.rules[symbol]
        if not alts: return []

        # Coverage-guided selection: Pick least used production
        candidates = []
        for i, alt in enumerate(alts):
            count = self.production_counts[(symbol, i)]
            candidates.append((count, i, alt))
        
        candidates.sort(key=lambda x: x[0])
        min_count = candidates[0][0]
        best_candidates = [c for c in candidates if c[0] == min_count]
        
        choice = random.choice(best_candidates)
        self.production_counts[(symbol, choice[1])] += 1
        
        result = []
        for token in choice[2]:
            result.extend(self.generate(token, depth + 1, max_depth))
            
        return result

    def clean_token(self, t):
        if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
            return t[1:-1]
        return t

    def dummy_terminal(self, t):
        low = t.lower()
        if 'ident' in low or 'name' in low: return f"id_{random.randint(1,99)}"
        if 'table' in low: return f"t_{random.randint(1,99)}"
        if 'col' in low: return f"c_{random.randint(1,99)}"
        if 'num' in low or 'int' in low: return str(random.randint(0, 100))
        if 'str' in low: return "'val'"
        if 'op' in low: return "="
        return "x" 

    def post_process(self, tokens):
        s = " ".join(tokens)
        # Fix spacing around punctuation
        for p in ",();.":
            s = s.replace(f" {p}", p).replace(f"{p} ", f"{p} ")
        
        # Specific fixes
        s = s.replace("( ", "(").replace(" )", ")")
        s = s.replace(" .", ".")
        return s.strip()