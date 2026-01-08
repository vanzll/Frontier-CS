import random
import os

class Solution:
    def solve(self, resources_path: str) -> dict:
        # Generate a self-contained fuzzer with a comprehensive hardcoded SQL grammar.
        # This ensures maximizing coverage by producing valid, deeply nested, and edge-case SQL statements.
        # We execute all statements in a single batch (N=1) to maximize the efficiency bonus.
        
        fuzzer_code = r'''import random
import sys

# Increase recursion limit for deep AST generation
sys.setrecursionlimit(5000)

class GrammarFuzzer:
    def __init__(self):
        # Comprehensive SQL Grammar covering common recursive descent parser paths
        self.grammar = {
            # Root
            "<start>": [["<stmt>"]],
            "<stmt>": [
                ["<select_stmt>"], ["<insert_stmt>"], ["<update_stmt>"], 
                ["<delete_stmt>"], ["<create_stmt>"], ["<drop_stmt>"],
                ["<transaction_stmt>"]
            ],
            
            # SELECT
            "<select_stmt>": [
                ["SELECT", "<opt_distinct>", "<select_list>", "FROM", "<table_ref>", "<opt_join>", "<opt_where>", "<opt_group>", "<opt_order>", "<opt_limit>"]
            ],
            "<opt_distinct>": [[], ["DISTINCT"], ["ALL"]],
            "<select_list>": [["*"], ["<expr_list>"]],
            "<expr_list>": [["<expr>"], ["<expr>", ",", "<expr_list>"]],
            "<table_ref>": [
                ["<ident>"], 
                ["<ident>", "AS", "<ident>"], 
                ["<ident>", "<ident>"],
                ["(", "<select_stmt>", ")", "AS", "<ident>"]
            ],
            "<opt_join>": [[], ["<join_type>", "JOIN", "<table_ref>", "ON", "<expr>", "<opt_join>"]],
            "<join_type>": [["INNER"], ["LEFT"], ["RIGHT"], ["FULL"], ["CROSS"], ["LEFT", "OUTER"], ["RIGHT", "OUTER"]],
            "<opt_where>": [[], ["WHERE", "<expr>"]],
            "<opt_group>": [[], ["GROUP BY", "<expr_list>", "<opt_having>"]],
            "<opt_having>": [[], ["HAVING", "<expr>"]],
            "<opt_order>": [[], ["ORDER BY", "<expr_list>", "<opt_dir>"]],
            "<opt_dir>": [[], ["ASC"], ["DESC"]],
            "<opt_limit>": [[], ["LIMIT", "<int_literal>"], ["LIMIT", "<int_literal>", "OFFSET", "<int_literal>"]],

            # INSERT
            "<insert_stmt>": [["INSERT", "INTO", "<ident>", "<opt_cols>", "VALUES", "<values_list>"]],
            "<opt_cols>": [[], ["(", "<ident_list>", ")"]],
            "<ident_list>": [["<ident>"], ["<ident>", ",", "<ident_list>"]],
            "<values_list>": [["(", "<expr_list>", ")"], ["(", "<expr_list>", ")", ",", "<values_list>"]],

            # UPDATE
            "<update_stmt>": [["UPDATE", "<ident>", "SET", "<assign_list>", "<opt_where>"]],
            "<assign_list>": [["<ident>", "=", "<expr>"], ["<ident>", "=", "<expr>", ",", "<assign_list>"]],

            # DELETE
            "<delete_stmt>": [["DELETE", "FROM", "<ident>", "<opt_where>"]],

            # CREATE
            "<create_stmt>": [["CREATE", "TABLE", "<ident>", "(", "<col_def_list>", ")"]],
            "<col_def_list>": [["<ident>", "<type>", "<opt_constraint>"], ["<ident>", "<type>", "<opt_constraint>", ",", "<col_def_list>"]],
            "<type>": [["INT"], ["INTEGER"], ["VARCHAR", "(", "<int_literal>", ")"], ["TEXT"], ["FLOAT"], ["DOUBLE"], ["BOOLEAN"], ["DATE"], ["TIMESTAMP"]],
            "<opt_constraint>": [[], ["PRIMARY KEY"], ["NOT NULL"], ["UNIQUE"], ["DEFAULT", "<literal>"], ["CHECK", "(", "<expr>", ")"]],

            # DROP
            "<drop_stmt>": [["DROP", "TABLE", "<ident>"], ["DROP", "TABLE", "IF", "EXISTS", "<ident>"]],
            
            # TRANSACTION
            "<transaction_stmt>": [["BEGIN"], ["COMMIT"], ["ROLLBACK"]],

            # EXPRESSIONS
            "<expr>": [
                ["<term>"], 
                ["<term>", "<bin_op>", "<expr>"],
                ["<term>", "IS", "NULL"],
                ["<term>", "IS", "NOT", "NULL"],
                ["<term>", "BETWEEN", "<term>", "AND", "<term>"],
                ["<term>", "IN", "(", "<expr_list>", ")"],
                ["<term>", "IN", "(", "<select_stmt>", ")"],
                ["EXISTS", "(", "<select_stmt>", ")"]
            ],
            "<term>": [
                ["<factor>"], 
                ["NOT", "<factor>"], 
                ["-", "<factor>"],
                ["+", "<factor>"],
                ["~", "<factor>"]
            ],
            "<factor>": [
                ["<ident>"], 
                ["<literal>"], 
                ["(", "<expr>", ")"], 
                ["<agg_func>", "(", "<expr>", ")"],
                ["<agg_func>", "(", "DISTINCT", "<expr>", ")"],
                ["<func>", "(", "<expr_list>", ")"],
                ["CASE", "WHEN", "<expr>", "THEN", "<expr>", "ELSE", "<expr>", "END"],
                ["CAST", "(", "<expr>", "AS", "<type>", ")"]
            ],
            
            # OPERATORS & FUNCS
            "<bin_op>": [
                ["+"], ["-"], ["*"], ["/"], ["%"], ["||"], ["&"], ["|"], ["^"],
                ["="], ["!="], ["<>"], ["<"], [">"], ["<="], [">="], ["<=>"],
                ["AND"], ["OR"], ["LIKE"], ["NOT", "LIKE"]
            ],
            "<agg_func>": [["COUNT"], ["SUM"], ["AVG"], ["MIN"], ["MAX"]],
            "<func>": [["ABS"], ["LENGTH"], ["ROUND"], ["UPPER"], ["LOWER"], ["COALESCE"], ["NULLIF"], ["SUBSTR"]],

            # TERMINALS
            "<ident>": ["users", "products", "orders", "customers", "id", "name", "price", "qty", "created_at", "status", "t1", "t2", "idx", "val", "data", "metadata"],
            "<int_literal>": ["0", "1", "10", "42", "100", "-1", "99999"],
            "<literal>": ["0", "1", "3.14", "1.5e10", "'test'", "'foo bar'", "NULL", "TRUE", "FALSE", "'2023-01-01'", "''", "'O''Connor'", "'\n'"]
        }

    def generate(self, symbol, depth=0):
        # Safety fallback for recursion depth
        if depth > 15:
            # Return simple terminals to unwind recursion
            if "opt" in symbol: return ""
            if symbol == "<expr>" or symbol == "<term>" or symbol == "<factor>": return "1"
            if symbol == "<ident>": return "id"
            if symbol == "<literal>" or symbol == "<int_literal>": return "1"
            if symbol == "<type>": return "INT"
            if symbol == "<stmt>": return "SELECT 1"
            if symbol.endswith("_list"): return "" # Stop list expansion
            # Try to pick a terminal if available
            if symbol in self.grammar:
                terminals = [p for p in self.grammar[symbol] if isinstance(p[0], str)]
                if terminals: return random.choice(terminals)
                # Pick shortest rule
                shortest = min(self.grammar[symbol], key=len)
                if isinstance(shortest[0], str): return random.choice(self.grammar[symbol])
            return ""

        if symbol not in self.grammar:
            return symbol

        productions = self.grammar[symbol]
        
        # If productions is a list of strings (terminals), pick one
        if isinstance(productions[0], str):
            return random.choice(productions)
        
        # Productions is a list of rules (lists)
        # Weighting: Prefer deeper recursion early, terminals later
        if depth > 10:
            # Pick shorter rules to converge
            rule = min(productions, key=len)
        else:
            rule = random.choice(productions)
            
        return " ".join([self.generate(s, depth + 1) for s in rule]).strip()

def fuzz(parse_sql):
    gen = GrammarFuzzer()
    statements = []
    
    # 1. Generate diverse valid SQL (Massive batch)
    # 2000 statements usually executes well within 60s for standard parsers
    for _ in range(2500):
        try:
            stmt = gen.generate("<start>")
            # Cleanup artifact spaces
            while "  " in stmt: stmt = stmt.replace("  ", " ")
            if stmt:
                statements.append(stmt)
        except:
            pass
            
    # 2. Add manual edge cases to hit tokenizer/error paths
    edge_cases = [
        "", ";", "   ",
        # Keywords as identifiers
        "SELECT select FROM table",
        "CREATE TABLE table (int int)",
        # Unterminated strings
        "SELECT 'unterminated",
        "SELECT 'escaped '' quote",
        "SELECT \"double quote\"",
        # Number formats
        "SELECT 1.", "SELECT .5", "SELECT 1.2.3", "SELECT 1e5", "SELECT 1e-5", "SELECT 0xFF",
        # Comments
        "SELECT 1 -- comment",
        "SELECT 1 /* block */",
        "SELECT 1 /* unfinished",
        # Operators
        "SELECT 1+1", "SELECT 1- -1", "SELECT 1/*..*/+1",
        # Invalid Syntax
        "SELECT", "SELECT FROM", "INSERT INTO", "DROP",
        # Massive overflow
        "SELECT " + "9"*100,
        # Deep nesting
        "SELECT " + "("*50 + "1" + ")"*50,
        # Long identifiers
        "SELECT * FROM " + "a"*200,
    ]
    statements.extend(edge_cases)
    
    # 3. Random noise for tokenizer robustness
    chars = "abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{};':\",./<>?`~"
    for _ in range(200):
        l = random.randint(1, 100)
        statements.append("".join(random.choice(chars) for _ in range(l)))

    # Execute all in one call to minimize N (N=1 gives max efficiency bonus)
    parse_sql(statements)
    
    # Return False to indicate we are done (exhausted budget or strategy)
    return False
'''
        return {"code": fuzzer_code}