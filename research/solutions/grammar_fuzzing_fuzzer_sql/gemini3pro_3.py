import os
import random
from typing import Dict

class Solution:
    def solve(self, resources_path: str) -> Dict[str, str]:
        fuzzer_code = r'''
import random
import string
import os
import re

# Global flag to ensure single execution
# We generate a massive batch and run it once to maximize the efficiency bonus
# which is calculated based on minimizing the number of parse_sql calls.
EXECUTED = False

class SQLFuzzer:
    def __init__(self):
        # Comprehensive list of standard SQL keywords
        self.keywords = [
            "SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "UPDATE", "SET",
            "DELETE", "CREATE", "TABLE", "DROP", "JOIN", "ON", "AS", "ORDER", "BY",
            "GROUP", "HAVING", "LIMIT", "OFFSET", "UNION", "ALL", "DISTINCT",
            "NULL", "NOT", "AND", "OR", "LIKE", "BETWEEN", "IN", "IS", "CASE",
            "WHEN", "THEN", "ELSE", "END", "PRIMARY", "KEY", "FOREIGN", "REFERENCES",
            "INT", "VARCHAR", "CHAR", "TEXT", "FLOAT", "DOUBLE", "BOOLEAN", "DATE",
            "ASC", "DESC", "OUTER", "INNER", "LEFT", "RIGHT", "CROSS", "NATURAL",
            "DEFAULT", "CONSTRAINT", "INDEX", "VIEW", "TRIGGER", "PROCEDURE",
            "CAST", "EXISTS", "ANY", "ALL", "SOME"
        ]
        self.operators = ["=", "<>", "!=", "<", ">", "<=", ">=", "LIKE", "AND", "OR", "+", "-", "*", "/", "%", "||"]
        self.types = ["INT", "VARCHAR(255)", "TEXT", "BOOLEAN", "FLOAT", "DATE", "DECIMAL(10,2)"]
        self.tables = ["users", "products", "orders", "log", "t1", "t2", "data_table"]
        self.columns = ["id", "name", "price", "active", "created_at", "val", "category", "description", "score"]
        
        self.load_grammar_keywords()

    def load_grammar_keywords(self):
        """Attempt to extract keywords from the provided grammar file to catch dialect-specific syntax."""
        try:
            candidates = [
                "resources/sql_grammar.txt",
                "../resources/sql_grammar.txt",
                os.path.join(os.path.dirname(__file__), "resources/sql_grammar.txt"),
                os.path.join(os.path.dirname(__file__), "../resources/sql_grammar.txt")
            ]
            content = ""
            for p in candidates:
                if os.path.exists(p):
                    with open(p, "r") as f:
                        content = f.read()
                    break
            
            if content:
                # Heuristic: Uppercase words length >= 3
                found = re.findall(r"\b[A-Z_]{3,}\b", content)
                self.keywords.extend(found)
                # Heuristic: Quoted literals in BNF
                found_lit = re.findall(r"'([A-Z]+)'", content)
                self.keywords.extend(found_lit)
                self.keywords = list(set(self.keywords))
        except:
            pass

    def random_token(self):
        return "".join(random.choices(string.ascii_letters, k=random.randint(2, 10)))

    def random_literal(self):
        r = random.random()
        if r < 0.2: return str(random.randint(-1000, 1000))
        if r < 0.4: return f"'{self.random_token()}'"
        if r < 0.5: return str(random.uniform(-1000.0, 1000.0))
        if r < 0.6: return "NULL"
        if r < 0.7: return "TRUE"
        if r < 0.8: return "FALSE"
        if r < 0.9: return "?" # Placeholder
        return "1.2e-5" # Scientific notation

    def expression(self, depth=0):
        # Recursive expression generator with depth limit
        if depth > 3 or random.random() < 0.3:
            r = random.random()
            if r < 0.5: return random.choice(self.columns)
            return self.random_literal()
        
        r = random.random()
        if r < 0.1:
            return f"({self.expression(depth+1)})"
        elif r < 0.4:
            # Binary op
            return f"{self.expression(depth+1)} {random.choice(self.operators)} {self.expression(depth+1)}"
        elif r < 0.5:
            # Unary op
            return f"NOT {self.expression(depth+1)}"
        elif r < 0.6:
            # Function call
            func = random.choice(["COUNT", "SUM", "AVG", "MIN", "MAX", "ABS", "LENGTH", "UPPER", "LOWER"])
            return f"{func}({self.expression(depth+1)})"
        elif r < 0.7:
            # IN clause
            vals = ", ".join([self.expression(depth+2) for _ in range(random.randint(1, 4))])
            return f"{random.choice(self.columns)} IN ({vals})"
        elif r < 0.8:
            # BETWEEN
            return f"{random.choice(self.columns)} BETWEEN {self.expression(depth+1)} AND {self.expression(depth+1)}"
        elif r < 0.9:
            # IS NULL
            return f"{self.expression(depth+1)} IS {random.choice(['NULL', 'NOT NULL'])}"
        else:
            # CASE
            return f"CASE WHEN {self.expression(depth+1)} THEN {self.expression(depth+1)} ELSE {self.expression(depth+1)} END"

    def gen_select(self):
        cols = "*" if random.random() < 0.2 else ", ".join([self.expression() for _ in range(random.randint(1, 4))])
        
        # Table generation (simple or join or subquery)
        if random.random() < 0.1:
            tbl = f"({self.gen_select_simple()}) AS sub_{self.random_token()}"
        else:
            tbl = random.choice(self.tables)
            if random.random() < 0.3:
                join_type = random.choice(["JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN"])
                tbl += f" {join_type} {random.choice(self.tables)} ON {self.expression()}"
            
        stmt = f"SELECT {cols} FROM {tbl}"
        
        if random.random() < 0.6:
            stmt += f" WHERE {self.expression()}"
            
        if random.random() < 0.2:
            stmt += f" GROUP BY {random.choice(self.columns)}"
            if random.random() < 0.5:
                stmt += f" HAVING {self.expression()}"
                
        if random.random() < 0.2:
            stmt += f" ORDER BY {random.choice(self.columns)} {random.choice(['ASC', 'DESC'])}"
            
        if random.random() < 0.1:
            stmt += f" LIMIT {random.randint(1, 50)}"
            
        return stmt

    def gen_select_simple(self):
        # Helper to avoid infinite recursion in subqueries
        return f"SELECT {random.choice(self.columns)} FROM {random.choice(self.tables)}"

    def gen_insert(self):
        tbl = random.choice(self.tables)
        if random.random() < 0.2:
            return f"INSERT INTO {tbl} DEFAULT VALUES"
            
        col_list = random.sample(self.columns, k=random.randint(1, 3))
        vals_list = []
        for _ in range(random.randint(1, 3)): # Multiple rows
            vals_list.append(f"({', '.join([self.expression(depth=3) for _ in range(len(col_list))])})")
            
        return f"INSERT INTO {tbl} ({', '.join(col_list)}) VALUES {', '.join(vals_list)}"

    def gen_update(self):
        tbl = random.choice(self.tables)
        assigns = ", ".join([f"{c} = {self.expression()}" for c in random.sample(self.columns, k=random.randint(1, 3))])
        return f"UPDATE {tbl} SET {assigns} WHERE {self.expression()}"

    def gen_delete(self):
        return f"DELETE FROM {random.choice(self.tables)} WHERE {self.expression()}"
    
    def gen_create(self):
        tbl = self.random_token()
        col_defs = []
        for _ in range(random.randint(1, 5)):
            cname = self.random_token()
            ctype = random.choice(self.types)
            if random.random() < 0.3: ctype += " NOT NULL"
            if random.random() < 0.1: ctype += " PRIMARY KEY"
            col_defs.append(f"{cname} {ctype}")
        return f"CREATE TABLE {tbl} ({', '.join(col_defs)})"

    def gen_misc(self):
        # Edge cases
        return random.choice([
            f"DROP TABLE {random.choice(self.tables)}",
            f"DROP TABLE IF EXISTS {random.choice(self.tables)}",
            "SELECT 1",
            "BEGIN TRANSACTION",
            "COMMIT",
            "ROLLBACK"
        ])

    def generate_batch(self, count=1000):
        stmts = []
        generators = [self.gen_select, self.gen_insert, self.gen_update, self.gen_delete, self.gen_create, self.gen_misc]
        # Weighted to favor complex SELECTs
        weights = [0.45, 0.15, 0.15, 0.1, 0.1, 0.05]
        
        for _ in range(count):
            try:
                gen = random.choices(generators, weights=weights)[0]
                stmts.append(gen())
            except RecursionError:
                stmts.append("SELECT 1")
        return stmts

    def get_manual_cases(self):
        # A curated list of queries to ensure basic block coverage of common SQL constructs
        return [
            # Basics
            "", ";", "   ",
            "SELECT", "SELECT *", "SELECT * FROM", 
            "SELECT * FROM t",
            "SELECT a AS alias FROM t",
            "SELECT DISTINCT a FROM t",
            "SELECT * FROM t WHERE a=1",
            # Expressions
            "SELECT 1+1, 2-2, 3*3, 4/4, 5%2 FROM t",
            "SELECT a FROM t WHERE a > 1 AND (b < 2 OR c = 3)",
            "SELECT NOT TRUE, NOT FALSE",
            "SELECT a FROM t WHERE a IS NULL",
            "SELECT a FROM t WHERE a IS NOT NULL",
            "SELECT a FROM t WHERE a IN (1, 2, 3)",
            "SELECT a FROM t WHERE a BETWEEN 10 AND 20",
            "SELECT a FROM t WHERE a LIKE 'abc%'",
            # Joins
            "SELECT * FROM t1 JOIN t2 ON t1.id = t2.id",
            "SELECT * FROM t1 LEFT JOIN t2 ON t1.id = t2.id",
            "SELECT * FROM t1 RIGHT JOIN t2 ON t1.id = t2.id",
            "SELECT * FROM t1 CROSS JOIN t2",
            # Aggregates
            "SELECT count(*), sum(a), avg(b), min(c), max(d) FROM t",
            "SELECT a, count(*) FROM t GROUP BY a",
            "SELECT a, count(*) FROM t GROUP BY a HAVING count(*) > 5",
            # Ordering/Limits
            "SELECT * FROM t ORDER BY a ASC, b DESC",
            "SELECT * FROM t LIMIT 10 OFFSET 5",
            # DML/DDL
            "INSERT INTO t VALUES (1, 'a')",
            "INSERT INTO t (a, b) VALUES (1, 'a'), (2, 'b')",
            "UPDATE t SET a=10 WHERE b=5",
            "DELETE FROM t",
            "CREATE TABLE t (id INT PRIMARY KEY, val TEXT)",
            "DROP TABLE t",
            # Comments and Strings
            "SELECT 'string'", 'SELECT "string"',
            "SELECT 'str''ing'", # escaped quote
            "-- comment line",
            "SELECT 1; -- comment at end",
            "/* block comment */ SELECT 1",
            # Subqueries/Common Table Expressions
            "SELECT * FROM (SELECT a FROM t) AS sub",
            "WITH cte AS (SELECT 1) SELECT * FROM cte",
            # Numbers
            "SELECT 123, 123.45, .45, 123., 1e5, 1.2E-3, -10"
        ]

def fuzz(parse_sql):
    global EXECUTED
    if EXECUTED:
        return False
        
    fuzzer = SQLFuzzer()
    
    # List of all statements to execute
    stmts = []
    
    # 1. Manual Golden Corpus
    stmts.extend(fuzzer.get_manual_cases())
    
    # 2. Generative Fuzzing (Bulk Coverage)
    # 2000 statements should be processed easily within 60s and provide good path combination coverage
    stmts.extend(fuzzer.generate_batch(2000))
    
    # 3. Keyword Soup (Stress Parser Logic)
    # Generate random sequences of keywords to hit unexpected branches in the recursive descent parser
    for _ in range(500):
        length = random.randint(1, 10)
        # Mix keywords, ops, and identifiers
        soup = random.choices(fuzzer.keywords + fuzzer.operators + fuzzer.columns, k=length)
        stmts.append(" ".join(soup))
        
    # 4. Mutation Fuzzing (Robustness)
    # Mutate a subset of valid statements to test error handling
    base_stmts = stmts[:100]
    for s in base_stmts:
        if not s: continue
        
        # Insertion mutation
        chars = list(s)
        for _ in range(3):
            pos = random.randint(0, len(chars))
            chars.insert(pos, random.choice(string.punctuation))
        stmts.append("".join(chars))
        
        # Deletion mutation
        if len(s) > 3:
            chars = list(s)
            del chars[random.randint(0, len(chars)-1)]
            stmts.append("".join(chars))
            
        # Token shuffle
        tokens = s.split()
        if len(tokens) > 2:
            random.shuffle(tokens)
            stmts.append(" ".join(tokens))
            
    # Execute everything in one batch
    # This maximizes the N_ref efficiency bonus (N=1)
    parse_sql(stmts)
    
    EXECUTED = True
    return False
'''
        return {"code": fuzzer_code}