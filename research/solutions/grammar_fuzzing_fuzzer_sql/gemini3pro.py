import os
import re
import random

class Solution:
    def solve(self, resources_path: str) -> dict:
        # Strategy:
        # 1. Harvest potential keywords from the grammar and source code to maximize tokenizer coverage.
        # 2. Generate a robust fuzzer script that uses:
        #    - A structured generator for valid SQL (to hit deep parser logic).
        #    - A random sequence generator (keyword salad) for error handling paths.
        #    - A set of hardcoded edge cases.
        # 3. Configure the fuzzer to execute all statements in a single batch call to parse_sql
        #    to maximize the efficiency bonus (N=1).

        keywords = set()
        # Default standard SQL keywords
        defaults = [
            "SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "UPDATE", "SET", 
            "DELETE", "CREATE", "TABLE", "DROP", "AND", "OR", "NOT", "NULL", "TRUE", 
            "FALSE", "INT", "VARCHAR", "FLOAT", "PRIMARY", "KEY", "ORDER", "BY", 
            "GROUP", "HAVING", "LIMIT", "ASC", "DESC", "JOIN", "LEFT", "RIGHT", "INNER", 
            "OUTER", "ON", "AS", "DISTINCT", "COUNT", "SUM", "AVG", "MAX", "MIN", 
            "BETWEEN", "LIKE", "IN", "IS", "EXISTS", "UNION", "ALL", "CASE", "WHEN", 
            "THEN", "ELSE", "END", "TEXT", "BOOLEAN", "DATETIME", "DATE", "CHAR"
        ]
        keywords.update(defaults)

        try:
            # Heuristic 1: Extract uppercase words from grammar file
            g_path = os.path.join(resources_path, "sql_grammar.txt")
            if os.path.exists(g_path):
                with open(g_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    matches = re.findall(r"\b[A-Z_]{2,}\b", content)
                    keywords.update(matches)
            
            # Heuristic 2: Extract string literals from parser/tokenizer source
            e_path = os.path.join(resources_path, "sql_engine")
            for fname in ["parser.py", "tokenizer.py", "ast_nodes.py"]:
                fpath = os.path.join(e_path, fname)
                if os.path.exists(fpath):
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        matches = re.findall(r"['\"]([a-zA-Z_]+)['\"]", content)
                        # Filter for likely keywords
                        for m in matches:
                            if m.upper() == m and len(m) > 1:
                                keywords.add(m)
                            elif m.upper() in defaults:
                                keywords.add(m.upper())
        except Exception:
            pass

        kw_list = sorted(list(keywords))

        code = r'''
import random
import string

KEYWORDS = ''' + str(kw_list) + r'''

class SQLGenerator:
    def __init__(self):
        self.tables = ["users", "products", "orders", "items", "logs", "t1", "t2", "data", "admin"]
        self.columns = ["id", "name", "price", "status", "created_at", "qty", "description", "col1", "col2", "active", "score", "email"]
        self.ops = ["=", "!=", "<", ">", "<=", ">=", "LIKE", "NOT LIKE", "IN", "IS", "IS NOT"]
        self.aggs = ["COUNT", "SUM", "AVG", "MIN", "MAX"]
        self.types = ["INT", "VARCHAR(255)", "TEXT", "FLOAT", "BOOLEAN", "DATE"]

    def get_id(self):
        return random.choice(self.columns)
        
    def get_table(self):
        return random.choice(self.tables)
        
    def get_val(self):
        r = random.random()
        if r < 0.2: return str(random.randint(-100, 1000))
        if r < 0.4: return f"'{''.join(random.choices(string.ascii_letters, k=5))}'"
        if r < 0.5: return "NULL"
        if r < 0.6: return "TRUE"
        if r < 0.7: return "FALSE"
        if r < 0.8: return str(round(random.random() * 100, 2))
        return self.get_id()

    def get_condition(self, depth=0):
        if depth > 2 or random.random() < 0.4:
            op = random.choice(self.ops)
            if "IN" in op:
                return f"{self.get_id()} {op} ({self.get_val()}, {self.get_val()})"
            if "IS" in op:
                return f"{self.get_id()} {op} NULL"
            return f"{self.get_id()} {op} {self.get_val()}"
        
        c1 = self.get_condition(depth + 1)
        c2 = self.get_condition(depth + 1)
        return f"({c1} {random.choice(['AND', 'OR'])} {c2})"

    def gen_select(self):
        cols = "*"
        if random.random() < 0.8:
            c_list = []
            for _ in range(random.randint(1, 4)):
                c = self.get_id()
                if random.random() < 0.2: c = f"{random.choice(self.aggs)}({c})"
                c_list.append(c)
            cols = ", ".join(c_list)
            
        stmt = f"SELECT {cols} FROM {self.get_table()}"
        
        if random.random() < 0.3:
            j = random.choice(["JOIN", "LEFT JOIN", "INNER JOIN"])
            t2 = self.get_table()
            stmt += f" {j} {t2} ON {self.get_table()}.id = {t2}.id"

        if random.random() < 0.6: stmt += f" WHERE {self.get_condition()}"
        if random.random() < 0.2: stmt += f" GROUP BY {self.get_id()}"
        if random.random() < 0.15: stmt += f" ORDER BY {self.get_id()} DESC"
        if random.random() < 0.1: stmt += f" LIMIT {random.randint(1, 100)}"
        return stmt

    def gen_dml(self):
        r = random.random()
        if r < 0.33:
            cols = random.sample(self.columns, random.randint(1, 3))
            vals = [self.get_val() for _ in cols]
            return f"INSERT INTO {self.get_table()} ({', '.join(cols)}) VALUES ({', '.join(vals)})"
        elif r < 0.66:
            assigns = ", ".join([f"{c}={self.get_val()}" for c in random.sample(self.columns, random.randint(1, 2))])
            return f"UPDATE {self.get_table()} SET {assigns} WHERE {self.get_condition()}"
        else:
            return f"DELETE FROM {self.get_table()} WHERE {self.get_condition()}"

    def gen_ddl(self):
        r = random.random()
        if r < 0.6:
            cols = []
            for _ in range(random.randint(1, 4)):
                c = f"{self.get_id()}_{random.randint(1,99)} {random.choice(self.types)}"
                if random.random() < 0.2: c += " PRIMARY KEY"
                cols.append(c)
            return f"CREATE TABLE {self.get_table()}_{random.randint(1,999)} ({', '.join(cols)})"
        return f"DROP TABLE {self.get_table()}"
    
    def gen_keyword_salad(self):
        length = random.randint(2, 8)
        tokens = []
        for _ in range(length):
            if random.random() < 0.7: tokens.append(random.choice(KEYWORDS))
            else: tokens.append(random.choice(self.columns + self.ops))
        return " ".join(tokens)

def fuzz(parse_sql):
    gen = SQLGenerator()
    stmts = []
    
    # 1. Structured valid-ish SQL (Deep parser coverage)
    # Generate ~1500 valid statements
    for _ in range(800): stmts.append(gen.gen_select())
    for _ in range(500): stmts.append(gen.gen_dml())
    for _ in range(200): stmts.append(gen.gen_ddl())
    
    # 2. Keyword salad (Tokenizer/Parser error handling coverage)
    # Generate ~200 invalid/random sequences
    for _ in range(200): stmts.append(gen.gen_keyword_salad())
    
    # 3. Hardcoded edge cases
    stmts.extend([
        "", ";", "SELECT", "SELECT *", "SELECT FROM", 
        "INSERT INTO", "VALUES", "DROP", "CREATE",
        "SELECT * FROM t WHERE", "SELECT 1.2.3.4",
        "SELECT 'unclosed", 'SELECT "unclosed',
        "SELECT * FROM t -- comment", 
        "SELECT * FROM t /* comment */ WHERE 1=1",
        "SELECT * FROM t; DROP TABLE t",
        "SELECT * FROM t WHERE a = b AND",
        "SELECT count(*) FROM (SELECT 1) AS sub",
        "SELECT CASE WHEN 1=1 THEN 1 ELSE 0 END",
        "SELECT * FROM t WHERE col1 IS NULL"
    ])
    
    # Execute all in one massive batch to maximize efficiency bonus
    # Efficiency bonus = 30 * 2^(-N/500), where N=1 gives max score
    parse_sql(stmts)
    
    # Return False to indicate we are done
    return False
'''
        return {"code": code}