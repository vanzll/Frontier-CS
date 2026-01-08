import os
import re

class Solution:
    def solve(self, resources_path: str) -> dict:
        # Step 1: Attempt to extract specific keywords from the provided grammar file
        # to ensure the tokenizer and parser branches related to specific tokens are covered.
        keywords = set()
        # Default SQL keywords to ensure baseline coverage if extraction fails
        defaults = [
            "SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "UPDATE", "SET",
            "DELETE", "CREATE", "TABLE", "DROP", "ALTER", "ADD", "COLUMN", "KEY",
            "PRIMARY", "FOREIGN", "REFERENCES", "JOIN", "INNER", "LEFT", "RIGHT",
            "OUTER", "ON", "GROUP", "BY", "HAVING", "ORDER", "ASC", "DESC",
            "LIMIT", "OFFSET", "UNION", "ALL", "DISTINCT", "AND", "OR", "NOT",
            "NULL", "TRUE", "FALSE", "IS", "IN", "BETWEEN", "LIKE", "AS",
            "CASE", "WHEN", "THEN", "ELSE", "END", "CAST", "CONSTRAINT",
            "INT", "INTEGER", "VARCHAR", "CHAR", "TEXT", "FLOAT", "DOUBLE", "BOOLEAN",
            "COUNT", "SUM", "AVG", "MIN", "MAX"
        ]
        keywords.update(defaults)

        try:
            grammar_path = os.path.join(resources_path, 'sql_grammar.txt')
            if os.path.exists(grammar_path):
                with open(grammar_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Extract quoted tokens e.g. 'SELECT' or "SELECT"
                    quoted = re.findall(r"'([A-Z_]+)'", content)
                    keywords.update(quoted)
                    quoted_double = re.findall(r'"([A-Z_]+)"', content)
                    keywords.update(quoted_double)
                    # Extract uppercase words that look like keywords (len >= 2)
                    caps = re.findall(r'\b[A-Z_]{2,}\b', content)
                    keywords.update(caps)
        except Exception:
            pass

        keywords_list = sorted(list(keywords))

        # Step 2: Construct the fuzzer code.
        # We use string concatenation to inject the keywords_list safely.
        # The strategy is to generate a massive batch of diverse SQL statements and 
        # execute them in a single call to parse_sql to maximize the efficiency bonus (N=1).
        
        fuzzer_code = """
import random
import string
import sys

# Keywords extracted from the grammar file by the solution generator
KEYWORDS = """ + str(keywords_list) + """

class SQLFuzzerGen:
    def __init__(self):
        self.tables = ['users', 'products', 'orders', 't1', 't2', 'items', 'log']
        self.cols = ['id', 'name', 'price', 'active', 'created_at', 'score', 'data', 'val']
        self.ops = ['=', '!=', '<', '>', '<=', '>=', 'LIKE', 'IN']
        self.logics = ['AND', 'OR']
        
    def rnd_id(self):
        return random.choice(self.cols)
        
    def rnd_table(self):
        return random.choice(self.tables)
        
    def rnd_val(self):
        r = random.random()
        if r < 0.25: return str(random.randint(-100, 1000))
        if r < 0.5: return f"'{self.rnd_str()}'"
        if r < 0.6: return str(random.uniform(-1000.0, 1000.0))
        if r < 0.7: return 'NULL'
        if r < 0.8: return 'TRUE' if random.random() > 0.5 else 'FALSE'
        return str(random.randint(0, 10))

    def rnd_str(self):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 10)))

    def gen_expr(self, depth=0):
        # Limit recursion depth
        if depth > 2 or random.random() < 0.2:
            # Base predicate
            if random.random() < 0.5:
                return f"{self.rnd_id()} {random.choice(self.ops)} {self.rnd_val()}"
            else:
                return f"{self.rnd_id()} = {self.rnd_id()}"
        
        r = random.random()
        if r < 0.4:
            # Binary Logic
            return f"({self.gen_expr(depth+1)} {random.choice(self.logics)} {self.gen_expr(depth+1)})"
        elif r < 0.6:
            # Not
            return f"NOT {self.gen_expr(depth+1)}"
        elif r < 0.7:
            # IS NULL
            return f"{self.rnd_id()} IS {'NOT ' if random.random()<0.5 else ''}NULL"
        elif r < 0.8:
            # BETWEEN
            return f"{self.rnd_id()} BETWEEN {random.randint(0,5)} AND {random.randint(6,10)}"
        else:
            # Parenthesis
            return f"({self.gen_expr(depth+1)})"

    def gen_select(self):
        cols = "*"
        if random.random() < 0.6:
            cols = ", ".join(random.choices(self.cols, k=random.randint(1, 4)))
            if random.random() < 0.2: cols = "DISTINCT " + cols
            
        stmt = f"SELECT {cols} FROM {self.rnd_table()}"
        
        # Joins
        if random.random() < 0.3:
            jtype = random.choice(["JOIN", "LEFT JOIN", "INNER JOIN"])
            stmt += f" {jtype} {self.rnd_table()} ON {self.rnd_id()} = {self.rnd_id()}"
            
        # Where
        if random.random() < 0.6:
            stmt += f" WHERE {self.gen_expr()}"
            
        # Group By
        if random.random() < 0.2:
            stmt += f" GROUP BY {self.rnd_id()}"
            if random.random() < 0.5: stmt += f" HAVING {self.gen_expr()}"
            
        # Order By
        if random.random() < 0.2:
            stmt += f" ORDER BY {self.rnd_id()} {random.choice(['ASC', 'DESC'])}"
            
        # Limit
        if random.random() < 0.2:
            stmt += f" LIMIT {random.randint(1, 50)}"
            
        # Union/Compound (simple recursion)
        if random.random() < 0.1:
            stmt += f" UNION SELECT {cols} FROM {self.rnd_table()}"
            
        return stmt

    def gen_dml(self):
        r = random.random()
        if r < 0.33:
            # Insert
            vals = ", ".join([self.rnd_val() for _ in range(random.randint(1, 4))])
            if random.random() < 0.5:
                return f"INSERT INTO {self.rnd_table()} VALUES ({vals})"
            else:
                cols = ", ".join(random.choices(self.cols, k=random.randint(1, 4)))
                return f"INSERT INTO {self.rnd_table()} ({cols}) VALUES ({vals})"
        elif r < 0.66:
            # Update
            assigns = ", ".join([f"{c}={self.rnd_val()}" for c in random.choices(self.cols, k=random.randint(1,2))])
            return f"UPDATE {self.rnd_table()} SET {assigns} WHERE {self.gen_expr()}"
        else:
            # Delete
            return f"DELETE FROM {self.rnd_table()} WHERE {self.gen_expr()}"

    def gen_ddl(self):
        r = random.random()
        if r < 0.5:
            # Create
            cdefs = []
            types = ['INT', 'VARCHAR(255)', 'FLOAT', 'BOOLEAN']
            for c in random.choices(self.cols, k=random.randint(1, 4)):
                cdefs.append(f"{c} {random.choice(types)}")
            return f"CREATE TABLE {self.rnd_table()} ({', '.join(cdefs)})"
        else:
            return f"DROP TABLE {self.rnd_table()}"

    def generate(self):
        r = random.random()
        if r < 0.5: return self.gen_select()
        if r < 0.8: return self.gen_dml()
        return self.gen_ddl()

def fuzz(parse_sql):
    gen = SQLFuzzerGen()
    statements = []
    
    # 1. Structural Generation: Create valid complex queries to traverse parser depth
    for _ in range(2500):
        statements.append(gen.generate())
        
    # 2. Keyword Coverage: Ensure every known keyword is tokenized at least once
    # This hits switch cases/lookup tables in the tokenizer
    for kw in KEYWORDS:
        statements.append(kw)
        statements.append(f"SELECT {kw}")
        statements.append(f"SELECT * FROM {kw}")
        
    # 3. Static High-Value Corpus: Patterns that are hard to generate randomly
    manual = [
        "SELECT * FROM t",
        "SELECT 1",
        "SELECT 1+2, 3*4, 5/6, 7-8, 9%2",
        "SELECT count(*), sum(x) FROM t GROUP BY y HAVING sum(x) > 10",
        "SELECT * FROM (SELECT id FROM t) AS sub",
        "SELECT CAST(x AS INT) FROM t",
        "SELECT CASE WHEN x=1 THEN 'a' ELSE 'b' END FROM t",
        "SELECT * FROM t WHERE col IN (1, 2, 3)",
        "SELECT * FROM t WHERE col LIKE 'abc%'",
        "INSERT INTO t DEFAULT VALUES",
        "CREATE INDEX idx ON t(col)",
        "ALTER TABLE t ADD COLUMN c INT",
        "-- comment line",
        "/* block comment */",
        "SELECT 'unterminated string",
        "SELECT 123456789012345678901234567890", # Overflows
        "SELECT 1.2.3.4", # Bad float
        "SELECT @#$%", # Bad chars
        "", # Empty
        ";", # Semi
    ]
    statements.extend(manual)
    
    # 4. Mutation Fuzzing: Take valid queries and break them to hit error paths
    for _ in range(500):
        s = gen.gen_select()
        if not s: continue
        
        m_type = random.randint(0, 3)
        if m_type == 0: # Insert garbage
            pos = random.randint(0, len(s))
            char = random.choice(string.punctuation)
            s = s[:pos] + char + s[pos:]
        elif m_type == 1: # Delete character
            if len(s) > 1:
                pos = random.randint(0, len(s)-1)
                s = s[:pos] + s[pos+1:]
        elif m_type == 2: # Replace with keyword
            parts = s.split(' ')
            if len(parts) > 1:
                parts[random.randint(0, len(parts)-1)] = random.choice(KEYWORDS)
                s = " ".join(parts)
        
        statements.append(s)

    # 5. EXECUTION STRATEGY
    # We send ALL generated statements in a SINGLE batch call.
    # The efficiency bonus is 30 * 2^(-N/500).
    # With N=1, we get ~29.96 points bonus (max possible).
    # 60s is sufficient to parse ~4000 statements.
    
    parse_sql(statements)
    
    # Return False to stop fuzzing immediately after the first batch.
    return False
"""
        return {"code": fuzzer_code}