import os
import random
import time
import math
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional, Any
import hashlib
import re

class Solution:
    def solve(self, resources_path: str) -> dict:
        fuzzer_code = self._generate_fuzzer_code(resources_path)
        return {"code": fuzzer_code}
    
    def _generate_fuzzer_code(self, resources_path: str) -> str:
        """Generate the complete fuzzer code as a string."""
        return f'''
import os
import sys
import random
import time
import math
import re
import hashlib
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional, Any
import string

# Add resources path to sys.path to import sql_engine
RESOURCES_PATH = r"{resources_path}"
sys.path.insert(0, os.path.join(RESOURCES_PATH, "sql_engine"))

# Try to import for grammar understanding (optional, for better generation)
try:
    from sql_engine import tokenizer
    from sql_engine import parser
    HAS_PARSER = True
except ImportError:
    HAS_PARSER = False

class SQLFuzzer:
    """Main fuzzing engine with multiple generation strategies."""
    
    def __init__(self):
        self.start_time = None
        self.time_budget = 60.0  # seconds
        self.stats = defaultdict(int)
        
        # State for coverage-guided fuzzing (simulated)
        self.corpus = []  # Store interesting inputs
        self.input_hashes = set()  # Deduplication
        self.edge_cases = self._get_edge_cases()
        
        # Grammar elements
        self.keywords = [
            "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER",
            "TABLE", "VIEW", "INDEX", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT",
            "INNER", "OUTER", "ON", "GROUP BY", "ORDER BY", "HAVING", "LIMIT",
            "OFFSET", "UNION", "ALL", "DISTINCT", "AS", "SET", "VALUES",
            "INTO", "NULL", "NOT", "AND", "OR", "IN", "BETWEEN", "LIKE",
            "IS", "EXISTS", "CASE", "WHEN", "THEN", "ELSE", "END"
        ]
        
        self.functions = [
            "COUNT", "SUM", "AVG", "MIN", "MAX", "COALESCE", "NULLIF",
            "UPPER", "LOWER", "SUBSTR", "TRIM", "ROUND", "ABS", "CAST",
            "EXTRACT", "CURRENT_DATE", "CURRENT_TIMESTAMP"
        ]
        
        self.data_types = [
            "INTEGER", "INT", "SMALLINT", "BIGINT", "DECIMAL", "NUMERIC",
            "REAL", "DOUBLE", "FLOAT", "BOOLEAN", "CHAR", "VARCHAR",
            "TEXT", "DATE", "TIME", "TIMESTAMP", "INTERVAL", "BLOB"
        ]
        
        # Common table/column names
        self.tables = ["users", "orders", "products", "customers", "employees",
                      "departments", "sales", "inventory", "accounts", "logs"]
        
        self.columns = ["id", "name", "age", "salary", "price", "quantity",
                       "date", "timestamp", "status", "type", "value", "description",
                       "email", "address", "city", "country", "phone", "created_at"]
        
        # Generation weights
        self.weights = {
            'valid': 0.6,
            'edge_case': 0.3,
            'mutated': 0.1
        }
        
        # Mutation operators
        self.mutators = [
            self._mutate_keyword,
            self._mutate_identifier,
            self._mutate_literal,
            self._mutate_add_clause,
            self._mutate_remove_clause,
            self._mutate_swap_clauses,
            self._mutate_nest_query,
            self._mutate_add_subquery
        ]
        
    def _get_edge_cases(self) -> List[str]:
        """Generate known edge cases that stress the parser."""
        return [
            # Empty/null cases
            "",
            ";",
            "()",
            "(())",
            
            # Very long identifiers
            "a" * 1000,
            f'"{"a" * 500}"',
            
            # Unicode and special characters
            "SELECT '\\x00'",
            "SELECT '日本語'",
            "SELECT '\\n\\t\\\\'",
            
            # Extreme numbers
            "SELECT 999999999999999999999999999999",
            "SELECT 0.0000000000000000000000000001",
            "SELECT -0.0",
            "SELECT 1e308",
            "SELECT -1e308",
            
            # Nested expressions
            "SELECT ((((1))))",
            "SELECT 1 + 2 * 3 / 4 - 5",
            
            # Multiple statements
            "SELECT 1; SELECT 2; SELECT 3;",
            "INSERT INTO t VALUES (1); DELETE FROM t;",
            
            # Edge whitespace
            "\\nSELECT\\n*\\nFROM\\nt\\n",
            "SELECT\\t*\\tFROM\\tt",
            
            # Strange but valid SQL
            "SELECT * FROM (SELECT 1) AS t",
            "SELECT * FROM t WHERE 1=1 AND 2=2 OR 3=3",
            "SELECT CASE WHEN 1 THEN 2 ELSE 3 END",
            "SELECT ALL DISTINCT * FROM t",  # Might be invalid but tests parser
            
            # Complex joins
            "SELECT * FROM a LEFT JOIN b ON a.id = b.id RIGHT JOIN c ON b.id = c.id",
            
            # Subqueries in various places
            "SELECT * FROM (SELECT * FROM t) AS sub WHERE (SELECT COUNT(*) FROM t2) > 0",
            "SELECT * FROM t WHERE id IN (SELECT id FROM t2)",
            "SELECT (SELECT 1), (SELECT 2) FROM t",
            
            # Window functions (if supported)
            "SELECT ROW_NUMBER() OVER (ORDER BY id) FROM t",
            
            # CTEs (if supported)
            "WITH cte AS (SELECT 1 AS x) SELECT * FROM cte",
            
            # Constraints and DDL edge cases
            "CREATE TABLE t (id INT PRIMARY KEY CHECK (id > 0) UNIQUE NOT NULL DEFAULT 1)",
            "ALTER TABLE t ADD COLUMN c INT, DROP COLUMN d, RENAME TO t2",
            
            # Type casting
            "SELECT CAST(1 AS VARCHAR(255)), CAST('1' AS INT)",
            
            # Functions with various arguments
            "SELECT COUNT(*), COUNT(DISTINCT id), COUNT(1, 2, 3)",
            "SELECT SUBSTR('hello', 1, 2), COALESCE(NULL, NULL, 'default')",
            
            # Transactions
            "BEGIN TRANSACTION; COMMIT;",
            "START TRANSACTION; ROLLBACK;",
            
            # Index operations
            "CREATE INDEX idx ON t (col1, col2, col3)",
            "DROP INDEX IF EXISTS idx",
            
            # Views
            "CREATE VIEW v AS SELECT * FROM t WITH CHECK OPTION",
            
            # Aliases
            "SELECT 1 AS a, 2 b, 3 c FROM t t1, t t2",
            
            # Group by variations
            "SELECT a, b, COUNT(*) FROM t GROUP BY a, b, ROLLUP(a, b)",
            "SELECT a, b FROM t GROUP BY GROUPING SETS ((a), (b), ())",
            
            # Having without group by
            "SELECT * FROM t HAVING 1=1",
            
            # Order by with expressions
            "SELECT * FROM t ORDER BY 1, 2 DESC, a + b",
            
            # Limit/offset edge cases
            "SELECT * FROM t LIMIT 0",
            "SELECT * FROM t LIMIT ALL OFFSET 0",
            "SELECT * FROM t LIMIT -1",  # Invalid but tests error handling
            
            # Between symmetric
            "SELECT * FROM t WHERE id BETWEEN SYMMETRIC 5 AND 1",
            
            # Is distinct from
            "SELECT * FROM t WHERE id IS DISTINCT FROM NULL",
            
            # Full outer join
            "SELECT * FROM t1 FULL OUTER JOIN t2 ON t1.id = t2.id",
            
            # Natural join
            "SELECT * FROM t1 NATURAL JOIN t2",
            
            # Cross join
            "SELECT * FROM t1 CROSS JOIN t2",
            
            # Self join
            "SELECT * FROM t a JOIN t b ON a.id = b.parent_id",
            
            # Correlated subquery
            "SELECT * FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.id = t1.id)",
            
            # Set operations
            "SELECT 1 UNION SELECT 2 UNION ALL SELECT 3 INTERSECT SELECT 1 EXCEPT SELECT 0",
            
            # Case with multiple when
            "SELECT CASE WHEN a THEN 1 WHEN b THEN 2 WHEN c THEN 3 ELSE 4 END FROM t",
            
            # Nested case
            "SELECT CASE WHEN a THEN CASE WHEN b THEN 1 ELSE 2 END ELSE 3 END FROM t",
            
            # In with subquery and list
            "SELECT * FROM t WHERE id IN (1, 2, 3) OR id IN (SELECT id FROM t2)",
            
            # Exists with correlation
            "SELECT * FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.col = t1.col)",
            
            # Quantified comparisons
            "SELECT * FROM t WHERE id > ALL (SELECT id FROM t2)",
            "SELECT * FROM t WHERE id = ANY (SELECT id FROM t2)",
            
            # Window frame
            "SELECT SUM(x) OVER (ORDER BY y ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM t",
        ]
    
    def _generate_valid_sql(self) -> str:
        """Generate a valid SQL statement using grammar templates."""
        templates = [
            # SELECT statements
            "SELECT {columns} FROM {table} {where} {group_by} {order_by} {limit}",
            "SELECT {distinct} {columns} FROM {table} {joins} {where}",
            "SELECT {columns} FROM {table} {join_type} JOIN {table2} ON {condition}",
            "SELECT {func}({column}) FROM {table} {where}",
            "SELECT * FROM (SELECT {columns} FROM {table}) AS sub {where}",
            
            # INSERT statements
            "INSERT INTO {table} ({columns}) VALUES ({values})",
            "INSERT INTO {table} SELECT {columns} FROM {table2} {where}",
            
            # UPDATE statements
            "UPDATE {table} SET {assignments} {where}",
            
            # DELETE statements
            "DELETE FROM {table} {where}",
            
            # CREATE statements
            "CREATE TABLE {table} ({column_defs})",
            "CREATE {temp}TABLE {table} AS SELECT {columns} FROM {table2}",
            
            # DROP statements
            "DROP TABLE IF EXISTS {table}",
            
            # ALTER statements
            "ALTER TABLE {table} ADD COLUMN {column} {type}",
            "ALTER TABLE {table} DROP COLUMN {column}",
            
            # Complex queries
            "SELECT {columns} FROM {table} WHERE {column} {op} (SELECT {agg}({column2}) FROM {table2})",
            "SELECT {columns} FROM {table1} WHERE EXISTS (SELECT 1 FROM {table2} WHERE {condition})",
        ]
        
        template = random.choice(templates)
        
        # Fill template with random elements
        replacements = {
            'table': random.choice(self.tables),
            'table1': random.choice(self.tables),
            'table2': random.choice(self.tables),
            'columns': self._generate_column_list(),
            'column': random.choice(self.columns),
            'column2': random.choice(self.columns),
            'values': self._generate_value_list(),
            'where': self._generate_where_clause(),
            'group_by': self._generate_group_by(),
            'order_by': self._generate_order_by(),
            'limit': self._generate_limit(),
            'distinct': "DISTINCT" if random.random() > 0.7 else "",
            'joins': self._generate_join_clause(),
            'join_type': random.choice(["INNER", "LEFT", "RIGHT", "FULL"]),
            'condition': self._generate_condition(),
            'func': random.choice(self.functions),
            'agg': random.choice(["COUNT", "SUM", "AVG", "MIN", "MAX"]),
            'assignments': self._generate_assignments(),
            'column_defs': self._generate_column_defs(),
            'temp': "TEMPORARY " if random.random() > 0.8 else "",
            'op': random.choice([">", "<", "=", ">=", "<=", "<>", "!=", "IN", "NOT IN"]),
        }
        
        # Apply replacements
        for key, value in replacements.items():
            template = template.replace(f'{{{key}}}', value if value else '')
        
        # Clean up extra whitespace
        template = re.sub(r'\\s+', ' ', template).strip()
        
        # Sometimes add semicolon
        if random.random() > 0.5:
            template += ";"
        
        return template
    
    def _generate_column_list(self) -> str:
        """Generate a list of columns."""
        num_cols = random.randint(1, 5)
        cols = []
        for _ in range(num_cols):
            if random.random() > 0.3:
                cols.append(random.choice(self.columns))
            else:
                cols.append(f"{{random.choice(self.functions)}}({{random.choice(self.columns)}})")
        return ", ".join(cols)
    
    def _generate_value_list(self) -> str:
        """Generate a list of values for INSERT."""
        num_values = random.randint(1, 5)
        values = []
        for _ in range(num_values):
            values.append(self._generate_literal())
        return ", ".join(values)
    
    def _generate_literal(self) -> str:
        """Generate a random literal value."""
        r = random.random()
        if r < 0.3:
            return str(random.randint(-1000, 1000))
        elif r < 0.6:
            return f"'{self._random_string()}'"
        elif r < 0.8:
            return "NULL"
        elif r < 0.9:
            return str(random.random() * 1000)
        else:
            return "TRUE" if random.random() > 0.5 else "FALSE"
    
    def _random_string(self, length=None) -> str:
        """Generate a random string."""
        if length is None:
            length = random.randint(1, 20)
        chars = string.ascii_letters + string.digits + " _-"
        return ''.join(random.choice(chars) for _ in range(length))
    
    def _generate_where_clause(self) -> str:
        """Generate a WHERE clause."""
        if random.random() > 0.5:
            return ""
        return f"WHERE {self._generate_condition()}"
    
    def _generate_condition(self) -> str:
        """Generate a condition expression."""
        conditions = [
            f"{{random.choice(self.columns)}} = {{self._generate_literal()}}",
            f"{{random.choice(self.columns)}} IN ({{self._generate_value_list()}})",
            f"{{random.choice(self.columns)}} BETWEEN {{self._generate_literal()}} AND {{self._generate_literal()}}",
            f"{{random.choice(self.columns)}} LIKE '%{{self._random_string(5)}}%'",
            f"{{random.choice(self.columns)}} IS NOT NULL",
            f"{{random.choice(self.columns)}} > (SELECT {{random.choice(['MIN', 'MAX', 'AVG'])}}({{random.choice(self.columns)}}) FROM {{random.choice(self.tables)}})",
            f"({{self._generate_condition()}}) {{random.choice(['AND', 'OR'])}} ({{self._generate_condition()}})",
        ]
        return random.choice(conditions)
    
    def _generate_group_by(self) -> str:
        if random.random() > 0.7:
            cols = random.sample(self.columns, min(2, len(self.columns)))
            return f"GROUP BY {', '.join(cols)}"
        return ""
    
    def _generate_order_by(self) -> str:
        if random.random() > 0.7:
            cols = random.sample(self.columns, min(2, len(self.columns)))
            orders = [f"{col} {'ASC' if random.random() > 0.5 else 'DESC'}" for col in cols]
            return f"ORDER BY {', '.join(orders)}"
        return ""
    
    def _generate_limit(self) -> str:
        if random.random() > 0.7:
            return f"LIMIT {random.randint(0, 100)}"
        return ""
    
    def _generate_join_clause(self) -> str:
        if random.random() > 0.5:
            return ""
        joins = [
            f"JOIN {{random.choice(self.tables)}} ON {{random.choice(self.columns)}} = {{random.choice(self.columns)}}",
            f"LEFT JOIN {{random.choice(self.tables)}} USING ({{random.choice(self.columns)}})",
        ]
        return random.choice(joins)
    
    def _generate_assignments(self) -> str:
        num = random.randint(1, 3)
        assignments = []
        for _ in range(num):
            assignments.append(f"{{random.choice(self.columns)}} = {{self._generate_literal()}}")
        return ", ".join(assignments)
    
    def _generate_column_defs(self) -> str:
        num = random.randint(1, 5)
        defs = []
        for _ in range(num):
            col = random.choice(self.columns)
            type_ = random.choice(self.data_types)
            constraints = []
            if random.random() > 0.5:
                constraints.append("PRIMARY KEY")
            if random.random() > 0.5:
                constraints.append("NOT NULL")
            if random.random() > 0.7:
                constraints.append(f"DEFAULT {self._generate_literal()}")
            defs.append(f"{col} {type_} {' '.join(constraints)}".strip())
        return ", ".join(defs)
    
    def _mutate_keyword(self, sql: str) -> str:
        """Mutate by changing a SQL keyword."""
        words = sql.split()
        if not words:
            return sql
        
        keyword_positions = [i for i, w in enumerate(words) if w.upper() in self.keywords]
        if not keyword_positions:
            return sql
        
        pos = random.choice(keyword_positions)
        original = words[pos]
        
        # Try to find a similar keyword or random one
        candidates = [k for k in self.keywords if k != original.upper()]
        if candidates:
            new_keyword = random.choice(candidates)
            words[pos] = new_keyword if original.isupper() else new_keyword.lower()
        
        return " ".join(words)
    
    def _mutate_identifier(self, sql: str) -> str:
        """Mutate by changing an identifier."""
        # Simple implementation: replace table/column names
        for table in self.tables:
            if table in sql and random.random() > 0.5:
                sql = sql.replace(table, random.choice(self.tables), 1)
                break
        
        for col in self.columns:
            if col in sql and random.random() > 0.5:
                sql = sql.replace(col, random.choice(self.columns), 1)
                break
        
        return sql
    
    def _mutate_literal(self, sql: str) -> str:
        """Mutate literal values."""
        # Find numbers and strings
        def replace_number(match):
            num = match.group()
            try:
                val = float(num) if '.' in num else int(num)
                mutation = random.choice([
                    str(val + random.randint(-10, 10)),
                    str(-val),
                    "0",
                    "NULL",
                    f"'{self._random_string()}'"
                ])
                return mutation
            except:
                return num
        
        def replace_string(match):
            return f"'{self._random_string()}'"
        
        # Replace numbers
        sql = re.sub(r'\\b\\d+\\.?\\d*\\b', replace_number, sql)
        # Replace string literals
        sql = re.sub(r"'(.*?)'", replace_string, sql)
        
        return sql
    
    def _mutate_add_clause(self, sql: str) -> str:
        """Add a random clause to the SQL."""
        clauses = [
            f"WHERE {self._generate_condition()}",
            f"ORDER BY {random.choice(self.columns)}",
            f"LIMIT {random.randint(1, 100)}",
            f"GROUP BY {random.choice(self.columns)}",
            f"HAVING COUNT(*) > {random.randint(0, 10)}",
        ]
        
        # Insert at end (before semicolon if exists)
        if ';' in sql:
            return sql.replace(';', f" {random.choice(clauses)};")
        else:
            return f"{sql} {random.choice(clauses)}"
    
    def _mutate_remove_clause(self, sql: str) -> str:
        """Remove a clause from the SQL."""
        clauses = ['WHERE', 'ORDER BY', 'GROUP BY', 'HAVING', 'LIMIT', 'OFFSET']
        for clause in clauses:
            if clause in sql.upper():
                # Simple removal - in practice would need proper parsing
                parts = re.split(f'\\b{clause}\\b', sql, flags=re.IGNORECASE, maxsplit=1)
                if len(parts) > 1:
                    # Remove everything after the clause up to next keyword or end
                    return parts[0].strip()
        return sql
    
    def _mutate_swap_clauses(self, sql: str) -> str:
        """Swap positions of two clauses."""
        # Simple implementation for demonstration
        if 'WHERE' in sql.upper() and 'ORDER BY' in sql.upper():
            sql = sql.upper()
            where_idx = sql.find('WHERE')
            order_idx = sql.find('ORDER BY')
            
            if where_idx < order_idx:
                # Extract clauses (simplified)
                before_where = sql[:where_idx]
                where_clause = sql[where_idx:order_idx]
                after_order = sql[order_idx:]
                return before_where + after_order + where_clause
        
        return sql
    
    def _mutate_nest_query(self, sql: str) -> str:
        """Wrap query in a nested SELECT."""
        if sql.lower().startswith('select'):
            return f"SELECT * FROM ({sql}) AS nested"
        return sql
    
    def _mutate_add_subquery(self, sql: str) -> str:
        """Add a subquery somewhere."""
        subqueries = [
            f"(SELECT {random.choice(self.columns)} FROM {random.choice(self.tables)})",
            f"(SELECT COUNT(*) FROM {random.choice(self.tables)})",
        ]
        
        # Insert at random position
        words = sql.split()
        if len(words) > 2:
            pos = random.randint(1, len(words)-1)
            words.insert(pos, random.choice(subqueries))
            return " ".join(words)
        return sql
    
    def mutate(self, sql: str) -> str:
        """Apply random mutation to SQL."""
        if not sql or random.random() > 0.7:
            return self._generate_valid_sql()
        
        mutator = random.choice(self.mutators)
        try:
            result = mutator(sql)
            # Ensure we don't return empty
            return result if result and result.strip() else self._generate_valid_sql()
        except:
            return self._generate_valid_sql()
    
    def generate_batch(self, batch_size: int = 50) -> List[str]:
        """Generate a batch of SQL statements using multiple strategies."""
        statements = []
        
        while len(statements) < batch_size:
            strategy = random.choices(
                ['valid', 'edge_case', 'mutated'],
                weights=[self.weights['valid'], self.weights['edge_case'], self.weights['mutated']]
            )[0]
            
            if strategy == 'valid':
                sql = self._generate_valid_sql()
            elif strategy == 'edge_case':
                sql = random.choice(self.edge_cases)
            else:  # mutated
                if self.corpus and random.random() > 0.3:
                    parent = random.choice(self.corpus)
                    sql = self.mutate(parent)
                else:
                    sql = self._generate_valid_sql()
            
            # Deduplicate
            sql_hash = hashlib.md5(sql.encode()).hexdigest()
            if sql_hash not in self.input_hashes:
                self.input_hashes.add(sql_hash)
                statements.append(sql)
                
                # Add to corpus if interesting (for mutation breeding)
                if random.random() > 0.8 or 'SELECT' in sql.upper():
                    self.corpus.append(sql)
                    # Keep corpus manageable
                    if len(self.corpus) > 1000:
                        self.corpus.pop(0)
        
        return statements
    
    def should_continue(self) -> bool:
        """Check if we should continue fuzzing based on time budget."""
        if self.start_time is None:
            self.start_time = time.time()
            return True
        
        elapsed = time.time() - self.start_time
        return elapsed < self.time_budget - 2.0  # Stop 2 seconds early
    
    def adaptive_adjust(self):
        """Adaptively adjust generation strategy based on progress."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        progress = elapsed / self.time_budget
        
        # Shift weights over time
        if progress < 0.3:
            # Early phase: focus on valid SQL
            self.weights = {'valid': 0.7, 'edge_case': 0.2, 'mutated': 0.1}
        elif progress < 0.7:
            # Middle phase: more exploration
            self.weights = {'valid': 0.4, 'edge_case': 0.4, 'mutated': 0.2}
        else:
            # Late phase: focus on edge cases and mutation
            self.weights = {'valid': 0.2, 'edge_case': 0.5, 'mutated': 0.3}

# Global fuzzer instance
_fuzzer = None

def fuzz(parse_sql):
    """
    Main fuzzing function called by evaluator.
    Generate SQL statements and execute through parser.
    """
    global _fuzzer
    
    if _fuzzer is None:
        _fuzzer = SQLFuzzer()
    
    if not _fuzzer.should_continue():
        return False
    
    # Adaptive adjustment
    _fuzzer.adaptive_adjust()
    
    # Generate batch of statements
    batch_size = random.randint(30, 100)  # Vary batch size
    statements = _fuzzer.generate_batch(batch_size)
    
    # Execute through parser
    if statements:
        parse_sql(statements)
    
    # Update statistics
    _fuzzer.stats['batches'] += 1
    _fuzzer.stats['statements'] += len(statements)
    
    return True
'''