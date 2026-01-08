import os
import sys
import re
import importlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union


@dataclass(frozen=True)
class Terminal:
    value: str


@dataclass(frozen=True)
class NonTerminal:
    name: str


Symbol = Union[Terminal, NonTerminal]


class GrammarSQLGenerator:
    def __init__(
        self,
        grammar: Dict[str, List[List[Symbol]]],
        start_symbol: str,
        min_expansions: Dict[str, Optional[List[str]]],
        parse_sql,
    ):
        self.grammar = grammar
        self.start_symbol = start_symbol
        self.min_expansions = min_expansions or {}
        self.parse_sql = parse_sql
        self.max_depth = 8

    def _alt_contains_nt(self, alt: List[Symbol], nt_name: str) -> bool:
        for sym in alt:
            if isinstance(sym, NonTerminal) and sym.name == nt_name:
                return True
        return False

    def _fallback_for_nt(self, nt_name: str) -> List[str]:
        if nt_name in self.min_expansions and self.min_expansions[nt_name] is not None:
            return list(self.min_expansions[nt_name])
        # Generic placeholder
        return ["1"]

    def expand_nt(
        self,
        nt_name: str,
        depth: int,
        preferred_alts: Dict[str, int],
        used_preferred: Set[str],
    ) -> List[str]:
        if depth > self.max_depth:
            return self._fallback_for_nt(nt_name)

        alts = self.grammar.get(nt_name)
        if not alts:
            return self._fallback_for_nt(nt_name)

        chosen_idx: Optional[int] = None

        if nt_name in preferred_alts and nt_name not in used_preferred:
            idx = preferred_alts[nt_name]
            if 0 <= idx < len(alts):
                chosen_idx = idx
                used_preferred.add(nt_name)

        if chosen_idx is None:
            if depth > self.max_depth // 2:
                for i, alt in enumerate(alts):
                    if not self._alt_contains_nt(alt, nt_name):
                        chosen_idx = i
                        break
            if chosen_idx is None:
                chosen_idx = 0

        alt = alts[chosen_idx]
        tokens: List[str] = []
        for sym in alt:
            if isinstance(sym, Terminal):
                if sym.value:
                    tokens.append(sym.value)
            else:
                sub = self.expand_nt(sym.name, depth + 1, preferred_alts, used_preferred)
                if sub:
                    tokens.extend(sub)
        return tokens

    def format_tokens(self, tokens: List[str]) -> str:
        punct_no_space_before = {",", ")", ";", ".", "]", "}"}
        punct_no_space_after = {"(", "[", "{"}
        s = ""
        for tok in tokens:
            if not tok:
                continue
            if tok in punct_no_space_before:
                s = s.rstrip()
                s += tok + " "
            elif tok in punct_no_space_after:
                if s and not s.endswith(" "):
                    s += " "
                s += tok
            else:
                if s and not s.endswith(" "):
                    s += " "
                s += tok
        return s.strip()

    def generate_sentence(self, preferred_alts: Optional[Dict[str, int]] = None) -> Optional[str]:
        if self.start_symbol not in self.grammar:
            return None
        used_preferred: Set[str] = set()
        tokens = self.expand_nt(self.start_symbol, 0, preferred_alts or {}, used_preferred)
        if not tokens:
            return None
        stmt = self.format_tokens(tokens).strip()
        if not stmt:
            return None

        if self.parse_sql is None:
            return stmt

        try:
            self.parse_sql(stmt)
            return stmt
        except Exception:
            if not stmt.endswith(";"):
                candidate = stmt + ";"
                try:
                    self.parse_sql(candidate)
                    return candidate
                except Exception:
                    return None
            return None

    def generate_test_cases(self, max_statements: int = 120) -> List[str]:
        statements: List[str] = []
        seen: Set[str] = set()

        def add_stmt(sql: Optional[str]):
            if sql and sql not in seen:
                seen.add(sql)
                statements.append(sql)

        base = self.generate_sentence({})
        add_stmt(base)

        start_alts = self.grammar.get(self.start_symbol, [])
        for i in range(1, len(start_alts)):
            if len(statements) >= max_statements:
                break
            s = self.generate_sentence({self.start_symbol: i})
            add_stmt(s)

        if len(statements) >= max_statements:
            return statements

        for nt, alts in self.grammar.items():
            if len(statements) >= max_statements:
                break
            if nt == self.start_symbol:
                continue
            if len(alts) <= 1:
                continue
            for i in range(len(alts)):
                if len(statements) >= max_statements:
                    break
                s = self.generate_sentence({nt: i})
                add_stmt(s)

        return statements


class Solution:
    TOKEN_NAME_MAP: Dict[str, str] = {
        "COMMA": ",",
        "SEMI": ";",
        "SEMICOLON": ";",
        "LPAREN": "(",
        "RPAREN": ")",
        "LBRACKET": "[",
        "RBRACKET": "]",
        "LBRACE": "{",
        "RBRACE": "}",
        "DOT": ".",
        "STAR": "*",
        "ASTERISK": "*",
        "PLUS": "+",
        "MINUS": "-",
        "SLASH": "/",
        "DIVIDE": "/",
        "PERCENT": "%",
        "MOD": "%",
        "EQ": "=",
        "EQUALS": "=",
        "NEQ": "<>",
        "NOTEQUALS": "<>",
        "LT": "<",
        "GT": ">",
        "LTE": "<=",
        "GTE": ">=",
    }

    def solve(self, resources_path: str) -> list[str]:
        grammar_path = os.path.join(resources_path, "sql_grammar.txt")
        grammar: Dict[str, List[List[Symbol]]] = {}

        try:
            if os.path.exists(grammar_path):
                grammar = self._parse_grammar(grammar_path)
        except Exception:
            grammar = {}

        parse_sql = self._load_parse_sql(resources_path)

        statements: List[str] = []

        if grammar:
            start_symbol = self._choose_start_symbol(grammar)
            min_expansions = self._precompute_min_expansions(grammar, max_depth=6)
            generator = GrammarSQLGenerator(grammar, start_symbol, min_expansions, parse_sql)
            try:
                statements = generator.generate_test_cases(max_statements=120)
            except Exception:
                statements = []

        if len(statements) < 10:
            fallback = self._fallback_statements(parse_sql)
            seen = set(statements)
            for sql in fallback:
                if sql not in seen:
                    seen.add(sql)
                    statements.append(sql)
                if len(statements) >= 120:
                    break

        if not statements:
            statements = ["SELECT 1"]

        return statements

    def _load_parse_sql(self, resources_path: str):
        parse_sql = None
        try:
            if resources_path not in sys.path:
                sys.path.insert(0, resources_path)
            try:
                engine_pkg = importlib.import_module("sql_engine")
            except Exception:
                engine_pkg = None

            if engine_pkg is not None:
                if hasattr(engine_pkg, "parse_sql"):
                    parse_sql = getattr(engine_pkg, "parse_sql")
                else:
                    try:
                        parser_mod = importlib.import_module("sql_engine.parser")
                        if hasattr(parser_mod, "parse_sql"):
                            parse_sql = getattr(parser_mod, "parse_sql")
                    except Exception:
                        parse_sql = None
        except Exception:
            parse_sql = None
        return parse_sql

    def _parse_grammar(self, path: str) -> Dict[str, List[List[Symbol]]]:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        lines = raw.splitlines()
        clean_lines: List[str] = []
        for line in lines:
            line = re.sub(r"(--|//|#).*", "", line)
            stripped = line.rstrip()
            if stripped.strip():
                clean_lines.append(stripped)

        grammar: Dict[str, List[List[Symbol]]] = {}
        current_lhs: Optional[str] = None
        current_rhs: str = ""

        for line in clean_lines:
            stripped = line.strip()
            if not stripped:
                continue

            m = re.match(r"^([^\s]+)\s*(::=|:)\s*(.*)$", stripped)
            if m:
                if current_lhs is not None:
                    self._add_rule(grammar, current_lhs, current_rhs)
                lhs_raw = m.group(1)
                op_rhs = m.group(3).strip()
                current_lhs = self._normalize_nonterminal(lhs_raw)
                current_rhs = op_rhs
                if current_rhs.endswith(";"):
                    current_rhs = current_rhs[:-1].strip()
                    self._add_rule(grammar, current_lhs, current_rhs)
                    current_lhs = None
                    current_rhs = ""
            else:
                if current_lhs is None:
                    continue
                if current_rhs:
                    current_rhs += " "
                current_rhs += stripped
                if stripped.endswith(";"):
                    current_rhs = current_rhs[:-1].strip()
                    self._add_rule(grammar, current_lhs, current_rhs)
                    current_lhs = None
                    current_rhs = ""

        if current_lhs is not None and current_rhs:
            self._add_rule(grammar, current_lhs, current_rhs)

        return grammar

    def _add_rule(self, grammar: Dict[str, List[List[Symbol]]], lhs: str, rhs_text: str) -> None:
        rhs_text = rhs_text.strip()
        if not rhs_text:
            grammar.setdefault(lhs, []).append([])
            return

        alts = [s.strip() for s in rhs_text.split("|") if s.strip()]

        for alt in alts:
            tokens = self._tokenize_alt(alt)
            grammar.setdefault(lhs, []).append(tokens)

    def _tokenize_alt(self, alt: str) -> List[Symbol]:
        alt_stripped = alt.strip()
        if not alt_stripped:
            return []
        if alt_stripped.lower() in {"ε", "epsilon", "empty", "eps"}:
            return []

        pattern = r"""'[^']*'|"[^"]*"|<[^>]+>|\S+"""
        raw_tokens = re.findall(pattern, alt_stripped)

        symbols: List[Symbol] = []
        for tok in raw_tokens:
            tok = tok.strip()
            if not tok:
                continue
            if (tok[0] == "'" and tok[-1] == "'" and len(tok) >= 2) or (
                tok[0] == '"' and tok[-1] == '"' and len(tok) >= 2
            ):
                literal = tok[1:-1]
                symbols.append(Terminal(self._map_terminal_literal(literal)))
            elif tok.startswith("<") and tok.endswith(">") and len(tok) >= 3:
                name = tok[1:-1].strip()
                nt_name = self._normalize_nonterminal(name)
                symbols.append(NonTerminal(nt_name))
            else:
                if re.match(r"^[^\w<>]+$", tok):
                    symbols.append(Terminal(tok))
                elif re.match(r"^[A-Z_]+$", tok):
                    lexeme = self._uppercase_symbol_to_lexeme(tok)
                    symbols.append(Terminal(lexeme))
                else:
                    nt_name = self._normalize_nonterminal(tok)
                    symbols.append(NonTerminal(nt_name))
        return symbols

    def _normalize_nonterminal(self, name: str) -> str:
        name = name.strip()
        if name.startswith("<") and name.endswith(">"):
            name = name[1:-1].strip()
        if name.startswith("?"):
            name = name[1:]
        return name.strip()

    def _map_terminal_literal(self, literal: str) -> str:
        return literal

    def _uppercase_symbol_to_lexeme(self, sym: str) -> str:
        if sym in self.TOKEN_NAME_MAP:
            return self.TOKEN_NAME_MAP[sym]

        if "IDENT" in sym or sym in {"ID"} or sym.endswith("_ID") or sym.endswith("_NAME") or sym.endswith("_ALIAS") or sym.endswith("_IDENTIFIER"):
            return "col1"

        if any(x in sym for x in ["INT", "INTEGER", "NUMBER", "DIGIT", "NUMERIC", "DECIMAL", "FLOAT", "DOUBLE"]):
            return "1"

        if any(x in sym for x in ["STRING", "CHAR", "TEXT", "CSTRING", "VARCHAR"]):
            return "'text'"

        if any(x in sym for x in ["BOOL", "BOOLEAN"]):
            return "TRUE"

        return sym

    def _precompute_min_expansions(
        self, grammar: Dict[str, List[List[Symbol]]], max_depth: int = 5
    ) -> Dict[str, Optional[List[str]]]:
        memo: Dict[str, Optional[List[str]]] = {}

        def helper(nt: str, depth: int, path: Set[str]) -> Optional[List[str]]:
            if nt in memo:
                return memo[nt]
            if depth > max_depth or nt in path:
                memo[nt] = None
                return None
            alts = grammar.get(nt)
            if not alts:
                memo[nt] = None
                return None
            best: Optional[List[str]] = None
            for alt in alts:
                tokens: List[str] = []
                ok = True
                for sym in alt:
                    if isinstance(sym, Terminal):
                        tokens.append(sym.value)
                    else:
                        sub = helper(sym.name, depth + 1, path | {nt})
                        if sub is None:
                            ok = False
                            break
                        tokens.extend(sub)
                if ok:
                    if best is None or len(tokens) < len(best):
                        best = tokens
            memo[nt] = best
            return best

        for nt in grammar.keys():
            helper(nt, 0, set())

        return memo

    def _choose_start_symbol(self, grammar: Dict[str, List[List[Symbol]]]) -> str:
        preferred = [
            "start",
            "statement",
            "statements",
            "sql",
            "sql_stmt",
            "sql_stmts",
            "query",
            "queries",
            "program",
        ]
        keys = list(grammar.keys())
        lower_map = {k.lower(): k for k in keys}
        for cand in preferred:
            if cand in lower_map:
                return lower_map[cand]
        return keys[0] if keys else "start"

    def _fallback_statements(self, parse_sql) -> List[str]:
        templates = [
            "SELECT 1",
            "SELECT 1 + 2 AS sum",
            "SELECT * FROM t1",
            "SELECT col1, col2 FROM t1",
            "SELECT t1.col1, t2.col2 FROM t1 INNER JOIN t2 ON t1.id = t2.id",
            "SELECT col1, COUNT(*) FROM t1 GROUP BY col1",
            "SELECT col1 FROM t1 WHERE col2 = 10",
            "SELECT col1 FROM t1 WHERE col2 BETWEEN 1 AND 10",
            "SELECT col1 FROM t1 WHERE col2 IN (1, 2, 3)",
            "SELECT col1 FROM t1 WHERE col2 IS NULL",
            "SELECT col1 FROM t1 WHERE col2 IS NOT NULL",
            "SELECT col1 FROM t1 WHERE col2 LIKE 'abc%'",
            "SELECT DISTINCT col1 FROM t1",
            "SELECT col1 FROM t1 ORDER BY col1 DESC",
            "SELECT col1 FROM t1 ORDER BY col1 ASC, col2 DESC",
            "SELECT col1 FROM t1 LIMIT 10",
            "SELECT col1 FROM t1 OFFSET 5",
            "SELECT col1 FROM t1 LIMIT 5 OFFSET 10",
            "SELECT CASE WHEN col1 > 0 THEN 'pos' ELSE 'neg' END AS sign FROM t1",
            "SELECT (SELECT MAX(col2) FROM t2) AS max_col2 FROM t1",
            "SELECT col1 FROM t1 UNION SELECT col1 FROM t2",
            "SELECT col1 FROM t1 INTERSECT SELECT col1 FROM t2",
            "SELECT col1 FROM t1 EXCEPT SELECT col1 FROM t2",
            "SELECT * FROM t1 LEFT JOIN t2 ON t1.id = t2.id",
            "SELECT * FROM t1 RIGHT JOIN t2 ON t1.id = t2.id",
            "SELECT * FROM t1 FULL OUTER JOIN t2 ON t1.id = t2.id",
            "SELECT ABS(-1), UPPER('abc'), COUNT(DISTINCT col1) FROM t1",
            "SELECT col1 FROM t1 WHERE col2 = (SELECT MAX(col2) FROM t2)",
            "INSERT INTO t1 (col1, col2) VALUES (1, 'a')",
            "INSERT INTO t1 VALUES (1, 'a')",
            "UPDATE t1 SET col1 = 2 WHERE col2 = 'a'",
            "DELETE FROM t1 WHERE col1 = 1",
            "CREATE TABLE t1 (id INT PRIMARY KEY, col1 VARCHAR(100) NOT NULL)",
            "CREATE TABLE t2 (id INT, t1_id INT, CONSTRAINT fk_t1 FOREIGN KEY (t1_id) REFERENCES t1(id))",
            "DROP TABLE t1",
            "ALTER TABLE t1 ADD COLUMN col3 INT",
            "ALTER TABLE t1 DROP COLUMN col2",
            "CREATE INDEX idx_t1_col1 ON t1(col1)",
            "BEGIN TRANSACTION",
            "COMMIT",
            "ROLLBACK",
        ]

        results: List[str] = []
        for sql in templates:
            stmt = sql.strip()
            if not stmt:
                continue
            if parse_sql is not None:
                try:
                    parse_sql(stmt)
                    results.append(stmt)
                    continue
                except Exception:
                    if not stmt.endswith(";"):
                        candidate = stmt + ";"
                        try:
                            parse_sql(candidate)
                            results.append(candidate)
                            continue
                        except Exception:
                            continue
                    else:
                        continue
            else:
                results.append(stmt)
        if not results:
            results.append("SELECT 1")
        return results