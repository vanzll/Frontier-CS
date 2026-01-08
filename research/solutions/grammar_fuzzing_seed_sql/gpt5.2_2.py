import os
import sys
import re
import importlib
from dataclasses import is_dataclass

class Solution:
    def solve(self, resources_path: str) -> list[str]:
        resources_path = os.path.abspath(resources_path)
        if resources_path not in sys.path:
            sys.path.insert(0, resources_path)

        combined_upper = ""
        def _read_text(path: str) -> str:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception:
                return ""

        combined_upper += _read_text(os.path.join(resources_path, "sql_grammar.txt")).upper()
        engine_dir = os.path.join(resources_path, "sql_engine")
        combined_upper += _read_text(os.path.join(engine_dir, "parser.py")).upper()
        combined_upper += _read_text(os.path.join(engine_dir, "tokenizer.py")).upper()
        combined_upper += _read_text(os.path.join(engine_dir, "ast_nodes.py")).upper()

        try:
            sql_engine = importlib.import_module("sql_engine")
            parse_sql = getattr(sql_engine, "parse_sql", None)
            if parse_sql is None:
                parse_sql = importlib.import_module("sql_engine.parser").parse_sql
            ast_nodes_mod = importlib.import_module("sql_engine.ast_nodes")
        except Exception:
            return [
                "SELECT 1",
                "SELECT * FROM t WHERE a = 1",
                "INSERT INTO t(a) VALUES (1)",
                "UPDATE t SET a = 2 WHERE a = 1",
                "DELETE FROM t WHERE a = 2",
            ]

        def has_keywords(req):
            for kw in req:
                if kw.upper() not in combined_upper:
                    return False
            return True

        def extract_lex_features(stmt: str) -> set:
            s = stmt
            su = s.upper()
            feats = set()

            if "--" in s:
                feats.add("LEX:line_comment")
            if "/*" in s and "*/" in s:
                feats.add("LEX:block_comment")
            if "\n" in s:
                feats.add("LEX:newline")
            if "\t" in s:
                feats.add("LEX:tab")
            if ";" in s:
                feats.add("LEX:semicolon")
            if '"' in s:
                feats.add("LEX:double_quote")
            if "`" in s:
                feats.add("LEX:backtick")
            if "[" in s and "]" in s:
                feats.add("LEX:bracket")
            if "''" in s:
                feats.add("LEX:escaped_single_quote")
            if re.search(r"\b\d+(\.\d+)\b", s):
                feats.add("LEX:float")
            if re.search(r"\b\d+[eE][+-]?\d+\b", s):
                feats.add("LEX:exp_number")
            if re.search(r"\b0x[0-9a-fA-F]+\b", s):
                feats.add("LEX:hex_number")
            if "!=" in s:
                feats.add("LEX:neq_bang")
            if "<>" in s:
                feats.add("LEX:neq_angle")
            if "||" in s:
                feats.add("LEX:concat_op")
            if "?" in s:
                feats.add("LEX:qmark_param")
            if re.search(r":[A-Za-z_][A-Za-z_0-9]*\b", s):
                feats.add("LEX:named_param")
            if re.search(r"\$[0-9]+\b", s):
                feats.add("LEX:dollar_param")
            if re.search(r"\bX'[0-9A-Fa-f]+'\b", s):
                feats.add("LEX:blob_literal")
            if re.search(r"\bNULL\b", su):
                feats.add("KW:NULL")
            if re.search(r"\bTRUE\b", su):
                feats.add("KW:TRUE")
            if re.search(r"\bFALSE\b", su):
                feats.add("KW:FALSE")

            optional_keywords = [
                "WITH", "RECURSIVE", "SELECT", "DISTINCT", "ALL",
                "FROM", "WHERE", "GROUP", "BY", "HAVING", "ORDER", "LIMIT", "OFFSET",
                "UNION", "INTERSECT", "EXCEPT",
                "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "CROSS", "NATURAL", "OUTER",
                "ON", "USING",
                "INSERT", "INTO", "VALUES",
                "UPDATE", "SET",
                "DELETE",
                "CREATE", "TABLE", "VIEW", "INDEX", "DROP", "ALTER",
                "CASE", "WHEN", "THEN", "ELSE", "END",
                "CAST", "AS",
                "EXISTS", "IN", "BETWEEN", "LIKE", "ESCAPE",
                "IS", "NOT",
                "PRIMARY", "KEY", "UNIQUE", "CHECK", "DEFAULT", "REFERENCES", "FOREIGN",
                "BEGIN", "COMMIT", "ROLLBACK",
                "EXPLAIN", "PRAGMA",
            ]
            for kw in optional_keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", su):
                    feats.add("KW:" + kw)

            return feats

        def collect_ast_node_types(root) -> set:
            types = set()
            stack = [root]
            seen = set()

            while stack:
                obj = stack.pop()
                if obj is None:
                    continue
                oid = id(obj)
                if oid in seen:
                    continue
                seen.add(oid)

                t = type(obj)
                if getattr(t, "__module__", None) == getattr(ast_nodes_mod, "__name__", ""):
                    types.add(t.__name__)

                if isinstance(obj, (str, bytes, int, float, bool)):
                    continue

                if isinstance(obj, dict):
                    try:
                        stack.extend(obj.values())
                    except Exception:
                        pass
                    continue

                if isinstance(obj, (list, tuple, set, frozenset)):
                    try:
                        stack.extend(list(obj))
                    except Exception:
                        pass
                    continue

                if is_dataclass(obj):
                    try:
                        for f in obj.__dataclass_fields__.values():
                            try:
                                stack.append(getattr(obj, f.name))
                            except Exception:
                                pass
                    except Exception:
                        pass
                    continue

                if hasattr(obj, "__dict__"):
                    try:
                        stack.extend(list(obj.__dict__.values()))
                    except Exception:
                        pass
                    continue

                if isinstance(obj, tuple) and hasattr(obj, "_fields"):
                    try:
                        stack.extend(list(obj))
                    except Exception:
                        pass

            return types

        def try_parse_variants(stmt: str):
            base = stmt
            if base is None:
                return None, None
            base = base.strip()
            variants = []
            if base:
                variants.append(base)
            if base.endswith(";"):
                v = base.rstrip().rstrip(";").rstrip()
                if v and v != base:
                    variants.append(v)
            else:
                variants.append(base + ";")

            seenv = set()
            for v in variants:
                if not v:
                    continue
                if v in seenv:
                    continue
                seenv.add(v)
                try:
                    ast = parse_sql(v)
                    return v, ast
                except Exception:
                    continue
            return None, None

        # Candidate assembly
        candidates = []

        def add(cat: str, stmt: str, req=None, group=None):
            if req and not has_keywords(req):
                return
            candidates.append({"cat": cat, "stmt": stmt, "group": group})

        # Base selects (group)
        add("SELECT_BASIC", "SELECT 1", group="BASE_SELECT")
        add("SELECT_BASIC", "SELECT 1 FROM t", group="BASE_SELECT")
        add("SELECT_BASIC", "SELECT NULL", req=["NULL"])
        add("SELECT_BASIC", "SELECT TRUE, FALSE", req=["TRUE", "FALSE"])
        add("SELECT_BASIC", "SELECT 'a''b'", group="STRING_ESC")
        add("SELECT_BASIC", "SELECT 1e2, 1.2E-3", group="NUM_FORMS")

        # Tokenizer/comment/quoting (groups)
        add("TOKENIZER", "-- comment\nSELECT 1", group="COMMENT")
        add("TOKENIZER", "/* block comment */ SELECT 1", group="COMMENT")
        add("TOKENIZER", 'SELECT "Col Name" AS "Alias" FROM "My Table"', group="QUOTED_IDENT")
        add("TOKENIZER", "SELECT `a` FROM `t`", group="QUOTED_IDENT")
        add("TOKENIZER", "SELECT [a] FROM [t]", group="QUOTED_IDENT")

        # FROM / alias / qualified / star
        add("SELECT_FROM", "SELECT * FROM t")
        add("SELECT_FROM", "SELECT t.* FROM t AS t")
        add("SELECT_FROM", "SELECT s.t.a FROM s.t", group="QUALIFIED_IDENT")
        add("SELECT_FROM", "SELECT a AS x, b y FROM t tt", group="ALIASES")

        # WHERE variants
        add("SELECT_WHERE", "SELECT * FROM t WHERE a = 1")
        add("SELECT_WHERE", "SELECT * FROM t WHERE a <> 1 AND b != 2 OR c <= 3", req=["WHERE"])
        add("SELECT_WHERE", "SELECT * FROM t WHERE NOT (a = 1 OR b = 2)", req=["NOT"])
        add("SELECT_WHERE", "SELECT * FROM t WHERE a BETWEEN 1 AND 10", req=["BETWEEN"])
        add("SELECT_WHERE", "SELECT * FROM t WHERE a IN (1, 2, 3)", req=["IN"])
        add("SELECT_WHERE", "SELECT * FROM t WHERE a LIKE 'A%'", req=["LIKE"])
        add("SELECT_WHERE", "SELECT * FROM t WHERE a IS NULL", req=["IS", "NULL"])
        add("SELECT_WHERE", "SELECT * FROM t WHERE a IS NOT NULL", req=["IS", "NOT", "NULL"])
        add("SELECT_WHERE", "SELECT * FROM t WHERE a IN (SELECT a FROM t2)", req=["IN", "SELECT"])
        add("SELECT_WHERE", "SELECT * FROM t WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.id = t.id)", req=["EXISTS"])

        # Expressions / functions / case / cast
        add("EXPRESSIONS", "SELECT (1 + 2) * 3 - 4 / 2")
        add("EXPRESSIONS", "SELECT -1, +2", group="UNARY_OPS")
        add("EXPRESSIONS", "SELECT COALESCE(NULL, 1, 2)", req=["COALESCE", "NULL"])
        add("EXPRESSIONS", "SELECT NULLIF(1, 1)", req=["NULLIF"])
        add("EXPRESSIONS", "SELECT CASE WHEN 1 = 1 THEN 'yes' ELSE 'no' END", req=["CASE", "WHEN", "THEN", "END"])
        add("EXPRESSIONS", "SELECT CAST('1' AS INT)", req=["CAST", "AS"])
        add("EXPRESSIONS", "SELECT a || b FROM t", req=["||"])
        add("EXPRESSIONS", "SELECT COUNT(*), SUM(a), MIN(a), MAX(a) FROM t", req=["COUNT", "SUM", "MIN", "MAX"])

        # DISTINCT / ORDER / LIMIT / OFFSET
        add("SELECT_MODS", "SELECT DISTINCT a FROM t", req=["DISTINCT"])
        add("SELECT_MODS", "SELECT a FROM t ORDER BY a DESC, b ASC", req=["ORDER", "BY"])
        add("SELECT_MODS", "SELECT a FROM t ORDER BY 1", req=["ORDER"])
        add("SELECT_MODS", "SELECT * FROM t LIMIT 10", req=["LIMIT"])
        add("SELECT_MODS", "SELECT * FROM t LIMIT 10 OFFSET 5", req=["LIMIT", "OFFSET"])

        # GROUP BY / HAVING
        add("SELECT_GROUP", "SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(*) > 1", req=["GROUP", "BY", "HAVING"])

        # Joins
        add("SELECT_JOIN", "SELECT * FROM t1 JOIN t2 ON t1.id = t2.id", req=["JOIN", "ON"])
        add("SELECT_JOIN", "SELECT * FROM t1 INNER JOIN t2 ON t1.id = t2.id", req=["INNER", "JOIN"])
        add("SELECT_JOIN", "SELECT * FROM t1 LEFT JOIN t2 ON t1.id = t2.id", req=["LEFT", "JOIN"])
        add("SELECT_JOIN", "SELECT * FROM t1 LEFT OUTER JOIN t2 ON t1.id = t2.id", req=["LEFT", "OUTER", "JOIN"])
        add("SELECT_JOIN", "SELECT * FROM t1 RIGHT JOIN t2 ON t1.id = t2.id", req=["RIGHT", "JOIN"])
        add("SELECT_JOIN", "SELECT * FROM t1 FULL JOIN t2 ON t1.id = t2.id", req=["FULL", "JOIN"])
        add("SELECT_JOIN", "SELECT * FROM t1 CROSS JOIN t2", req=["CROSS", "JOIN"])
        add("SELECT_JOIN", "SELECT * FROM t1 NATURAL JOIN t2", req=["NATURAL", "JOIN"])
        add("SELECT_JOIN", "SELECT * FROM t1 JOIN t2 USING (id)", req=["USING", "JOIN"])

        # Subqueries in FROM
        add("SELECT_SUBQUERY", "SELECT * FROM (SELECT 1 AS x) sub", req=["SELECT"])
        add("SELECT_SUBQUERY", "SELECT x FROM (SELECT 1 AS x UNION ALL SELECT 2) u", req=["UNION", "ALL"])

        # Set ops
        add("SET_OP", "SELECT 1 UNION SELECT 2", req=["UNION"])
        add("SET_OP", "SELECT 1 UNION ALL SELECT 2", req=["UNION", "ALL"])
        add("SET_OP", "SELECT 1 INTERSECT SELECT 1", req=["INTERSECT"])
        add("SET_OP", "SELECT 1 EXCEPT SELECT 2", req=["EXCEPT"])

        # WITH / recursive
        add("CTE", "WITH c AS (SELECT 1 AS x) SELECT x FROM c", req=["WITH"])
        add("CTE", "WITH c(x) AS (SELECT 1) SELECT x FROM c", req=["WITH"])
        add("CTE", "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c WHERE n < 3) SELECT n FROM c", req=["WITH", "RECURSIVE", "UNION", "ALL", "WHERE"])

        # INSERT / UPDATE / DELETE
        add("INSERT", "INSERT INTO t(a, b) VALUES (1, 'x')", req=["INSERT", "INTO", "VALUES"])
        add("INSERT", "INSERT INTO t VALUES (1, 'x')", req=["INSERT", "VALUES"])
        add("INSERT", "INSERT INTO t(a) SELECT 1", req=["INSERT", "SELECT"])
        add("INSERT", "INSERT INTO t(a) VALUES (1), (2), (3)", req=["INSERT", "VALUES"])

        add("UPDATE", "UPDATE t SET a = 1 WHERE id = 1", req=["UPDATE", "SET"])
        add("UPDATE", "UPDATE t SET a = a + 1, b = b || 'x' WHERE b IS NOT NULL", req=["UPDATE", "SET", "WHERE"])

        add("DELETE", "DELETE FROM t WHERE id = 1", req=["DELETE", "FROM"])
        add("DELETE", "DELETE FROM t WHERE a IN (SELECT a FROM t2)", req=["DELETE", "IN", "SELECT"])

        # DDL
        add("CREATE", "CREATE TABLE t(id INT, name TEXT)", req=["CREATE", "TABLE"])
        add("CREATE", "CREATE TABLE IF NOT EXISTS t(id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INT DEFAULT 0)", req=["CREATE", "TABLE", "IF", "NOT", "EXISTS"])
        add("CREATE", "CREATE TABLE t(id INT, name TEXT, PRIMARY KEY (id))", req=["PRIMARY", "KEY"])
        add("CREATE", "CREATE TABLE t(id INT, parent_id INT, FOREIGN KEY (parent_id) REFERENCES t(id))", req=["FOREIGN", "REFERENCES"])

        add("DROP", "DROP TABLE t", req=["DROP", "TABLE"])
        add("DROP", "DROP TABLE IF EXISTS t", req=["DROP", "TABLE", "IF", "EXISTS"])

        add("ALTER", "ALTER TABLE t ADD COLUMN c INT", req=["ALTER", "TABLE", "ADD"])
        add("ALTER", "ALTER TABLE t RENAME TO t2", req=["ALTER", "TABLE", "RENAME"])
        add("ALTER", "ALTER TABLE t RENAME COLUMN a TO b", req=["ALTER", "TABLE", "RENAME", "COLUMN"])

        add("VIEW", "CREATE VIEW v AS SELECT * FROM t", req=["CREATE", "VIEW", "AS"])
        add("INDEX", "CREATE INDEX idx_t_a ON t(a)", req=["CREATE", "INDEX", "ON"])

        # Transactions / misc
        add("TRANSACTION", "BEGIN", req=["BEGIN"])
        add("TRANSACTION", "COMMIT", req=["COMMIT"])
        add("TRANSACTION", "ROLLBACK", req=["ROLLBACK"])
        add("MISC", "EXPLAIN SELECT 1", req=["EXPLAIN"])
        add("MISC", "PRAGMA foreign_keys = ON", req=["PRAGMA"])
        add("MISC", "SELECT ? AS p1, :name AS p2, $1 AS p3", req=["SELECT"])

        add("LITERALS", "SELECT X'0A0B'", req=["X'"])
        add("LITERALS", "SELECT 0x2A", req=["0X"])

        # Validate and compute features
        valid = []
        by_group = {}
        for cand in candidates:
            stmt0 = cand["stmt"]
            parsed_stmt, ast = try_parse_variants(stmt0)
            if parsed_stmt is None:
                continue
            try:
                node_types = collect_ast_node_types(ast)
            except Exception:
                node_types = set()
            lex_feats = extract_lex_features(parsed_stmt)
            feats = set(node_types) | lex_feats | {f"CAT:{cand['cat']}"}
            info = {
                "cat": cand["cat"],
                "stmt": parsed_stmt,
                "features": feats,
                "length": len(parsed_stmt),
                "group": cand.get("group"),
            }
            valid.append(info)
            g = info["group"]
            if g:
                by_group.setdefault(g, []).append(info)

        if not valid:
            return ["SELECT 1"]

        # Choose best per group (optional)
        selected = []
        selected_set = set()
        covered = set()

        def _pick_best(items):
            if not items:
                return None
            items_sorted = sorted(items, key=lambda x: (-len(x["features"]), x["length"], x["stmt"]))
            return items_sorted[0]

        must_groups = ["BASE_SELECT", "COMMENT", "QUOTED_IDENT", "STRING_ESC", "NUM_FORMS"]
        for g in must_groups:
            best = _pick_best(by_group.get(g, []))
            if best and best["stmt"] not in selected_set:
                selected.append(best)
                selected_set.add(best["stmt"])
                covered |= best["features"]

        remaining = [c for c in valid if c["stmt"] not in selected_set]

        max_cases = 35

        # Greedy by feature gain
        while remaining and len(selected) < max_cases:
            best_idx = -1
            best_gain = 0
            best_len = 10**18
            for i, c in enumerate(remaining):
                gain = len(c["features"] - covered)
                if gain > best_gain or (gain == best_gain and gain > 0 and c["length"] < best_len):
                    best_idx = i
                    best_gain = gain
                    best_len = c["length"]
            if best_idx < 0 or best_gain <= 0:
                break
            best = remaining.pop(best_idx)
            if best["stmt"] in selected_set:
                continue
            selected.append(best)
            selected_set.add(best["stmt"])
            covered |= best["features"]

        # Ensure at least one per major category if available (within cap)
        preferred_cat_order = [
            "SELECT_BASIC", "TOKENIZER", "SELECT_FROM", "SELECT_WHERE", "EXPRESSIONS",
            "SELECT_GROUP", "SELECT_MODS", "SELECT_JOIN", "SELECT_SUBQUERY",
            "SET_OP", "CTE", "INSERT", "UPDATE", "DELETE",
            "CREATE", "ALTER", "DROP", "VIEW", "INDEX",
            "TRANSACTION", "MISC", "LITERALS",
        ]
        selected_cats = {c["cat"] for c in selected}
        if len(selected) < max_cases:
            remaining_all = [c for c in valid if c["stmt"] not in selected_set]
            for cat in preferred_cat_order:
                if len(selected) >= max_cases:
                    break
                if cat in selected_cats:
                    continue
                opts = [c for c in remaining_all if c["cat"] == cat]
                if not opts:
                    continue
                best = sorted(opts, key=lambda x: (-len(x["features"] - covered), x["length"], x["stmt"]))[0]
                if best["stmt"] in selected_set:
                    continue
                selected.append(best)
                selected_set.add(best["stmt"])
                selected_cats.add(cat)
                covered |= best["features"]

        # Final stable order: keep chosen order but remove any accidental duplicates
        out = []
        seen = set()
        for c in selected:
            s = c["stmt"].strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)

        if not out:
            out = ["SELECT 1"]
        return out