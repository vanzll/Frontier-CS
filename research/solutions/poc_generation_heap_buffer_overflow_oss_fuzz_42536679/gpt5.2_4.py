import os
import re
import tarfile
import struct
import binascii
from typing import Dict, List, Optional, Tuple, Iterable


GROUND_TRUTH_LEN = 2936


def _is_tar(path: str) -> bool:
    try:
        return tarfile.is_tarfile(path)
    except Exception:
        return False


def _read_file_from_tar(tf: tarfile.TarFile, m: tarfile.TarInfo, limit: Optional[int] = None) -> bytes:
    f = tf.extractfile(m)
    if f is None:
        return b""
    try:
        if limit is None:
            return f.read()
        return f.read(limit)
    finally:
        try:
            f.close()
        except Exception:
            pass


def _iter_dir_files(root: str) -> Iterable[Tuple[str, int]]:
    for base, _, files in os.walk(root):
        for fn in files:
            p = os.path.join(base, fn)
            try:
                st = os.stat(p)
            except Exception:
                continue
            if not os.path.isfile(p):
                continue
            yield p, st.st_size


def _read_file_from_dir(path: str, limit: Optional[int] = None) -> bytes:
    try:
        with open(path, "rb") as f:
            if limit is None:
                return f.read()
            return f.read(limit)
    except Exception:
        return b""


def _lower_ascii(b: bytes) -> str:
    try:
        return b.decode("utf-8", "ignore").lower()
    except Exception:
        return ""


def _detect_format_by_magic(head: bytes) -> Optional[str]:
    if len(head) >= 8 and head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if len(head) >= 3 and head[0:3] == b"GIF":
        return "gif"
    if len(head) >= 2 and head[:2] == b"BM":
        return "bmp"
    if len(head) >= 4 and (head[:4] == b"II*\x00" or head[:4] == b"MM\x00*"):
        return "tiff"
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if len(head) >= 12 and head[4:8] == b"ftyp":
        brands = head[8:12]
        if brands in (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"avif", b"avis"):
            return "heif"
        return "isobmff"
    if len(head) >= 3 and head[:3] == b"\xFF\xD8\xFF":
        return "jpeg"
    if len(head) >= 12 and head[:12] == b"\x00\x00\x00\x0cjP  \r\n\x87\n":
        return "jp2"
    if len(head) >= 4 and head[:4] == b"qoif":
        return "qoi"
    if len(head) >= 4 and head[:4] == b"8BPS":
        return "psd"
    if len(head) >= 4 and head[:4] == b"\x00\x00\x01\x00":
        return "ico"
    if len(head) >= 2 and head[:1] == b"P" and head[1:2] in b"123456":
        return "pnm"
    return None


def _fmt_keywords() -> Dict[str, List[str]]:
    return {
        "heif": ["heif", "heic", "avif", "ispe", "iprp", "ipco", "irot", "pixi", "iinf", "iloc", "meta", "isobmff"],
        "png": ["png", "ihdr", "libpng", "spng"],
        "jpeg": ["jpeg", "jpg", "jpeglib", "libjpeg", "turbojpeg", "tjdecompress"],
        "tiff": ["tiff", "tif", "tiffio", "libtiff"],
        "webp": ["webp", "vp8", "vp8l", "vp8x"],
        "gif": ["gif"],
        "bmp": ["bmp", "bitmap"],
        "jp2": ["jpeg2000", "jpeg 2000", "jp2", "j2k", "openjpeg"],
        "qoi": ["qoi", "qoif"],
        "pnm": ["pnm", "ppm", "pgm", "pbm", "netpbm"],
        "ico": ["ico", "icon"],
        "psd": ["psd", "photoshop"],
    }


def _compute_format_preferences(hints_texts: List[str], hints_paths: List[str]) -> List[str]:
    scores: Dict[str, int] = {}
    kw = _fmt_keywords()

    def add_text(t: str, weight: int) -> None:
        lt = t.lower()
        for fmt, kws in kw.items():
            s = 0
            for k in kws:
                if k in lt:
                    s += 10 + min(len(k), 10)
            if s:
                scores[fmt] = scores.get(fmt, 0) + s * weight

    for t in hints_texts:
        add_text(t, 3)
    for p in hints_paths:
        add_text(p, 1)

    default_order = ["heif", "png", "jpeg", "tiff", "webp", "gif", "bmp", "jp2", "qoi", "pnm", "ico", "psd", "isobmff"]
    for fmt in default_order:
        scores.setdefault(fmt, 0)

    order = sorted(scores.keys(), key=lambda f: (-scores[f], default_order.index(f) if f in default_order else 999))
    return order


def _crc32(data: bytes) -> int:
    return binascii.crc32(data) & 0xFFFFFFFF


def _patch_png_zero_dim(data: bytes, zero_width: bool = True, zero_height: bool = False) -> Optional[bytes]:
    if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    out = bytearray(data)
    i = 8
    while i + 12 <= len(out):
        try:
            length = struct.unpack(">I", out[i:i+4])[0]
        except Exception:
            return None
        ctype = bytes(out[i+4:i+8])
        data_off = i + 8
        crc_off = data_off + length
        nxt = crc_off + 4
        if crc_off < 0 or nxt > len(out):
            return None
        if ctype == b"IHDR" and length == 13 and data_off + 13 <= len(out):
            w = struct.unpack(">I", out[data_off:data_off+4])[0]
            h = struct.unpack(">I", out[data_off+4:data_off+8])[0]
            if zero_width:
                w = 0
            if zero_height:
                h = 0
            out[data_off:data_off+4] = struct.pack(">I", w)
            out[data_off+4:data_off+8] = struct.pack(">I", h)
            crc = _crc32(out[i+4:i+8+length])
            out[crc_off:crc_off+4] = struct.pack(">I", crc)
            return bytes(out)
        i = nxt
    return None


def _patch_gif_zero_dim(data: bytes, zero_width: bool = True, zero_height: bool = False) -> Optional[bytes]:
    if len(data) < 10 or data[:3] != b"GIF":
        return None
    out = bytearray(data)
    if zero_width:
        out[6:8] = b"\x00\x00"
    if zero_height:
        out[8:10] = b"\x00\x00"
    return bytes(out)


def _patch_bmp_zero_dim(data: bytes, zero_width: bool = True, zero_height: bool = False) -> Optional[bytes]:
    if len(data) < 26 or data[:2] != b"BM":
        return None
    out = bytearray(data)
    # BITMAPINFOHEADER width/height at offsets 18/22 (little endian), may vary but common
    if len(out) >= 26:
        if zero_width:
            out[18:22] = struct.pack("<I", 0)
        if zero_height:
            out[22:26] = struct.pack("<I", 0)
        return bytes(out)
    return None


def _patch_jpeg_zero_dim(data: bytes, zero_width: bool = True, zero_height: bool = False) -> Optional[bytes]:
    if len(data) < 4 or data[:2] != b"\xFF\xD8":
        return None
    out = bytearray(data)
    i = 2
    while i + 4 <= len(out):
        if out[i] != 0xFF:
            i += 1
            continue
        while i < len(out) and out[i] == 0xFF:
            i += 1
        if i >= len(out):
            break
        marker = out[i]
        i += 1
        if marker in (0xD9, 0xDA):  # EOI, SOS: stop at SOS
            break
        if i + 2 > len(out):
            break
        seglen = struct.unpack(">H", out[i:i+2])[0]
        if seglen < 2 or i + seglen > len(out):
            break
        seg_start = i + 2
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):  # SOF markers
            # segment: [precision][height][width]...
            if seglen >= 8 and seg_start + 5 <= len(out):
                h_off = seg_start + 1
                w_off = seg_start + 3
                if zero_height:
                    out[h_off:h_off+2] = b"\x00\x00"
                if zero_width:
                    out[w_off:w_off+2] = b"\x00\x00"
                return bytes(out)
        i = i + seglen
    return None


def _patch_tiff_zero_dim(data: bytes, zero_width: bool = True, zero_height: bool = False) -> Optional[bytes]:
    if len(data) < 8:
        return None
    endian = data[:2]
    if endian == b"II":
        le = True
    elif endian == b"MM":
        le = False
    else:
        return None
    if le:
        u16 = lambda b: struct.unpack("<H", b)[0]
        u32 = lambda b: struct.unpack("<I", b)[0]
        p16 = lambda v: struct.pack("<H", v)
        p32 = lambda v: struct.pack("<I", v)
    else:
        u16 = lambda b: struct.unpack(">H", b)[0]
        u32 = lambda b: struct.unpack(">I", b)[0]
        p16 = lambda v: struct.pack(">H", v)
        p32 = lambda v: struct.pack(">I", v)

    if u16(data[2:4]) != 42:
        return None
    ifd_off = u32(data[4:8])
    if ifd_off == 0 or ifd_off + 2 > len(data):
        return None
    out = bytearray(data)
    try:
        n = u16(out[ifd_off:ifd_off+2])
    except Exception:
        return None
    base = ifd_off + 2
    for idx in range(n):
        ent = base + idx * 12
        if ent + 12 > len(out):
            break
        tag = u16(out[ent:ent+2])
        typ = u16(out[ent+2:ent+4])
        cnt = u32(out[ent+4:ent+8])
        valoff = ent + 8

        def patch_value_zero():
            if typ == 3 and cnt == 1:
                # SHORT in first 2 bytes of value field
                if le:
                    out[valoff:valoff+2] = p16(0)
                else:
                    out[valoff:valoff+2] = p16(0)
                return True
            if typ == 4 and cnt == 1:
                out[valoff:valoff+4] = p32(0)
                return True
            # handle offset storage for SHORT/LONG arrays with first element
            if typ == 3 and cnt >= 1:
                off = u32(out[valoff:valoff+4])
                if off + 2 <= len(out):
                    out[off:off+2] = p16(0)
                    return True
            if typ == 4 and cnt >= 1:
                off = u32(out[valoff:valoff+4])
                if off + 4 <= len(out):
                    out[off:off+4] = p32(0)
                    return True
            return False

        if tag == 256 and zero_width:
            if patch_value_zero():
                return bytes(out)
        if tag == 257 and zero_height:
            if patch_value_zero():
                return bytes(out)
    if zero_width and not zero_height:
        # If only width requested and not found, try height tag as well; some libs use both
        return bytes(out)
    return None


def _patch_webp_zero_dim(data: bytes, zero_width: bool = True, zero_height: bool = False) -> Optional[bytes]:
    if len(data) < 16 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    out = bytearray(data)
    i = 12
    # iterate chunks: fourcc (4) + size (4 le) + payload + pad
    while i + 8 <= len(out):
        fourcc = bytes(out[i:i+4])
        size = struct.unpack("<I", out[i+4:i+8])[0]
        payload = i + 8
        end = payload + size
        if end > len(out):
            break
        if fourcc == b"VP8 " and size >= 10:
            # Key frame: 3 bytes frame tag + 3 start code + 2 width + 2 height
            w_off = payload + 6
            h_off = payload + 8
            if h_off + 2 <= len(out):
                if zero_width:
                    out[w_off:w_off+2] = b"\x00\x00"
                if zero_height:
                    out[h_off:h_off+2] = b"\x00\x00"
                return bytes(out)
        # pad to even
        i = end + (end & 1)
    return None


def _patch_heif_ispe_zero_dim(data: bytes, zero_width: bool = True, zero_height: bool = False) -> Optional[bytes]:
    # scan for ispe box occurrences and patch first plausible one
    if len(data) < 16:
        return None
    out = bytearray(data)
    n = len(out)
    i = 0
    while i + 8 <= n:
        if i + 8 <= n and out[i+4:i+8] == b"ispe":
            try:
                size = struct.unpack(">I", out[i:i+4])[0]
            except Exception:
                size = 0
            if size == 1 and i + 16 <= n:
                # largesize
                try:
                    size = struct.unpack(">Q", out[i+8:i+16])[0]
                    hdr = 16
                except Exception:
                    size = 0
                    hdr = 16
            else:
                hdr = 8
            if size < hdr + 12 or size > 10_000_000 or i + size > n:
                i += 1
                continue
            data_off = i + hdr
            # Full box: version/flags 4, then width 4, height 4
            if data_off + 12 <= n:
                if zero_width:
                    out[data_off+4:data_off+8] = b"\x00\x00\x00\x00"
                if zero_height:
                    out[data_off+8:data_off+12] = b"\x00\x00\x00\x00"
                return bytes(out)
        i += 1
    return None


def _patch_qoi_zero_dim(data: bytes, zero_width: bool = True, zero_height: bool = False) -> Optional[bytes]:
    if len(data) < 14 or data[:4] != b"qoif":
        return None
    out = bytearray(data)
    if zero_width:
        out[4:8] = b"\x00\x00\x00\x00"
    if zero_height:
        out[8:12] = b"\x00\x00\x00\x00"
    return bytes(out)


def _patch_pnm_zero_dim(data: bytes, zero_width: bool = True, zero_height: bool = False) -> Optional[bytes]:
    # Not reliable to patch; regenerate a minimal header with 0 dimension but keep tail to increase mismatch
    if len(data) < 2 or data[:1] != b"P" or data[1:2] not in b"123456":
        return None
    # Attempt to keep original tail after first three lines; best-effort
    tail = data
    header = b"P6\n0 1\n255\n" if zero_width and not zero_height else b"P6\n1 0\n255\n" if (zero_height and not zero_width) else b"P6\n0 0\n255\n"
    # Keep some payload to create mismatch
    return header + tail[:2048]


def _make_minimal_png_mismatch() -> bytes:
    # PNG with IHDR width=0 height=1 but IDAT expands to much more than expected.
    sig = b"\x89PNG\r\n\x1a\n"
    width = 0
    height = 1
    bit_depth = 8
    color_type = 2  # RGB
    comp = 0
    filt = 0
    inter = 0
    ihdr_data = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, comp, filt, inter)
    ihdr = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data
    ihdr += struct.pack(">I", _crc32(b"IHDR" + ihdr_data))

    # Compressed payload: large amount of bytes (filter+pixel bytes etc), regardless of declared width/height.
    raw = b"\x00" + (b"\x00" * 8192)  # way more than expected for width=0,height=1
    try:
        import zlib
        compd = zlib.compress(raw, 9)
    except Exception:
        compd = raw
    idat = struct.pack(">I", len(compd)) + b"IDAT" + compd
    idat += struct.pack(">I", _crc32(b"IDAT" + compd))

    iend = struct.pack(">I", 0) + b"IEND" + b""
    iend += struct.pack(">I", _crc32(b"IEND"))

    return sig + ihdr + idat + iend


def _patch_any_zero_dim(fmt: str, data: bytes) -> Optional[bytes]:
    if fmt == "png":
        return _patch_png_zero_dim(data, zero_width=True, zero_height=False)
    if fmt == "gif":
        return _patch_gif_zero_dim(data, zero_width=True, zero_height=False)
    if fmt == "bmp":
        return _patch_bmp_zero_dim(data, zero_width=True, zero_height=False)
    if fmt == "jpeg":
        return _patch_jpeg_zero_dim(data, zero_width=True, zero_height=False)
    if fmt == "tiff":
        return _patch_tiff_zero_dim(data, zero_width=True, zero_height=False)
    if fmt == "webp":
        return _patch_webp_zero_dim(data, zero_width=True, zero_height=False)
    if fmt == "heif":
        return _patch_heif_ispe_zero_dim(data, zero_width=True, zero_height=False)
    if fmt == "qoi":
        return _patch_qoi_zero_dim(data, zero_width=True, zero_height=False)
    if fmt == "pnm":
        return _patch_pnm_zero_dim(data, zero_width=True, zero_height=False)
    return None


def _likely_text_path(path: str) -> bool:
    p = path.lower()
    if p.endswith((".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx", ".inc")):
        return True
    if p.endswith((".cmake", "cmakelists.txt", "meson.build", "meson_options.txt", ".gn", ".gyp", ".gypi", ".mk", "makefile", "configure.ac", "configure.in", ".am", ".in")):
        return True
    if p.endswith((".txt", ".md", ".rst", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg")):
        return True
    return False


def _collect_hints_and_candidates_from_tar(src_path: str) -> Tuple[List[str], List[str], List[Tuple[str, tarfile.TarInfo, int, str]]]:
    # returns: hints_texts, hints_paths, candidates[(fmt, member, size, name)]
    hints_texts: List[str] = []
    hints_paths: List[str] = []
    candidates: List[Tuple[str, tarfile.TarInfo, int, str]] = []

    with tarfile.open(src_path, "r:*") as tf:
        members = tf.getmembers()
        for m in members:
            if not m.isfile():
                continue
            name = m.name
            size = int(getattr(m, "size", 0) or 0)
            lname = name.lower()
            hints_paths.append(lname)

            # Text hints: fuzzers/build files
            if size > 0 and size <= 1_200_000 and _likely_text_path(lname):
                b = _read_file_from_tar(tf, m, limit=256_000)
                if b:
                    s = _lower_ascii(b)
                    if "llvmfuzzertestoneinput" in s or "fuzz" in lname or "fuzzer" in lname or "oss-fuzz" in s:
                        hints_texts.append(s)
                    else:
                        # still useful occasionally, but keep small
                        if any(k in s for k in ("heif", "avif", "png", "jpeg", "tiff", "webp", "gif", "bmp", "jp2", "qoi")):
                            hints_texts.append(s[:20000])

            # Binary candidates: recognize by magic
            if 32 <= size <= 1_500_000:
                head = _read_file_from_tar(tf, m, limit=64)
                if not head:
                    continue
                fmt = _detect_format_by_magic(head)
                if fmt in ("png", "gif", "bmp", "tiff", "webp", "heif", "jpeg", "jp2", "qoi", "pnm", "ico", "psd"):
                    candidates.append((fmt, m, size, name))

    return hints_texts, hints_paths, candidates


def _collect_hints_and_candidates_from_dir(src_dir: str) -> Tuple[List[str], List[str], List[Tuple[str, str, int, str]]]:
    hints_texts: List[str] = []
    hints_paths: List[str] = []
    candidates: List[Tuple[str, str, int, str]] = []

    for p, size in _iter_dir_files(src_dir):
        rel = os.path.relpath(p, src_dir)
        lrel = rel.lower().replace("\\", "/")
        hints_paths.append(lrel)
        if size > 0 and size <= 1_200_000 and _likely_text_path(lrel):
            b = _read_file_from_dir(p, limit=256_000)
            if b:
                s = _lower_ascii(b)
                if "llvmfuzzertestoneinput" in s or "fuzz" in lrel or "fuzzer" in lrel or "oss-fuzz" in s:
                    hints_texts.append(s)
                else:
                    if any(k in s for k in ("heif", "avif", "png", "jpeg", "tiff", "webp", "gif", "bmp", "jp2", "qoi")):
                        hints_texts.append(s[:20000])

        if 32 <= size <= 1_500_000:
            head = _read_file_from_dir(p, limit=64)
            if not head:
                continue
            fmt = _detect_format_by_magic(head)
            if fmt in ("png", "gif", "bmp", "tiff", "webp", "heif", "jpeg", "jp2", "qoi", "pnm", "ico", "psd"):
                candidates.append((fmt, p, size, rel))
    return hints_texts, hints_paths, candidates


def _choose_best_candidate(fmt_order: List[str], candidates: List[Tuple[str, object, int, str]]) -> Optional[Tuple[str, object, int, str]]:
    if not candidates:
        return None

    fmt_rank = {f: i for i, f in enumerate(fmt_order)}

    def path_bonus(path: str) -> int:
        lp = path.lower()
        bonus = 0
        for k in ("fuzz", "corpus", "test", "tests", "data", "sample", "samples", "example", "examples"):
            if k in lp:
                bonus -= 100
        return bonus

    # Prefer format earlier in fmt_order, and size close to ground truth (and not huge)
    best = None
    best_key = None
    for fmt, ref, size, name in candidates:
        r = fmt_rank.get(fmt, 999)
        # penalize unknown/less likely isobmff unless specifically preferred
        if fmt == "isobmff":
            r += 50
        # size heuristic: prefer around ground truth, but not too small (<256) and not too large
        size_pen = abs(size - GROUND_TRUTH_LEN)
        if size < 256:
            size_pen += 5000
        if size > 200_000:
            size_pen += (size - 200_000) // 10
        key = (r, size_pen, size, path_bonus(name), name)
        if best_key is None or key < best_key:
            best_key = key
            best = (fmt, ref, size, name)
    return best


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Collect hints and sample candidates
        if os.path.isdir(src_path):
            hints_texts, hints_paths, candidates0 = _collect_hints_and_candidates_from_dir(src_path)
            candidates: List[Tuple[str, object, int, str]] = [(fmt, p, sz, rel) for (fmt, p, sz, rel) in candidates0]
            fmt_order = _compute_format_preferences(hints_texts, hints_paths + [os.path.basename(src_path).lower()])
            best = _choose_best_candidate(fmt_order, candidates)
            if best is not None:
                fmt, p, _, _ = best
                data = _read_file_from_dir(str(p), limit=None)
                patched = _patch_any_zero_dim(fmt, data)
                if patched is not None:
                    return patched
            # Fallback: generate minimal mismatch PNG
            return _make_minimal_png_mismatch()

        if _is_tar(src_path):
            hints_texts, hints_paths, candidates0 = _collect_hints_and_candidates_from_tar(src_path)
            candidates: List[Tuple[str, object, int, str]] = [(fmt, m, sz, name) for (fmt, m, sz, name) in candidates0]
            fmt_order = _compute_format_preferences(hints_texts, hints_paths + [os.path.basename(src_path).lower()])
            best = _choose_best_candidate(fmt_order, candidates)
            if best is not None:
                fmt, m, _, _ = best
                with tarfile.open(src_path, "r:*") as tf:
                    data = _read_file_from_tar(tf, m, limit=None)
                patched = _patch_any_zero_dim(fmt, data)
                if patched is not None:
                    return patched
            return _make_minimal_png_mismatch()

        # Unknown input; fallback
        return _make_minimal_png_mismatch()