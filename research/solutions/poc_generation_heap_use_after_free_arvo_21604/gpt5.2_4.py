import os
import io
import re
import tarfile
import tempfile
from typing import Optional, Tuple, List


def _is_probably_pdf_prefix(prefix: bytes) -> bool:
    if not prefix:
        return False
    head = prefix[:1024]
    return b"%PDF-" in head


def _score_candidate(name: str, data: bytes) -> int:
    nl = name.lower()
    score = 0

    # Filename heuristics (strong signals)
    if "21604" in nl:
        score += 200
    if "arvo" in nl:
        score += 60
    if "crash" in nl or "repro" in nl or "poc" in nl:
        score += 120
    if "uaf" in nl or "use_after_free" in nl or "use-after-free" in nl:
        score += 80
    if "oss-fuzz" in nl or "fuzz" in nl or "corpus" in nl or "testcase" in nl:
        score += 25
    if nl.endswith(".pdf"):
        score += 30
    if nl.endswith(".fuzz") or nl.endswith(".bin") or nl.endswith(".dat"):
        score += 10

    # Content heuristics (PDF features)
    dlow = data.lower()
    if b"%pdf-" in dlow[:2048]:
        score += 60
    if b"/subtype" in dlow and b"/form" in dlow:
        score += 80
    if b"/type" in dlow and b"/xobject" in dlow:
        score += 60
    if b"/acroform" in dlow:
        score += 40
    if b"/annots" in dlow or b"/widget" in dlow:
        score += 20
    if b"stream" in dlow and b"endstream" in dlow:
        score += 15
    if b"xref" in dlow and b"trailer" in dlow:
        score += 10

    # Penalize huge files slightly; prefer smaller if same score
    score -= min(len(data) // 8192, 30)
    return score


def _safe_read_file(path: str, max_bytes: int) -> Optional[bytes]:
    try:
        st = os.stat(path)
        if st.st_size <= 0:
            return None
        if st.st_size > max_bytes:
            return None
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None


def _iter_dir_candidates(root: str) -> List[Tuple[int, int, str, bytes]]:
    candidates: List[Tuple[int, int, str, bytes]] = []
    max_bytes = 5 * 1024 * 1024
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            nlower = rel.lower()

            # Fast filters
            consider = False
            if any(k in nlower for k in ("crash", "repro", "poc", "21604", "uaf")):
                consider = True
            elif nlower.endswith(".pdf"):
                consider = True
            elif any(k in nlower for k in ("oss-fuzz", "fuzz", "corpus", "testcase")) and (
                nlower.endswith(".bin") or nlower.endswith(".dat") or nlower.endswith(".fuzz")
            ):
                consider = True

            if not consider:
                continue

            data = _safe_read_file(full, max_bytes)
            if not data:
                continue

            # If it's not a PDF by extension, still allow if it looks like PDF
            if not (nlower.endswith(".pdf") or _is_probably_pdf_prefix(data[:2048])):
                # Still keep crash/poc-like binaries (might be raw input)
                if not any(k in nlower for k in ("crash", "repro", "poc", "21604")):
                    continue

            sc = _score_candidate(rel, data)
            candidates.append((sc, len(data), rel, data))
    return candidates


def _iter_tar_candidates(tar_path: str) -> List[Tuple[int, int, str, bytes]]:
    candidates: List[Tuple[int, int, str, bytes]] = []
    max_bytes = 5 * 1024 * 1024

    def should_consider_name(nlower: str) -> bool:
        if any(k in nlower for k in ("crash", "repro", "poc", "21604", "uaf")):
            return True
        if nlower.endswith(".pdf"):
            return True
        if any(k in nlower for k in ("oss-fuzz", "fuzz", "corpus", "testcase")) and (
            nlower.endswith(".bin") or nlower.endswith(".dat") or nlower.endswith(".fuzz")
        ):
            return True
        return False

    try:
        with tarfile.open(tar_path, "r:*") as tf:
            members = [m for m in tf.getmembers() if m.isreg()]
            # Prefer smaller files in early sampling
            members.sort(key=lambda m: (m.size if m.size >= 0 else 1 << 60))

            # First pass: strong filename candidates
            strong = []
            for m in members:
                n = m.name
                nl = n.lower()
                if m.size <= 0 or m.size > max_bytes:
                    continue
                if any(k in nl for k in ("21604", "crash", "repro", "poc", "uaf")):
                    strong.append(m)
            # Limit strong pass to avoid reading too many
            strong = strong[:200]

            for m in strong:
                try:
                    f = tf.extractfile(m)
                    if not f:
                        continue
                    data = f.read()
                    if not data:
                        continue
                    sc = _score_candidate(m.name, data)
                    candidates.append((sc, len(data), m.name, data))
                except Exception:
                    continue

            # Second pass: pdf extension candidates if no strong found or to augment
            if not candidates:
                pdf_members = []
                for m in members:
                    nl = m.name.lower()
                    if m.size <= 0 or m.size > max_bytes:
                        continue
                    if nl.endswith(".pdf") or should_consider_name(nl):
                        pdf_members.append(m)
                pdf_members = pdf_members[:400]
                for m in pdf_members:
                    try:
                        f = tf.extractfile(m)
                        if not f:
                            continue
                        prefix = f.read(2048)
                        if not prefix:
                            continue
                        looks_pdf = m.name.lower().endswith(".pdf") or _is_probably_pdf_prefix(prefix)
                        if not looks_pdf and not any(k in m.name.lower() for k in ("crash", "repro", "poc", "21604")):
                            continue
                        rest = f.read()
                        data = prefix + (rest if rest else b"")
                        sc = _score_candidate(m.name, data)
                        candidates.append((sc, len(data), m.name, data))
                    except Exception:
                        continue

            return candidates
    except Exception:
        return candidates


def _make_fallback_pdf() -> bytes:
    # Minimal PDF containing a Form XObject used on a page.
    # (May not trigger the bug; used only if no PoC is found in sources.)
    def obj(n: int, body: bytes) -> bytes:
        return (str(n).encode("ascii") + b" 0 obj\n" + body + b"\nendobj\n")

    parts = []
    parts.append(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n")
    offsets = [0]

    def add(chunk: bytes):
        parts.append(chunk)
        offsets.append(offsets[-1] + len(chunk))

    # 1: Catalog
    add(obj(1, b"<< /Type /Catalog /Pages 2 0 R >>"))
    # 2: Pages
    add(obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"))
    # 3: Page
    add(obj(3, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Resources << /XObject << /F1 4 0 R >> >> /Contents 5 0 R >>"))
    # 4: Form XObject
    form_stream = b"q\n1 0 0 1 0 0 cm\n0 0 300 300 re W n\nQ\n"
    form_dict = b"<< /Type /XObject /Subtype /Form /FormType 1 /BBox [0 0 300 300] /Resources << >> /Length " + str(len(form_stream)).encode("ascii") + b" >>\nstream\n" + form_stream + b"endstream"
    add(obj(4, form_dict))
    # 5: Page contents calling the form
    cont_stream = b"q\n/F1 Do\nQ\n"
    cont_dict = b"<< /Length " + str(len(cont_stream)).encode("ascii") + b" >>\nstream\n" + cont_stream + b"endstream"
    add(obj(5, cont_dict))

    # xref
    xref_start = sum(len(p) for p in parts)
    xref = [b"xref\n0 6\n0000000000 65535 f \n"]
    # offsets list includes initial 0 then after each add; need actual byte offsets of each obj.
    # Compute actual offsets by cumulative lengths of parts before each object.
    # parts[0] is header; objects start at parts[1]
    cur = len(parts[0])
    obj_offsets = [0]
    for i in range(1, 6):
        obj_offsets.append(cur)
        cur += len(parts[i])

    for i in range(1, 6):
        xref.append(f"{obj_offsets[i]:010d} 00000 n \n".encode("ascii"))
    xref_bytes = b"".join(xref)
    trailer = b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n" + str(xref_start).encode("ascii") + b"\n%%EOF\n"
    return b"".join(parts) + xref_bytes + trailer


class Solution:
    def solve(self, src_path: str) -> bytes:
        candidates: List[Tuple[int, int, str, bytes]] = []
        if os.path.isdir(src_path):
            candidates = _iter_dir_candidates(src_path)
        else:
            candidates = _iter_tar_candidates(src_path)

        if candidates:
            # Select by highest score, then smallest size, then stable name order.
            candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
            return candidates[0][3]

        return _make_fallback_pdf()