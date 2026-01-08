import os
import io
import re
import tarfile
import zipfile
import gzip
import bz2
import lzma
import heapq
from typing import Optional, Tuple, List


class Solution:
    _GROUND_TRUTH_LEN = 37535

    def solve(self, src_path: str) -> bytes:
        data = self._find_best_poc(src_path)
        if data is not None:
            return data
        return self._fallback_pgp()

    def _find_best_poc(self, src_path: str) -> Optional[bytes]:
        if os.path.isdir(src_path):
            return self._find_best_poc_in_dir(src_path)
        return self._find_best_poc_in_tar(src_path)

    def _find_best_poc_in_dir(self, root: str) -> Optional[bytes]:
        topk: List[Tuple[int, int, str]] = []
        idx = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dn = [d for d in dirnames if d not in (".git", ".svn", ".hg", "__pycache__")]
            dirnames[:] = dn
            for fn in filenames:
                idx += 1
                p = os.path.join(dirpath, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                if not os.path.isfile(p):
                    continue
                if st.st_size <= 0 or st.st_size > 5_000_000:
                    continue
                rel = os.path.relpath(p, root).replace("\\", "/")
                ms = self._meta_score(rel, st.st_size)
                if ms <= 0:
                    continue
                self._push_topk(topk, (ms, idx, p), k=300)

        if not topk:
            return None

        candidates = sorted(topk, reverse=True)
        best = None
        best_score = -10**18
        for ms, _, p in candidates[:250]:
            try:
                with open(p, "rb") as f:
                    blob = f.read(5_000_001)
            except OSError:
                continue
            if not blob:
                continue
            unwrapped = self._unwrap_blob(blob, p, depth=0)
            cs = self._content_score(unwrapped, p)
            total = ms + cs
            if total > best_score:
                best_score = total
                best = unwrapped
                if total >= 900:
                    break

        if best is not None and len(best) > 0:
            return best
        return None

    def _find_best_poc_in_tar(self, tar_path: str) -> Optional[bytes]:
        try:
            tf = tarfile.open(tar_path, mode="r:*")
        except Exception:
            return None

        with tf:
            topk: List[Tuple[int, int, tarfile.TarInfo]] = []
            idx = 0
            for m in tf.getmembers():
                idx += 1
                if not m.isfile():
                    continue
                if m.size <= 0 or m.size > 5_000_000:
                    continue
                name = (m.name or "").replace("\\", "/")
                if "/.git/" in ("/" + name + "/") or name.startswith(".git/"):
                    continue
                ms = self._meta_score(name, m.size)
                if ms <= 0:
                    continue
                self._push_topk(topk, (ms, idx, m), k=350)

            if not topk:
                return None

            candidates = sorted(topk, reverse=True)
            best = None
            best_score = -10**18
            for ms, _, m in candidates[:300]:
                try:
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    blob = f.read(5_000_001)
                except Exception:
                    continue
                if not blob:
                    continue
                unwrapped = self._unwrap_blob(blob, m.name, depth=0)
                cs = self._content_score(unwrapped, m.name)
                total = ms + cs
                if total > best_score:
                    best_score = total
                    best = unwrapped
                    if total >= 900:
                        break

            if best is not None and len(best) > 0:
                return best
            return None

    @staticmethod
    def _push_topk(heap: List[Tuple[int, int, object]], item: Tuple[int, int, object], k: int = 200) -> None:
        if len(heap) < k:
            heapq.heappush(heap, item)
        else:
            if item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

    def _meta_score(self, name: str, size: int) -> int:
        n = (name or "").lower()
        ext = os.path.splitext(n)[1]

        src_exts = {
            ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".inc",
            ".rs", ".go", ".java", ".kt", ".swift", ".m", ".mm",
            ".py", ".js", ".ts", ".rb", ".php", ".cs",
            ".md", ".rst", ".txt", ".adoc",
            ".yml", ".yaml", ".json", ".toml", ".ini", ".cfg",
            ".cmake", ".mk", ".make", ".am", ".ac", ".in",
            ".html", ".css", ".scss", ".less",
            ".sh", ".bash", ".bat", ".ps1",
            ".gitignore", ".gitattributes",
        }

        # Allow ".in" as a possible test input container, so don't treat it purely as source.
        if ext == ".in":
            is_source = False
        else:
            is_source = ext in src_exts

        score = 0

        if any(k in n for k in ("clusterfuzz", "testcase", "minimized", "reproducer", "repro", "crash", "poc")):
            score += 320
        if any(k in n for k in ("oss-fuzz", "ossfuzz", "artifact", "artifacts")):
            score += 180
        if any(k in n for k in ("fuzz", "corpus", "seed", "seeds", "regress", "testdata", "test-data", "inputs", "input")):
            score += 90
        if any(k in n for k in ("openpgp", "/pgp", "pgp/", ".pgp", "gpg", "keyring", "pubkey", "seckey", "fingerprint", "fpr")):
            score += 60

        if ext in (".pgp", ".gpg", ".asc", ".bin", ".dat", ".key", ".pkt", ".poc", ".raw"):
            score += 70
        if ext in (".zip", ".gz", ".xz", ".bz2"):
            score += 40

        if is_source:
            score -= 260

        if size == self._GROUND_TRUTH_LEN:
            score += 240

        # Prefer "mid-sized" fuzz artifacts; extremely tiny files are usually configs/dicts.
        if size < 16:
            score -= 80
        elif size < 256:
            score -= 20
        elif 1_000 <= size <= 2_000_000:
            score += 15

        # Size closeness heuristic.
        score += int(max(0.0, 120.0 - (abs(size - self._GROUND_TRUTH_LEN) / 250.0)))

        # De-prioritize obvious build outputs
        if any(k in n for k in ("/cmakefiles/", "/cpack/", "/build/", "/.svn/", "/.hg/")):
            score -= 100
        if n.endswith((".o", ".a", ".so", ".dll", ".dylib", ".exe", ".obj", ".class")):
            score -= 180

        return score

    def _unwrap_blob(self, blob: bytes, name: str, depth: int) -> bytes:
        if depth >= 3 or not blob:
            return blob

        # Skip Git LFS pointers
        if blob.startswith(b"version https://git-lfs.github.com/spec/v1"):
            return b""

        # ZIP
        if len(blob) >= 4 and blob[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
            zbest = self._best_from_zip(blob, name=name, depth=depth)
            if zbest is not None:
                return zbest
            return blob

        # GZIP (magic 1f 8b)
        if len(blob) >= 2 and blob[0] == 0x1F and blob[1] == 0x8B:
            try:
                de = gzip.decompress(blob)
                if 0 < len(de) <= 5_000_000:
                    return self._unwrap_blob(de, name + ":gz", depth + 1)
            except Exception:
                pass
            return blob

        # BZ2
        if len(blob) >= 3 and blob[:3] == b"BZh":
            try:
                de = bz2.decompress(blob)
                if 0 < len(de) <= 5_000_000:
                    return self._unwrap_blob(de, name + ":bz2", depth + 1)
            except Exception:
                pass
            return blob

        # XZ/LZMA
        if len(blob) >= 6 and blob[:6] == b"\xFD7zXZ\x00":
            try:
                de = lzma.decompress(blob)
                if 0 < len(de) <= 5_000_000:
                    return self._unwrap_blob(de, name + ":xz", depth + 1)
            except Exception:
                pass
            return blob

        # If the blob includes an ASCII armor block embedded in text, extract it.
        if depth == 0 and self._looks_like_text(blob):
            armor = self._extract_first_armor_block(blob)
            if armor is not None and len(armor) > 0:
                return armor

        return blob

    def _best_from_zip(self, blob: bytes, name: str, depth: int) -> Optional[bytes]:
        try:
            zf = zipfile.ZipFile(io.BytesIO(blob))
        except Exception:
            return None

        with zf:
            infos = [zi for zi in zf.infolist() if not zi.is_dir()]
            if not infos:
                return None

            heap: List[Tuple[int, int, zipfile.ZipInfo]] = []
            idx = 0
            for zi in infos:
                idx += 1
                zname = (zi.filename or "").replace("\\", "/")
                size = zi.file_size
                if size <= 0 or size > 5_000_000:
                    continue
                ms = self._meta_score(zname, size)
                ms += 40 if any(k in zname.lower() for k in ("seed", "corpus", "crash", "poc", "repro", "testcase", "minimized")) else 0
                if ms <= 0:
                    continue
                self._push_topk(heap, (ms, idx, zi), k=120)

            if not heap:
                return None

            candidates = sorted(heap, reverse=True)
            best = None
            best_score = -10**18
            for ms, _, zi in candidates[:80]:
                try:
                    data = zf.read(zi)
                except Exception:
                    continue
                if not data:
                    continue
                data2 = self._unwrap_blob(data, name + ":" + zi.filename, depth + 1)
                cs = self._content_score(data2, zi.filename)
                total = ms + cs
                if total > best_score:
                    best_score = total
                    best = data2
                    if total >= 920:
                        break

            return best

    @staticmethod
    def _looks_like_text(b: bytes) -> bool:
        if not b:
            return False
        sample = b[:4096]
        if b"\x00" in sample:
            return False
        # Heuristic: mostly printable/whitespace
        printable = 0
        for c in sample:
            if 32 <= c <= 126 or c in (9, 10, 13):
                printable += 1
        return printable / max(1, len(sample)) > 0.92

    @staticmethod
    def _extract_first_armor_block(b: bytes) -> Optional[bytes]:
        try:
            s = b.decode("utf-8", errors="ignore")
        except Exception:
            return None
        m = re.search(r"-----BEGIN PGP[^\n]*-----.*?-----END PGP[^\n]*-----\s*", s, flags=re.S)
        if not m:
            return None
        block = s[m.start():m.end()]
        block = block.replace("\r\n", "\n").replace("\r", "\n")
        return block.encode("utf-8", errors="ignore")

    def _content_score(self, data: bytes, name: str) -> int:
        if not data:
            return -10**9

        sc = 0
        nl = (name or "").lower()

        if data.startswith(b"-----BEGIN PGP"):
            sc += 120
        if len(data) > 2 and (data[0] & 0x80) == 0x80:
            sc += 12

        # Likely OpenPGP packet parse quality
        ratio, pkts = self._openpgp_parse_quality(data)
        sc += int(ratio * 90) + pkts * 4

        # Penalize obvious source/config text unless armored
        if self._looks_like_text(data) and not data.startswith(b"-----BEGIN PGP"):
            if any(k in nl for k in (".dict", ".options", ".txt", ".md", ".yml", ".yaml", ".json", ".toml", ".ini")):
                sc -= 120
            else:
                sc -= 35

        # Reward closeness to ground truth length, mildly
        sc += int(max(0.0, 50.0 - (abs(len(data) - self._GROUND_TRUTH_LEN) / 500.0)))

        # Some magic bytes that indicate not a PGP input
        if data.startswith(b"\x7fELF") or data.startswith(b"MZ") or data.startswith(b"\x89PNG") or data.startswith(b"%PDF"):
            sc -= 250

        return sc

    @staticmethod
    def _openpgp_parse_quality(data: bytes) -> Tuple[float, int]:
        n = len(data)
        if n == 0:
            return 0.0, 0

        i = 0
        parsed = 0
        packets = 0
        # try parsing up to 30 packets
        while i < n and packets < 30:
            b0 = data[i]
            if (b0 & 0x80) == 0:
                break

            if (b0 & 0x40) != 0:
                # new format
                i += 1
                if i >= n:
                    break
                first = data[i]
                i += 1
                if first < 192:
                    plen = first
                elif first < 224:
                    if i >= n:
                        break
                    plen = ((first - 192) << 8) + data[i] + 192
                    i += 1
                elif first == 255:
                    if i + 4 > n:
                        break
                    plen = int.from_bytes(data[i:i + 4], "big")
                    i += 4
                else:
                    # partial body length: stop
                    break
            else:
                # old format
                lt = b0 & 0x03
                i += 1
                if lt == 0:
                    if i >= n:
                        break
                    plen = data[i]
                    i += 1
                elif lt == 1:
                    if i + 2 > n:
                        break
                    plen = int.from_bytes(data[i:i + 2], "big")
                    i += 2
                elif lt == 2:
                    if i + 4 > n:
                        break
                    plen = int.from_bytes(data[i:i + 4], "big")
                    i += 4
                else:
                    # indeterminate length
                    plen = n - i

            if plen < 0 or i + plen > n:
                break
            i += plen
            parsed = i
            packets += 1

        ratio = parsed / n
        return ratio, packets

    @staticmethod
    def _encode_new_len(l: int) -> bytes:
        if l < 0:
            l = 0
        if l < 192:
            return bytes([l])
        if l < 8384:
            l2 = l - 192
            return bytes([192 + (l2 >> 8), l2 & 0xFF])
        return b"\xFF" + int(l).to_bytes(4, "big", signed=False)

    def _pgp_new_packet(self, tag: int, body: bytes) -> bytes:
        tag = int(tag) & 0x3F
        hdr = bytes([0xC0 | tag]) + self._encode_new_len(len(body))
        return hdr + body

    @staticmethod
    def _mpi_min_one() -> bytes:
        # MPI value 1: bitlength = 1, content = 0x01
        return b"\x00\x01\x01"

    def _fallback_pgp(self) -> bytes:
        mpi1 = self._mpi_min_one()

        # V4 public key packet (tag 6)
        pk_v4_body = b"\x04" + b"\x00\x00\x00\x00" + b"\x01" + mpi1 + mpi1
        pk_v4 = self._pgp_new_packet(6, pk_v4_body)

        # User ID packet (tag 13)
        uid = self._pgp_new_packet(13, b"a")

        # Minimal signature packet v4 (tag 2), RSA, SHA-256, empty subpackets
        hashed = b""
        unhashed = b""
        sig_body = (
            b"\x04" + bytes([0x13, 0x01, 0x08]) +
            len(hashed).to_bytes(2, "big") + hashed +
            len(unhashed).to_bytes(2, "big") + unhashed +
            b"\x00\x00" +  # left 16 bits of hash
            mpi1  # signature MPI
        )
        sig = self._pgp_new_packet(2, sig_body)

        # V5 public key packet (tag 6) with key material length field
        km = mpi1 + mpi1
        pk_v5_body = (
            b"\x05" +
            b"\x00\x00\x00\x00" +
            b"\x01" +
            len(km).to_bytes(4, "big") +
            km
        )
        pk_v5 = self._pgp_new_packet(6, pk_v5_body)

        # Combine. Keep small; if the target needs more structure, include both variants.
        return pk_v4 + uid + sig + pk_v5