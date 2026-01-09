import struct
from typing import Optional


def _u16be(x: int) -> bytes:
    return struct.pack(">H", x & 0xFFFF)


def _segment(marker: int, payload: bytes) -> bytes:
    return b"\xFF" + bytes([marker & 0xFF]) + _u16be(len(payload) + 2) + payload


def _build_minimal_jpeg(width: int = 8, height: int = 8) -> bytes:
    # Minimal baseline JPEG (SOF0) with 3 components (YCbCr 4:4:4)
    # Uses a single quant table and a single DC/AC Huffman table for all components.
    # Encodes one MCU (since 8x8, 4:4:4) with all-zero coefficients -> constant gray.

    soi = b"\xFF\xD8"
    eoi = b"\xFF\xD9"

    # APP0 JFIF
    app0_payload = b"JFIF\x00" + b"\x01\x01" + b"\x00" + b"\x00\x01" + b"\x00\x01" + b"\x00\x00"
    app0 = _segment(0xE0, app0_payload)

    # DQT: 1 table, 8-bit precision, ID=0, all ones
    dqt_payload = b"\x00" + (b"\x01" * 64)
    dqt = _segment(0xDB, dqt_payload)

    # SOF0
    sof0_payload = bytes([8]) + _u16be(height) + _u16be(width) + bytes([
        3,          # Nf
        1, 0x11, 0, # C1: Y, H=1 V=1, QT=0
        2, 0x11, 0, # C2: Cb, H=1 V=1, QT=0
        3, 0x11, 0  # C3: Cr, H=1 V=1, QT=0
    ])
    sof0 = _segment(0xC0, sof0_payload)

    # DHT: define DC/AC table 0 only, complete with 2 symbols at length 1
    bits_len1_two = bytes([2] + [0] * 15)  # 16 bytes: number of codes for lengths 1..16
    dc_vals = bytes([0, 1])                # categories 0 and 1
    ac_vals = bytes([0x00, 0x01])          # EOB and (0 run, size 1)
    dht_payload = (
        bytes([0x00]) + bits_len1_two + dc_vals +  # DC table 0
        bytes([0x10]) + bits_len1_two + ac_vals    # AC table 0
    )
    dht = _segment(0xC4, dht_payload)

    # SOS: 3 components, all use Huffman table 0 for DC and AC
    sos_payload = bytes([
        3,        # Ns
        1, 0x00,  # Y: Td=0 Ta=0
        2, 0x00,  # Cb: Td=0 Ta=0
        3, 0x00,  # Cr: Td=0 Ta=0
        0,        # Ss
        63,       # Se
        0         # Ah/Al
    ])
    sos = _segment(0xDA, sos_payload)

    # Entropy-coded data for one MCU with 3 blocks:
    # Each block: DC symbol 0 -> '0', AC EOB symbol 0x00 -> '0' => '00'
    # 3 blocks => 6 bits of 0. Pad with 1s to byte boundary => '00000011' = 0x03
    scan_data = b"\x03"

    return soi + app0 + dqt + sof0 + dht + sos + scan_data + eoi


class Solution:
    def solve(self, src_path: str) -> bytes:
        return _build_minimal_jpeg(8, 8)