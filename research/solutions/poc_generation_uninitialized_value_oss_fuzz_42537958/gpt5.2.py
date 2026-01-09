import os
import tarfile
from typing import Dict, Tuple


def _be16(n: int) -> bytes:
    return bytes(((n >> 8) & 0xFF, n & 0xFF))


def _build_huffman_table(bits, huffval) -> Dict[int, Tuple[int, int]]:
    code = 0
    k = 0
    table: Dict[int, Tuple[int, int]] = {}
    for i in range(1, 17):
        for _ in range(bits[i]):
            if k >= len(huffval):
                break
            sym = huffval[k]
            k += 1
            table[sym] = (code, i)
            code += 1
        code <<= 1
    return table


class _BitWriter:
    __slots__ = ("buf", "acc", "nbits")

    def __init__(self):
        self.buf = bytearray()
        self.acc = 0
        self.nbits = 0

    def write(self, code: int, length: int) -> None:
        if length <= 0:
            return
        self.acc = (self.acc << length) | (code & ((1 << length) - 1))
        self.nbits += length
        while self.nbits >= 8:
            self.nbits -= 8
            b = (self.acc >> self.nbits) & 0xFF
            self.buf.append(b)
            if b == 0xFF:
                self.buf.append(0x00)
            if self.nbits:
                self.acc &= (1 << self.nbits) - 1
            else:
                self.acc = 0

    def flush(self) -> None:
        if self.nbits:
            pad_len = 8 - self.nbits
            b = ((self.acc << pad_len) | ((1 << pad_len) - 1)) & 0xFF
            self.buf.append(b)
            if b == 0xFF:
                self.buf.append(0x00)
            self.acc = 0
            self.nbits = 0


def _make_constant_jpeg_420(width: int, height: int) -> bytes:
    # Standard quantization tables (zigzag order)
    q_luma = bytes([
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99
    ])
    q_chroma = bytes([
        17, 18, 24, 47, 99, 99, 99, 99,
        18, 21, 26, 66, 99, 99, 99, 99,
        24, 26, 56, 99, 99, 99, 99, 99,
        47, 66, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99
    ])

    # Standard Huffman tables (Annex K)
    bits_dc_luma = [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    val_dc_luma = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    bits_dc_chroma = [0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    val_dc_chroma = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    bits_ac_luma = [0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D]
    val_ac_luma = [
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
        0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
        0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
        0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
        0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
        0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
        0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
        0xF9, 0xFA
    ]

    bits_ac_chroma = [0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77]
    val_ac_chroma = [
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
        0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
        0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
        0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
        0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
        0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
        0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
        0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
        0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
        0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
        0xF9, 0xFA
    ]

    htdc0 = _build_huffman_table(bits_dc_luma, val_dc_luma)
    htac0 = _build_huffman_table(bits_ac_luma, val_ac_luma)
    htdc1 = _build_huffman_table(bits_dc_chroma, val_dc_chroma)
    htac1 = _build_huffman_table(bits_ac_chroma, val_ac_chroma)

    dc0_code, dc0_len = htdc0[0]
    eob0_code, eob0_len = htac0[0x00]
    dc1_code, dc1_len = htdc1[0]
    eob1_code, eob1_len = htac1[0x00]

    out = bytearray()
    out += b"\xFF\xD8"  # SOI

    # APP0 JFIF
    app0 = bytearray()
    app0 += b"JFIF\x00"
    app0 += b"\x01\x01"  # version 1.01
    app0 += b"\x00"      # units
    app0 += b"\x00\x01\x00\x01"  # density 1x1
    app0 += b"\x00\x00"  # thumbnail
    out += b"\xFF\xE0" + _be16(2 + len(app0)) + app0

    # DQT with 2 tables
    dqt = bytearray()
    dqt += bytes([0x00]) + q_luma
    dqt += bytes([0x01]) + q_chroma
    out += b"\xFF\xDB" + _be16(2 + len(dqt)) + dqt

    # SOF0
    sof0 = bytearray()
    sof0 += b"\x08"  # precision
    sof0 += _be16(height)
    sof0 += _be16(width)
    sof0 += b"\x03"  # components
    sof0 += b"\x01" + bytes([0x22]) + b"\x00"  # Y: H2 V2, QT0
    sof0 += b"\x02" + bytes([0x11]) + b"\x01"  # Cb: H1 V1, QT1
    sof0 += b"\x03" + bytes([0x11]) + b"\x01"  # Cr: H1 V1, QT1
    out += b"\xFF\xC0" + _be16(2 + len(sof0)) + sof0

    # DHT with 4 tables
    dht = bytearray()
    dht += bytes([0x00]) + bytes(bits_dc_luma[1:17]) + bytes(val_dc_luma)
    dht += bytes([0x10]) + bytes(bits_ac_luma[1:17]) + bytes(val_ac_luma)
    dht += bytes([0x01]) + bytes(bits_dc_chroma[1:17]) + bytes(val_dc_chroma)
    dht += bytes([0x11]) + bytes(bits_ac_chroma[1:17]) + bytes(val_ac_chroma)
    out += b"\xFF\xC4" + _be16(2 + len(dht)) + dht

    # SOS
    sos = bytearray()
    sos += b"\x03"  # Ns
    sos += b"\x01\x00"  # Y uses DC0/AC0
    sos += b"\x02\x11"  # Cb uses DC1/AC1
    sos += b"\x03\x11"  # Cr uses DC1/AC1
    sos += b"\x00\x3F\x00"  # Ss Se AhAl
    out += b"\xFF\xDA" + _be16(2 + len(sos)) + sos

    # Entropy-coded data: all blocks have DC diff=0, AC all zeros => EOB
    max_h = 2
    max_v = 2
    mcu_w = 8 * max_h
    mcu_h = 8 * max_v
    mcus_x = (width + mcu_w - 1) // mcu_w
    mcus_y = (height + mcu_h - 1) // mcu_h

    bw = _BitWriter()
    for _ in range(mcus_y):
        for _ in range(mcus_x):
            # 4 Y blocks
            for _ in range(4):
                bw.write(dc0_code, dc0_len)
                bw.write(eob0_code, eob0_len)
            # Cb
            bw.write(dc1_code, dc1_len)
            bw.write(eob1_code, eob1_len)
            # Cr
            bw.write(dc1_code, dc1_len)
            bw.write(eob1_code, eob1_len)

    bw.flush()
    out += bw.buf

    out += b"\xFF\xD9"  # EOI
    return bytes(out)


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Return a small, valid JPEG (YCC 4:2:0, odd size) intended to exercise
        # compress/transform code paths that can expose MSan uninitialized reads.
        # src_path is unused but kept for API compliance.
        _ = src_path
        return _make_constant_jpeg_420(17, 17)