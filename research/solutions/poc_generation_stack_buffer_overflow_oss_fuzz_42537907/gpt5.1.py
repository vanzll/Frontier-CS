import os


class Solution:
    def _hevc_nal_header(self, nal_type: int, nuh_layer_id: int = 0, temporal_id: int = 0) -> bytes:
        fzb = 0
        tid_plus1 = (temporal_id & 0x7) + 1
        byte0 = (fzb << 7) | ((nal_type & 0x3F) << 1) | ((nuh_layer_id >> 5) & 0x01)
        byte1 = ((nuh_layer_id & 0x1F) << 3) | (tid_plus1 & 0x07)
        return bytes([byte0, byte1])

    def _build_hevc_like(self, total_size: int) -> bytes:
        nal_types = [32, 33, 34, 19, 1]
        payload_len = 256
        data = bytearray()
        idx = 0
        while len(data) < total_size:
            nal_type = nal_types[idx % len(nal_types)]
            idx += 1
            data += b'\x00\x00\x00\x01'
            data += self._hevc_nal_header(nal_type)
            data += b'\xff' * payload_len
        return bytes(data[:total_size])

    def solve(self, src_path: str) -> bytes:
        return self._build_hevc_like(1445)