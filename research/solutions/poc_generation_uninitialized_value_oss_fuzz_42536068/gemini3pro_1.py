import os
import sys
import tarfile
import subprocess
import tempfile
import shutil
import random
import time
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        """
        base_dir = tempfile.mkdtemp()
        try:
            # 1. Extract source code
            try:
                with tarfile.open(src_path) as tar:
                    tar.extractall(base_dir)
            except Exception:
                # Fallback if tar extraction fails
                return b""

            # Locate source root
            src_root = base_dir
            contents = os.listdir(base_dir)
            if len(contents) == 1 and os.path.isdir(os.path.join(base_dir, contents[0])):
                src_root = os.path.join(base_dir, contents[0])

            # 2. Build configuration
            # We target MemorySanitizer (MSan) as the vulnerability is "Uninitialized Value"
            env = os.environ.copy()
            env['CC'] = 'clang'
            env['CXX'] = 'clang++'
            # Flags for MSan. 
            # -fno-omit-frame-pointer and -g are for debugging/traces
            # -O1 is usually good for fuzzing speed while keeping sanitizer effective
            san_flags = "-fsanitize=memory -fsanitize-memory-track-origins -g -O1 -fno-omit-frame-pointer"
            env['CFLAGS'] = san_flags
            env['CXXFLAGS'] = san_flags
            env['LDFLAGS'] = san_flags
            
            # Make MSan exit with a specific code on error to detect it easily
            # 77 is arbitrary but distinct
            env['MSAN_OPTIONS'] = 'halt_on_error=1:exitcode=77'

            build_success = False
            
            # Detect build system and build
            # Check for CMake
            if os.path.exists(os.path.join(src_root, "CMakeLists.txt")):
                build_dir = os.path.join(src_root, "build_fuzz")
                os.makedirs(build_dir, exist_ok=True)
                # Configure
                subprocess.run(
                    ["cmake", src_root, "-DBUILD_SHARED_LIBS=OFF", "-DBUILD_TESTING=OFF"],
                    cwd=build_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                # Build
                subprocess.run(
                    ["make", "-j8"], 
                    cwd=build_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                build_success = True
            
            # Check for Autotools / Makefile
            elif os.path.exists(os.path.join(src_root, "configure")) or os.path.exists(os.path.join(src_root, "Makefile")):
                if os.path.exists(os.path.join(src_root, "autogen.sh")):
                    subprocess.run(["./autogen.sh"], cwd=src_root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if os.path.exists(os.path.join(src_root, "configure")):
                    subprocess.run(
                        ["./configure", "--disable-shared"], 
                        cwd=src_root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                
                subprocess.run(
                    ["make", "-j8"], 
                    cwd=src_root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                build_success = True

            # 3. Locate Target Binary
            executables = []
            # Recursive search for executables
            for root, dirs, files in os.walk(base_dir):
                for f in files:
                    path = os.path.join(root, f)
                    if os.access(path, os.X_OK) and not os.path.isdir(path):
                        if f.endswith('.sh') or f.endswith('.py') or f.endswith('.pl'):
                            continue
                        # Simple ELF check
                        try:
                            with open(path, 'rb') as bf:
                                if bf.read(4) == b'\x7fELF':
                                    executables.append(path)
                        except:
                            pass
            
            if not executables:
                return b"" # Build failed or no binaries

            # Heuristic to pick the best fuzz target
            # Priorities: 'fuzz', 'test', 'demo', 'info', 'header'
            priorities = ['fuzz', 'test', 'demo', 'example', 'util', 'info', 'header']
            
            def score_exe(exe_path):
                name = os.path.basename(exe_path).lower()
                for i, p in enumerate(priorities):
                    if p in name:
                        return i
                return len(priorities)
            
            executables.sort(key=score_exe)
            target_bin = executables[0]

            # 4. Gather Seeds
            seeds = []
            # Look for common file formats in the source tree
            exts = ['.exr', '.xml', '.tif', '.tiff', '.jpg', '.png', '.pdf', '.json']
            for root, dirs, files in os.walk(base_dir):
                for f in files:
                    if any(f.lower().endswith(ext) for ext in exts):
                        p = os.path.join(root, f)
                        if os.path.getsize(p) < 500000: # Skip very large files
                            seeds.append(p)
            
            # Heuristic for OpenEXR (based on problem context)
            if "exr" in src_root.lower() or "openexr" in src_root.lower():
                dummy_exr = os.path.join(base_dir, "seed.exr")
                # Magic bytes for EXR: 0x76, 0x2f, 0x31, 0x01
                with open(dummy_exr, "wb") as f:
                    f.write(b'\x76\x2f\x31\x01' + b'\x00' * 64)
                seeds.append(dummy_exr)

            if not seeds:
                # Fallback seed
                dummy = os.path.join(base_dir, "seed.bin")
                with open(dummy, "wb") as f:
                    f.write(b"A" * 128)
                seeds.append(dummy)

            # 5. Fuzzing Loop
            population = []
            for s in seeds:
                try:
                    with open(s, "rb") as f:
                        population.append(bytearray(f.read()))
                except:
                    pass
            
            if not population:
                population.append(bytearray(b"A" * 100))

            start_time = time.time()
            # Run for a limited time to generate PoC
            time_limit = 45 

            while time.time() - start_time < time_limit:
                # Select parent
                parent = random.choice(population)
                # Mutate
                child = bytearray(parent)
                mut_kind = random.randint(0, 3)

                if mut_kind == 0 and len(child) > 0: # Byte flip
                    idx = random.randint(0, len(child)-1)
                    child[idx] = random.randint(0, 255)
                elif mut_kind == 1 and len(child) > 0: # Bit flip
                    idx = random.randint(0, len(child)-1)
                    bit = random.randint(0, 7)
                    child[idx] ^= (1 << bit)
                elif mut_kind == 2: # Delete
                    if len(child) > 5:
                        start = random.randint(0, len(child)-2)
                        length = random.randint(1, min(16, len(child)-start))
                        del child[start:start+length]
                elif mut_kind == 3: # Insert
                    pos = random.randint(0, len(child))
                    length = random.randint(1, 16)
                    child[pos:pos] = os.urandom(length)

                poc_data = bytes(child)
                
                # Write to temp file
                test_file = os.path.join(base_dir, "fuzz_input")
                with open(test_file, "wb") as f:
                    f.write(poc_data)

                # Execute
                try:
                    # Run target with the input file
                    # MSAN_OPTIONS set to exit with 77 on error
                    proc = subprocess.run(
                        [target_bin, test_file],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        env=env,
                        timeout=0.5
                    )
                    
                    # Check for MSan trigger
                    if proc.returncode == 77 or b"MemorySanitizer" in proc.stderr or b"Use-of-uninitialized-value" in proc.stderr:
                        return poc_data
                    
                    # Also check for Segfaults which might be caused by uninit values
                    if proc.returncode == -11: # SIGSEGV
                         return poc_data

                except subprocess.TimeoutExpired:
                    continue
                except Exception:
                    continue

            # If no crash found, return the last generated input or a seed
            # (Ideally we should have found it, but we must return bytes)
            return bytes(population[0])

        finally:
            shutil.rmtree(base_dir, ignore_errors=True)