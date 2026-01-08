import os
import sys
import subprocess
import tempfile
import shutil
import tarfile
import random
import time
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        base_dir = tempfile.mkdtemp()
        try:
            # 1. Prepare Source Code
            src_work_dir = os.path.join(base_dir, "work")
            os.makedirs(src_work_dir, exist_ok=True)
            
            # Extract or copy source
            if os.path.isdir(src_path):
                subprocess.run(["cp", "-r", src_path + "/.", src_work_dir], check=False, stderr=subprocess.DEVNULL)
            else:
                try:
                    mode = "r"
                    if src_path.endswith('.xz'): mode = "r:xz"
                    elif src_path.endswith('.gz') or src_path.endswith('.tgz'): mode = "r:gz"
                    with tarfile.open(src_path, mode) as tar:
                        tar.extractall(path=src_work_dir)
                except Exception:
                    # Fallback
                    pass

            # Locate root with Makefile
            upx_root = src_work_dir
            for root, dirs, files in os.walk(src_work_dir):
                if "Makefile" in files:
                    upx_root = root
                    break

            # 2. Compile UPX with ASAN
            env = os.environ.copy()
            env["CC"] = "clang"
            env["CXX"] = "clang++"
            flags = "-fsanitize=address -g -O1"
            env["CFLAGS"] = flags
            env["CXXFLAGS"] = flags
            env["LDFLAGS"] = "-fsanitize=address"
            
            # Build
            try:
                subprocess.run(["make", "-j8"], cwd=upx_root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except:
                pass
            
            # Find binary
            upx_bin = None
            candidates = [
                os.path.join(upx_root, "src", "upx.out"),
                os.path.join(upx_root, "build", "upx.out"),
                os.path.join(upx_root, "upx.out"),
                os.path.join(upx_root, "src", "upx"),
                os.path.join(upx_root, "upx")
            ]
            for c in candidates:
                if os.path.exists(c) and os.access(c, os.X_OK):
                    upx_bin = c
                    break
            
            if not upx_bin:
                # Scan
                for root, dirs, files in os.walk(src_work_dir):
                    if "upx" in files:
                        p = os.path.join(root, "upx")
                        if os.access(p, os.X_OK):
                            upx_bin = p
                            break
                            
            if not upx_bin:
                return b""

            # 3. Generate Seed (Small Packed ELF)
            dummy_c = os.path.join(base_dir, "a.c")
            with open(dummy_c, "w") as f:
                f.write("int main(){return 0;}")
            
            dummy_exe = os.path.join(base_dir, "a.out")
            subprocess.run(["clang", "-s", dummy_c, "-o", dummy_exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            packed_exe = os.path.join(base_dir, "p.out")
            # Use -1 for speed and simplicity
            subprocess.run([upx_bin, "-1", "-f", "-o", packed_exe, dummy_exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if not os.path.exists(packed_exe):
                return b""

            with open(packed_exe, "rb") as f:
                seed = f.read()

            # 4. Fuzzing Loop
            best_poc = b""
            start_time = time.time()
            
            while time.time() - start_time < 50:
                mut = bytearray(seed)
                
                # Mutation Strategy
                r = random.random()
                if r < 0.3:
                    # Truncation
                    l = len(mut)
                    if l > 100:
                        mut = mut[:random.randint(100, l)]
                elif r < 0.6:
                    # Random Byte Overwrites
                    for _ in range(random.randint(1, 20)):
                        pos = random.randint(0, len(mut)-1)
                        mut[pos] = random.randint(0, 255)
                else:
                    # Integer Overwrites (Header corruption)
                    if len(mut) > 4:
                        for _ in range(random.randint(1, 5)):
                            pos = random.randint(0, len(mut)-4)
                            val = random.choice([0xFFFFFFFF, 0x00000000, 0x7FFFFFFF, 0x80000000])
                            struct.pack_into("<I", mut, pos, val)

                # Write test case
                t_file = os.path.join(base_dir, "t.upx")
                with open(t_file, "wb") as f:
                    f.write(mut)
                
                # Run UPX Test
                try:
                    res = subprocess.run([upx_bin, "-t", t_file], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=0.5)
                    if res.returncode != 0:
                        if b"AddressSanitizer" in res.stderr and (b"heap-buffer-overflow" in res.stderr or b"SEGV" in res.stderr):
                            best_poc = bytes(mut)
                            break
                except subprocess.TimeoutExpired:
                    pass
            
            # 5. Minimize
            if best_poc:
                current = best_poc
                step = 64
                while len(current) > 200 and step > 0:
                    if len(current) <= step:
                        step //= 2
                        continue
                    
                    # Try truncating from end
                    candidate = current[:-step]
                    t_file = os.path.join(base_dir, "m.upx")
                    with open(t_file, "wb") as f:
                        f.write(candidate)
                    
                    try:
                        res = subprocess.run([upx_bin, "-t", t_file], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=0.5)
                        if res.returncode != 0 and (b"AddressSanitizer" in res.stderr):
                            current = candidate
                        else:
                            step //= 2
                    except:
                        step //= 2
                return current

            return b""
            
        except Exception:
            return b""
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)