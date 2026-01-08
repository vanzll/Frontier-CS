import sys
import os
import subprocess
import tempfile
import shutil
import tarfile
import random
import time
import struct
import stat

class Solution:
    def solve(self, src_path: str) -> bytes:
        work_dir = tempfile.mkdtemp()
        try:
            # 1. Extract source code
            with tarfile.open(src_path) as tar:
                tar.extractall(path=work_dir)
            
            src_root = work_dir
            # Find the actual source directory if nested
            for name in os.listdir(work_dir):
                p = os.path.join(work_dir, name)
                if os.path.isdir(p):
                    src_root = p
                    break
            
            # 2. Configure build environment for ASAN
            env = os.environ.copy()
            flags = "-g -O1 -fsanitize=address -fno-omit-frame-pointer"
            env['CFLAGS'] = flags
            env['CXXFLAGS'] = flags
            env['LDFLAGS'] = "-fsanitize=address"
            
            # 3. Build the project
            # Attempt autodetection of build system
            built = False
            
            # Strategy A: configure
            if os.path.exists(os.path.join(src_root, 'configure')):
                try:
                    subprocess.run(
                        ['./configure', '--disable-shared', '--enable-static', '--disable-bsdcpio', '--disable-bsdcat', '--without-zlib', '--without-bz2lib', '--without-xml2'],
                        cwd=src_root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                    )
                    subprocess.run(
                        ['make', '-j8'],
                        cwd=src_root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                    )
                    built = True
                except:
                    pass
            
            # Strategy B: CMake
            if not built and os.path.exists(os.path.join(src_root, 'CMakeLists.txt')):
                try:
                    bdir = os.path.join(src_root, 'build_fuzz')
                    os.makedirs(bdir, exist_ok=True)
                    subprocess.run(
                        ['cmake', '..', '-DENABLE_OPENSSL=OFF', '-DENABLE_LIBXML2=OFF', '-DENABLE_EXPAT=OFF'],
                        cwd=bdir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                    )
                    subprocess.run(
                        ['make', '-j8'],
                        cwd=bdir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                    )
                    src_root = bdir # executables likely here
                    built = True
                except:
                    pass

            # Strategy C: Raw make
            if not built:
                try:
                    subprocess.run(
                        ['make', '-j8'],
                        cwd=src_root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                    )
                except:
                    pass

            # 4. Identify the target binary
            target_bin = None
            candidates = []
            for root, dirs, files in os.walk(src_root):
                for f in files:
                    fp = os.path.join(root, f)
                    # Check if executable
                    if os.access(fp, os.X_OK) and not os.path.isdir(fp):
                        if f.endswith('.sh') or f.endswith('.py') or f.endswith('.o') or f.endswith('.so'):
                            continue
                        candidates.append(fp)
            
            # Prioritize 'bsdtar' as this is likely libarchive
            for c in candidates:
                if 'bsdtar' in os.path.basename(c):
                    target_bin = c
                    break
            
            if not target_bin and candidates:
                # Heuristic: pick the largest executable? Or one with 'test' in name?
                # Usually the main tool is what we want.
                target_bin = candidates[0]

            if not target_bin:
                # Fallback if build fails
                return b'Rar!\x1a\x07\x00' + b'A'*39

            # 5. Fuzzing Loop
            seeds = [
                b'Rar!\x1a\x07\x00',         # RAR 4.x
                b'Rar!\x1a\x07\x01\x00',     # RAR 5
                b'PK\x03\x04',               # Zip
                b'7z\xbc\xaf\x27\x1c',       # 7z
                b'begin 644 ',               # UUEncode
                b'\x1f\x8b',                 # Gzip
                b'BZh',                      # Bzip2
                b'!<arch>\n',                # AR
                b'xar!',                     # XAR
            ]
            
            start_time = time.time()
            # Run for up to 45 seconds
            while time.time() - start_time < 45:
                # Generate Input
                seed = random.choice(seeds)
                # Ground truth is 46 bytes, so we target small files
                target_len = 46
                payload = bytearray(seed)
                
                # Fill rest with random data
                while len(payload) < target_len:
                    payload.append(random.randint(0, 255))
                
                # Mutate
                # Focus on mutating bytes that might be offsets (integer overflow)
                num_mutations = random.randint(1, 5)
                for _ in range(num_mutations):
                    idx = random.randint(0, len(payload)-1)
                    op = random.randint(0, 3)
                    if op == 0: # Random byte
                        payload[idx] = random.randint(0, 255)
                    elif op == 1: # Bit flip
                        payload[idx] ^= (1 << random.randint(0, 7))
                    elif op == 2: # Interesting int injection
                        if idx <= len(payload) - 4:
                            val = random.choice([0xFFFFFFFF, 0x7FFFFFFF, 0x80000000, 0xFFFF, 0x7FFF, 0x8000])
                            try:
                                payload[idx:idx+4] = struct.pack('<I', val)
                            except:
                                pass
                    elif op == 3: # Delete/Insert (Length change)
                        if random.random() > 0.5:
                            del payload[idx]
                        else:
                            payload.insert(idx, random.randint(0, 255))

                test_input = bytes(payload)
                
                # Write to temp file
                tf_path = os.path.join(work_dir, 'poc.dat')
                with open(tf_path, 'wb') as f:
                    f.write(test_input)
                
                try:
                    # Run target
                    # -tf lists contents (safe parsing)
                    proc = subprocess.run(
                        [target_bin, '-tf', tf_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        env=env,
                        timeout=1.0
                    )
                    
                    if proc.returncode != 0:
                        # Analyze stderr
                        err = proc.stderr.decode(errors='ignore')
                        if "AddressSanitizer" in err and ("heap-buffer-overflow" in err or "negative" in err or "SEGV" in err):
                            return test_input
                except subprocess.TimeoutExpired:
                    continue
                except Exception:
                    continue

            # Fallback if no crash found
            return b'Rar!\x1a\x07\x00' + b'\xff' * 39

        except Exception:
            return b'Rar!\x1a\x07\x00' + b'\x00' * 39
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)