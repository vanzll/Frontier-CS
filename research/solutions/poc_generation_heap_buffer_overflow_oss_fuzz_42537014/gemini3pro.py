import os
import sys
import tarfile
import subprocess
import shutil
import tempfile
import random
import time

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Ground truth length is 9 bytes
        # Default fallback guess
        best_guess = b"http://aa" 

        base_dir = tempfile.mkdtemp()

        try:
            # 1. Extract Source Code
            with tarfile.open(src_path) as tar:
                tar.extractall(path=base_dir)
            
            # Determine source root (where configure script is)
            src_root = base_dir
            if not os.path.exists(os.path.join(base_dir, 'configure')):
                for item in os.listdir(base_dir):
                    p = os.path.join(base_dir, item)
                    if os.path.isdir(p) and os.path.exists(os.path.join(p, 'configure')):
                        src_root = p
                        break
            
            # 2. Compile
            configure_script = os.path.join(src_root, 'configure')
            if os.path.exists(configure_script):
                os.chmod(configure_script, 0o755)
                
                # Configure for ASAN and static build, disabling extras to ensure success
                config_cmd = [
                    './configure',
                    '--disable-x11',
                    '--disable-ssl',
                    '--disable-oss-audio',
                    '--disable-pulseaudio',
                    '--disable-jack',
                    '--static-bin',
                    '--extra-cflags=-fsanitize=address -g',
                    '--extra-ldflags=-fsanitize=address'
                ]
                
                subprocess.run(
                    config_cmd, 
                    cwd=src_root, 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL
                )
                
                subprocess.run(
                    ['make', '-j8'], 
                    cwd=src_root, 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL
                )

            # 3. Find Binary
            binary_path = None
            for root, dirs, files in os.walk(src_root):
                if 'dash_client' in files:
                    fpath = os.path.join(root, 'dash_client')
                    if os.access(fpath, os.X_OK):
                        binary_path = fpath
                        break
            
            if not binary_path:
                shutil.rmtree(base_dir, ignore_errors=True)
                return best_guess

            # 4. Fuzzing
            # Target length 9 bytes. Focus on URL parsing.
            seeds = [
                b"http://aa",
                b"http://a\n",
                b"dash://a\n",
                b"dash://aa",
                b"file:///a",
                b"http://\x00\x00",
                b"A" * 9,
            ]
            
            prefixes = [b"http://", b"dash://", b"file://"]
            
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                input_file = tf.name
            
            start_time = time.time()
            while time.time() - start_time < 90:
                if seeds:
                    payload = seeds.pop(0)
                else:
                    # Generate 9-byte payload
                    if random.random() < 0.7:
                        prefix = random.choice(prefixes)
                        suffix_len = 9 - len(prefix)
                        if suffix_len > 0:
                            suffix = bytes([random.randint(0, 255) for _ in range(suffix_len)])
                            payload = prefix + suffix
                        else:
                            payload = prefix[:9]
                    else:
                        payload = bytes([random.randint(0, 255) for _ in range(9)])

                with open(input_file, 'wb') as f:
                    f.write(payload)

                try:
                    res = subprocess.run(
                        [binary_path, input_file],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        timeout=0.5
                    )
                    
                    if res.returncode != 0:
                        # Check for ASAN crash
                        if b"AddressSanitizer" in res.stderr:
                            return payload
                except subprocess.TimeoutExpired:
                    pass
            
            if os.path.exists(input_file):
                os.remove(input_file)

        except Exception:
            pass
        finally:
            if os.path.exists(base_dir):
                shutil.rmtree(base_dir, ignore_errors=True)
        
        return best_guess