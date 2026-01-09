import os
import sys
import tarfile
import tempfile
import shutil
import subprocess
import random
import time
import glob
from concurrent.futures import ThreadPoolExecutor

class Solution:
    def solve(self, src_path: str) -> bytes:
        base_dir = tempfile.mkdtemp()
        try:
            # Extract source
            with tarfile.open(src_path) as tar:
                tar.extractall(path=base_dir)
            
            # Find the actual source directory
            src_dir = base_dir
            for item in os.listdir(base_dir):
                candidate = os.path.join(base_dir, item)
                if os.path.isdir(candidate) and 'configure' in os.listdir(candidate):
                    src_dir = candidate
                    break
            
            # Setup build environment
            build_env = os.environ.copy()
            # -O1 is faster to compile than -O2, sufficient for ASAN
            build_env["CFLAGS"] = "-fsanitize=address -g -O1"
            build_env["CXXFLAGS"] = "-fsanitize=address -g -O1"
            build_env["LDFLAGS"] = "-fsanitize=address"
            
            # Configure
            # Disable non-essential features to speed up build and avoid missing deps
            configure_cmd = [
                "./configure",
                "--static-bin",
                "--disable-shared",
                "--disable-x11",
                "--disable-gl",
                "--disable-ssl",
                "--disable-ffmpeg",
                "--disable-jack",
                "--disable-pulseaudio",
                "--disable-alsa"
            ]
            
            subprocess.run(
                configure_cmd,
                cwd=src_dir,
                env=build_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
            
            # Make
            subprocess.run(
                ["make", "-j8"],
                cwd=src_dir,
                env=build_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
            
            # Locate MP4Box binary
            mp4box = None
            possible_paths = [
                os.path.join(src_dir, "bin", "gcc", "MP4Box"),
                os.path.join(src_dir, "bin", "MP4Box")
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    mp4box = p
                    break
            
            if not mp4box:
                found = glob.glob(os.path.join(src_dir, "**", "MP4Box"), recursive=True)
                if found:
                    mp4box = found[0]
                else:
                    # If build failed, we can't generate the specific PoC dynamically
                    return b""

            # Collect seeds
            seeds = []
            extensions = {'.mp4', '.hvc', '.hevc', '.265', '.cmp'}
            for root, dirs, files in os.walk(src_dir):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in extensions:
                        path = os.path.join(root, f)
                        if os.path.getsize(path) < 100 * 1024:
                            try:
                                with open(path, "rb") as fd:
                                    seeds.append((fd.read(), ext))
                            except:
                                pass
            
            if not seeds:
                # Fallback minimal seed
                dummy = b'\x00\x00\x00\x20\x66\x74\x79\x70\x69\x73\x6f\x6d\x00\x00\x02\x00\x69\x73\x6f\x6d\x69\x73\x6f\x32\x61\x76\x63\x31\x6d\x70\x34\x31'
                seeds.append((dummy, '.mp4'))

            # Fuzzing
            result_poc = None
            start_time = time.time()
            max_duration = 300
            
            # Configure ASAN options to avoid leaks noise and ensure abort
            build_env["ASAN_OPTIONS"] = "detect_leaks=0:abort_on_error=1:symbolize=0"

            def worker():
                nonlocal result_poc
                local_seeds = list(seeds)
                # Cap seeds to avoid memory issues
                if len(local_seeds) > 100:
                    local_seeds = random.sample(local_seeds, 100)
                
                while time.time() - start_time < max_duration:
                    if result_poc:
                        return
                    
                    seed_data, seed_ext = random.choice(local_seeds)
                    
                    # Mutation
                    if random.random() < 0.1:
                        mutated = bytearray(seed_data)
                    else:
                        mutated = bytearray(seed_data)
                        num_mutations = random.randint(1, 5)
                        for _ in range(num_mutations):
                            if len(mutated) == 0:
                                mutated.append(0)
                                continue
                            
                            op = random.randint(0, 3)
                            l = len(mutated)
                            if op == 0: # Flip
                                idx = random.randint(0, l - 1)
                                mutated[idx] ^= random.randint(1, 255)
                            elif op == 1: # Insert
                                idx = random.randint(0, l)
                                chunk = random.randbytes(random.randint(1, 16))
                                mutated[idx:idx] = chunk
                            elif op == 2: # Delete
                                if l > 4:
                                    idx = random.randint(0, l - 2)
                                    length = random.randint(1, min(16, l - idx))
                                    del mutated[idx:idx + length]
                            elif op == 3: # Int overwrite
                                if l > 4:
                                    idx = random.randint(0, l - 4)
                                    val = random.choice([0xFFFFFFFF, 0x7FFFFFFF, 0x80000000, 0x0000FFFF, 0xFFFF0000])
                                    try:
                                        mutated[idx:idx+4] = val.to_bytes(4, 'big')
                                    except:
                                        pass
                    
                    data = bytes(mutated)
                    
                    # File creation
                    fd, tmp_path = tempfile.mkstemp(suffix=seed_ext)
                    os.close(fd)
                    with open(tmp_path, "wb") as f:
                        f.write(data)
                    
                    try:
                        # -info triggers parsing. For raw hevc, it imports.
                        cmd = [mp4box, "-info", tmp_path]
                        proc = subprocess.run(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            env=build_env,
                            timeout=1.0
                        )
                        
                        if proc.returncode != 0:
                            stderr_out = proc.stderr.decode(errors='ignore')
                            if "stack-buffer-overflow" in stderr_out:
                                result_poc = data
                                os.remove(tmp_path)
                                return
                    except:
                        pass
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(worker) for _ in range(8)]
                for f in futures:
                    f.result()
                    if result_poc:
                        break
            
            return result_poc if result_poc else b""

        except Exception:
            return b""
        finally:
            shutil.rmtree(base_dir)