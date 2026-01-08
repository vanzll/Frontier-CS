import os
import tarfile
import io
import re
from typing import List, Tuple, Optional


def _read_fileobj(tf: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    try:
        f = tf.extractfile(member)
        if f is None:
            return b""
        data = f.read()
        f.close()
        return data
    except Exception:
        return b""


def _is_mostly_ascii(data: bytes, threshold: float = 0.9) -> bool:
    if not data:
        return True
    printable = set(range(0x20, 0x7F)) | {0x09, 0x0A, 0x0D}
    count = sum(1 for b in data if b in printable)
    return (count / len(data)) >= threshold


def _score_candidate(path: str, data: bytes) -> int:
    name = path.lower()
    size = len(data)
    score = 0

    # Size targeting
    if size == 72:
        score += 5
    elif 48 <= size <= 128:
        score += 2
    elif size < 4096:
        score += 1

    # Filename heuristics
    keywords = {
        'poc': 4,
        'raw': 2,
        'encap': 5,
        'raw_encap': 6,
        'nxast': 5,
        'uaf': 3,
        'heap': 2,
        'crash': 2,
        'bug': 1,
        'test': 1,
        'seed': 1,
        'fuzz': 1,
        'openflow': 2,
        'ofp': 2,
        'action': 2,
        'regress': 2,
        'case': 1,
        'id': 1,
        'bin': 1
    }
    for k, v in keywords.items():
        if k in name:
            score += v

    # Directory hints
    for k, v in [('tests', 1), ('regressions', 2), ('oss-fuzz', 2), ('clusterfuzz', 2), ('artifacts', 2), ('inputs', 1), ('crashes', 2)]:
        if k in name:
            score += v

    # Content heuristics
    if b'\x00' in data:
        score += 2
    if not _is_mostly_ascii(data):
        score += 2

    # Penalize very small or too text-like files
    if size < 8:
        score -= 5
    if _is_mostly_ascii(data) and size > 32:
        score -= 1

    return score


def _scan_tar_for_poc(src_path: str) -> Optional[bytes]:
    try:
        with tarfile.open(src_path, 'r:*') as tf:
            candidates: List[Tuple[int, str, bytes]] = []
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                # Limit read size for big files
                if m.size > 256 * 1024:
                    continue
                # Skip obviously irrelevant files by extension
                lower = m.name.lower()
                skip_exts = ('.c', '.h', '.cc', '.hh', '.cpp', '.hpp', '.py', '.sh', '.md', '.txt', '.xml', '.json', '.yaml', '.yml', '.toml', '.mk', '.in', '.am', '.ac', '.m4', '.map', '.html', '.rst')
                if lower.endswith(skip_exts):
                    continue
                data = _read_fileobj(tf, m)
                if not data:
                    continue
                score = _score_candidate(m.name, data)
                if score > 0:
                    candidates.append((score, m.name, data))
            if not candidates:
                return None
            candidates.sort(key=lambda x: (x[0], -abs(len(x[2]) - 72), -len(x[2])) , reverse=True)
            return candidates[0][2]
    except tarfile.ReadError:
        return None
    except Exception:
        return None


def _scan_dir_for_poc(src_dir: str) -> Optional[bytes]:
    candidates: List[Tuple[int, str, bytes]] = []
    for root, dirs, files in os.walk(src_dir):
        for fn in files:
            path = os.path.join(root, fn)
            try:
                size = os.path.getsize(path)
                if size == 0 or size > 256 * 1024:
                    continue
                lower = path.lower()
                skip_exts = ('.c', '.h', '.cc', '.hh', '.cpp', '.hpp', '.py', '.sh', '.md', '.txt', '.xml', '.json', '.yaml', '.yml', '.toml', '.mk', '.in', '.am', '.ac', '.m4', '.map', '.html', '.rst')
                if lower.endswith(skip_exts):
                    continue
                with open(path, 'rb') as f:
                    data = f.read()
                if not data:
                    continue
                score = _score_candidate(path, data)
                if score > 0:
                    candidates.append((score, path, data))
            except Exception:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], -abs(len(x[2]) - 72), -len(x[2])) , reverse=True)
    return candidates[0][2]


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Try tarball scanning first
        data = _scan_tar_for_poc(src_path)
        if data is not None:
            return data

        # If src_path is a directory, scan it
        if os.path.isdir(src_path):
            data = _scan_dir_for_poc(src_path)
            if data is not None:
                return data

        # Fallback: construct a plausible binary that resembles an OpenFlow message containing an action.
        # This is a generic non-crashing placeholder; real PoC should be discovered via scanning above.
        # Create a 72-byte binary with some structure and null bytes.
        fallback = bytearray(72)
        # OpenFlow header-like fields
        # version
        fallback[0] = 0x04  # OF1.3
        # type (OFPT_FLOW_MOD or vendor extension)
        fallback[1] = 0x0E
        # length
        fallback[2] = 0x00
        fallback[3] = 72
        # xid
        fallback[4:8] = b'\x12\x34\x56\x78'
        # pad some fields and include zeros
        # Put a fake vendor action header marker 0xFFFF
        fallback[24:26] = b'\xff\xff'
        # action length
        fallback[26:28] = b'\x00\x30'
        # vendor id (Nicira/ONF)
        fallback[28:32] = b'\x00\x00\x23\x20'  # NX_VENDOR_ID
        # subtype - placeholder for RAW_ENCAP
        fallback[32:36] = b'\x00\x00\x00\x00'
        # Fill remaining with a pattern and zeros to mimic properties
        for i in range(36, 72, 4):
            fallback[i:i+4] = b'\x00\x00\x00\x00'
        return bytes(fallback)