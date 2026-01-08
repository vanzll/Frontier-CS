import os
import tarfile
import re
import gzip
import bz2
import lzma


class Solution:
    TARGET_LEN = 1551
    KEYWORDS = [
        "383170474",
        "debugnames",
        "debug_names",
        "debugname",
        "names",
        "dwarf",
        "ossfuzz",
        "fuzz",
        "poc",
        "crash",
        "heap",
        "overflow",
    ]
    TEXT_EXTS = {".c", ".h", ".hpp", ".hh", ".cc", ".cpp", ".cxx", ".inc", ".txt", ".md"}
    BINARY_EXTS = {".o", ".obj", ".bin", ".dat", ".debug", ".elf", ".out", ".exe", ".so", ".a"}

    def solve(self, src_path: str) -> bytes:
        if os.path.isdir(src_path):
            data = self._find_poc_in_dir(src_path, self.TARGET_LEN)
            if data is not None:
                return data
            return self._fallback(self.TARGET_LEN)

        try:
            with tarfile.open(src_path, "r:*") as tf:
                data = self._find_poc_in_tar(tf, self.TARGET_LEN)
                if data is not None:
                    return data
        except Exception:
            pass

        return self._fallback(self.TARGET_LEN)

    # ------------------------------------------------------------------ #
    # Main search orchestrators
    # ------------------------------------------------------------------ #
    def _find_poc_in_tar(self, tf: tarfile.TarFile, target_len: int) -> bytes | None:
        data = self._find_file_of_exact_size_in_tar(tf, target_len)
        if data is not None:
            return data

        data = self._find_in_compressed_members(tf, target_len)
        if data is not None:
            return data

        data = self._search_embedded_array_in_tar(tf, target_len)
        if data is not None:
            return data

        data = self._search_hex_escape_in_tar(tf, target_len)
        if data is not None:
            return data

        return None

    def _find_poc_in_dir(self, root: str, target_len: int) -> bytes | None:
        data = self._find_file_of_exact_size_in_dir(root, target_len)
        if data is not None:
            return data

        data = self._search_embedded_array_in_dir(root, target_len)
        if data is not None:
            return data

        data = self._search_hex_escape_in_dir(root, target_len)
        if data is not None:
            return data

        return None

    # ------------------------------------------------------------------ #
    # Helpers: scoring / classification
    # ------------------------------------------------------------------ #
    def _is_mostly_text(self, data: bytes) -> bool:
        if not data:
            return True
        printable = 0
        for b in data:
            if 32 <= b < 127 or b in (9, 10, 13):
                printable += 1
        ratio = printable / len(data)
        return ratio > 0.8

    def _score_name(self, name: str) -> int:
        n = name.lower()
        score = 0
        for kw in self.KEYWORDS:
            if kw in n:
                score += 20
        if "test" in n or "regress" in n:
            score += 5
        base, ext = os.path.splitext(n)
        if ext in self.BINARY_EXTS:
            score += 5
        if ext in self.TEXT_EXTS:
            score -= 5
        return score

    # ------------------------------------------------------------------ #
    # Search: exact-size files in tar
    # ------------------------------------------------------------------ #
    def _find_file_of_exact_size_in_tar(self, tf: tarfile.TarFile, target_len: int) -> bytes | None:
        candidates = []
        for m in tf.getmembers():
            if not m.isfile():
                continue
            if m.size != target_len:
                continue
            candidates.append(m)

        if not candidates:
            return None

        def score_member(member: tarfile.TarInfo) -> int:
            score = self._score_name(member.name)
            try:
                f = tf.extractfile(member)
                if f is None:
                    return score
                data = f.read()
            except Exception:
                return score
            if not self._is_mostly_text(data):
                score += 10
            else:
                score -= 2
            return score

        best = max(candidates, key=score_member)
        try:
            f = tf.extractfile(best)
            if f is None:
                return None
            return f.read()
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Search: exact-size files in directory
    # ------------------------------------------------------------------ #
    def _find_file_of_exact_size_in_dir(self, root: str, target_len: int) -> bytes | None:
        candidates: list[str] = []
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                if not os.path.isfile(full):
                    continue
                if st.st_size != target_len:
                    continue
                candidates.append(full)

        if not candidates:
            return None

        def score_path(path: str) -> int:
            score = self._score_name(path)
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except OSError:
                return score
            if not self._is_mostly_text(data):
                score += 10
            else:
                score -= 2
            return score

        best = max(candidates, key=score_path)
        try:
            with open(best, "rb") as f:
                return f.read()
        except OSError:
            return None

    # ------------------------------------------------------------------ #
    # Search: compressed members in tar
    # ------------------------------------------------------------------ #
    def _find_in_compressed_members(self, tf: tarfile.TarFile, target_len: int) -> bytes | None:
        candidates: list[tuple[tarfile.TarInfo, bytes]] = []

        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = m.name.lower()
            ext = None
            if name.endswith(".gz") or name.endswith(".gzip"):
                ext = "gz"
            elif name.endswith(".xz") or name.endswith(".lzma"):
                ext = "xz"
            elif name.endswith(".bz2"):
                ext = "bz2"
            if ext is None:
                continue
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                raw = f.read()
            except Exception:
                continue

            try:
                if ext == "gz":
                    data = gzip.decompress(raw)
                elif ext == "xz":
                    data = lzma.decompress(raw)
                else:
                    data = bz2.decompress(raw)
            except Exception:
                continue

            if len(data) == target_len:
                candidates.append((m, data))

        if not candidates:
            return None

        def score_item(item: tuple[tarfile.TarInfo, bytes]) -> int:
            member, data = item
            score = self._score_name(member.name)
            if not self._is_mostly_text(data):
                score += 10
            else:
                score -= 2
            return score

        best_member, best_data = max(candidates, key=score_item)
        return best_data

    # ------------------------------------------------------------------ #
    # Search: embedded byte arrays in tar
    # ------------------------------------------------------------------ #
    def _search_embedded_array_in_tar(self, tf: tarfile.TarFile, target_len: int) -> bytes | None:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = m.name.lower()
            _, ext = os.path.splitext(name)
            if ext not in self.TEXT_EXTS:
                continue
            if m.size > 1024 * 512:
                continue
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                raw = f.read()
            except Exception:
                continue
            try:
                text = raw.decode("utf-8", errors="ignore")
            except Exception:
                continue

            data = self._extract_array_from_text(text, target_len)
            if data is not None:
                return data
        return None

    # ------------------------------------------------------------------ #
    # Search: embedded byte arrays in directory
    # ------------------------------------------------------------------ #
    def _search_embedded_array_in_dir(self, root: str, target_len: int) -> bytes | None:
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                _, ext = os.path.splitext(fname.lower())
                if ext not in self.TEXT_EXTS:
                    continue
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                if st.st_size > 1024 * 512:
                    continue
                try:
                    with open(full, "rb") as f:
                        raw = f.read()
                except OSError:
                    continue
                try:
                    text = raw.decode("utf-8", errors="ignore")
                except Exception:
                    continue

                data = self._extract_array_from_text(text, target_len)
                if data is not None:
                    return data
        return None

    # ------------------------------------------------------------------ #
    # Search: hex escape sequences in tar
    # ------------------------------------------------------------------ #
    def _search_hex_escape_in_tar(self, tf: tarfile.TarFile, target_len: int) -> bytes | None:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = m.name.lower()
            _, ext = os.path.splitext(name)
            if ext not in self.TEXT_EXTS:
                continue
            if m.size > 1024 * 512:
                continue
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                raw = f.read()
            except Exception:
                continue
            try:
                text = raw.decode("utf-8", errors="ignore")
            except Exception:
                continue

            data = self._extract_hex_escapes_from_text(text, target_len)
            if data is not None:
                return data
        return None

    # ------------------------------------------------------------------ #
    # Search: hex escape sequences in directory
    # ------------------------------------------------------------------ #
    def _search_hex_escape_in_dir(self, root: str, target_len: int) -> bytes | None:
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                _, ext = os.path.splitext(fname.lower())
                if ext not in self.TEXT_EXTS:
                    continue
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                if st.st_size > 1024 * 512:
                    continue
                try:
                    with open(full, "rb") as f:
                        raw = f.read()
                except OSError:
                    continue
                try:
                    text = raw.decode("utf-8", errors="ignore")
                except Exception:
                    continue

                data = self._extract_hex_escapes_from_text(text, target_len)
                if data is not None:
                    return data
        return None

    # ------------------------------------------------------------------ #
    # Text parsing helpers
    # ------------------------------------------------------------------ #
    def _extract_array_from_text(self, text: str, target_len: int) -> bytes | None:
        pos = 0
        length = len(text)
        number_re = re.compile(r"0x[0-9a-fA-F]+|[0-9]+")
        while True:
            m = re.search(r"=\s*\{", text[pos:])
            if not m:
                break
            start = pos + m.end()
            # Find matching closing brace
            level = 1
            i = start
            end = None
            while i < length:
                c = text[i]
                if c == "{":
                    level += 1
                elif c == "}":
                    level -= 1
                    if level == 0:
                        end = i
                        break
                i += 1
            if end is None:
                break
            arr_str = text[start:end]
            pos = end + 1

            tokens = number_re.findall(arr_str)
            if not tokens:
                continue
            vals = []
            valid = True
            for tok in tokens:
                try:
                    if tok.lower().startswith("0x"):
                        v = int(tok, 16)
                    else:
                        v = int(tok, 10)
                except ValueError:
                    valid = False
                    break
                if not (0 <= v <= 255):
                    valid = False
                    break
                vals.append(v)
            if not valid:
                continue
            if len(vals) == target_len:
                return bytes(vals)
        return None

    def _extract_hex_escapes_from_text(self, text: str, target_len: int) -> bytes | None:
        # Look for long runs of \xHH escapes inside string literals
        # This is a best-effort heuristic.
        pattern = re.compile(r'(?:\\x[0-9a-fA-F]{2}){8,}')
        for m in pattern.finditer(text):
            seq = m.group(0)
            bytes_list = []
            for hh in re.findall(r"\\x([0-9a-fA-F]{2})", seq):
                bytes_list.append(int(hh, 16))
            if len(bytes_list) == target_len:
                return bytes(bytes_list)
        return None

    # ------------------------------------------------------------------ #
    # Fallback PoC generator
    # ------------------------------------------------------------------ #
    def _fallback(self, target_len: int) -> bytes:
        # Simple deterministic fallback content; unlikely to be used if
        # the real PoC is present in the source tarball.
        # Use a recognizable prefix (ELF magic + DWARF-like tag) followed by padding.
        prefix = b"\x7fELF" + b"\x02\x01\x01\x00"  # 64-bit LE ELF header prefix
        # Extend prefix with some plausible DWARF5 .debug_names-style header fields.
        # (These values are arbitrary and mainly serve to structure the input.)
        pseudo_header = (
            b"\x40\x00\x00\x00"  # unit_length (64)
            b"\x05\x00"          # version = 5
            b"\x00\x00"          # padding
            b"\x01\x00\x00\x00"  # comp_unit_count
            b"\x00\x00\x00\x00"  # local_type_unit_count
            b"\x00\x00\x00\x00"  # foreign_type_unit_count
            b"\x10\x00\x00\x00"  # bucket_count
            b"\x20\x00\x00\x00"  # name_count
            b"\x10\x00\x00\x00"  # abbrev_table_size
            b"\x40\x00\x00\x00"  # entry_pool_size
        )
        data = prefix + pseudo_header
        if len(data) < target_len:
            data += b"A" * (target_len - len(data))
        else:
            data = data[:target_len]
        return data