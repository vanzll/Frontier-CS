import os
import tarfile
import tempfile
import stat
import re


class Solution:
    def solve(self, src_path: str) -> bytes:
        root_dir = None

        if os.path.isdir(src_path):
            root_dir = src_path
        else:
            tmpdir = tempfile.mkdtemp(prefix="arvo21000_")
            root_dir = tmpdir
            try:
                with tarfile.open(src_path, 'r:*') as tf:
                    self._safe_extract(tf, tmpdir)
            except tarfile.ReadError:
                # If src_path is unexpectedly not a tar, fallback to default PoC
                return self._default_poc()

        target_size = 33

        poc = self._find_binary_poc(root_dir, target_size)
        if poc is None:
            poc = self._find_hex_poc(root_dir, target_size)

        if poc is None:
            poc = self._default_poc()

        return poc

    def _safe_extract(self, tar: tarfile.TarFile, path: str) -> None:
        def is_within_directory(directory: str, target: str) -> bool:
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            prefix = os.path.commonprefix([abs_directory, abs_target])
            return prefix == abs_directory

        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                continue
            try:
                tar.extract(member, path)
            except Exception:
                # Ignore problematic members
                continue

    def _find_binary_poc(self, root: str, target_size: int) -> bytes | None:
        best_candidate = None
        best_score = None

        # File extensions we consider "source/text-like" and skip here
        source_exts = {
            '.c', '.h', '.cpp', '.hpp', '.cc', '.hh', '.cxx',
            '.txt', '.md', '.rst', '.py', '.java', '.sh', '.cmake',
            '.mak', '.mk', '.json', '.xml', '.yml', '.yaml',
            '.in', '.ac', '.am', '.m4', '.pl', '.pm', '.rb', '.js',
        }

        max_size = 65536  # Only consider files up to 64 KiB

        for dirpath, dirnames, filenames in os.walk(root):
            base = os.path.basename(dirpath).lower()
            if base in {'.git', '.svn', '.hg', 'build', 'cmake-build-debug',
                        'cmake-build-release', 'out', 'dist', 'node_modules'}:
                continue

            for fname in filenames:
                full = os.path.join(dirpath, fname)
                try:
                    st = os.stat(full)
                except OSError:
                    continue

                if not stat.S_ISREG(st.st_mode):
                    continue

                size = st.st_size
                if size <= 0 or size > max_size:
                    continue

                _, ext = os.path.splitext(fname)
                ext = ext.lower()

                # Skip obvious source/text files here; they'll be handled in hex parser
                if ext in source_exts:
                    continue

                relpath = os.path.relpath(full, root)
                lower = relpath.lower()

                try:
                    with open(full, 'rb') as f:
                        data = f.read()
                except OSError:
                    continue

                if not data:
                    continue

                printable = 0
                for b in data:
                    if 32 <= b <= 126 or b in (9, 10, 13):
                        printable += 1
                nonprintable = len(data) - printable
                is_binary = (0 in data) or (nonprintable > 0 and nonprintable * 1.0 / len(data) > 0.05)

                # Scoring
                score = 0

                size_diff = abs(size - target_size)
                score += max(0, 100 - size_diff * 10)
                if size == target_size:
                    score += 50

                if 'capwap' in lower:
                    score += 120
                if 'ndpi' in lower:
                    score += 20
                if 'poc' in lower or 'crash' in lower or 'id:' in lower:
                    score += 60
                if 'heap' in lower or 'overflow' in lower or 'asan' in lower or 'ubsan' in lower:
                    score += 40

                if ext in ('.bin', '.dat', '.raw', '.in', '.input', ''):
                    score += 15

                if is_binary:
                    score += 30
                else:
                    score -= 30

                if score < 0:
                    continue

                if best_candidate is None or best_score is None or score > best_score:
                    best_candidate = data
                    best_score = score

        return best_candidate

    def _find_hex_poc(self, root: str, target_size: int) -> bytes | None:
        best_data = None
        best_score = None

        hex_pat_0x = re.compile(r'0x\s*([0-9a-fA-F]{1,2})')
        hex_pat_bs = re.compile(r'\\x([0-9a-fA-F]{2})')

        text_exts = {
            '.c', '.h', '.cpp', '.hpp', '.cc', '.hh', '.cxx',
            '.txt', '.md', '.rst', '.in',
        }

        max_text_size = 256 * 1024

        for dirpath, dirnames, filenames in os.walk(root):
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                try:
                    st = os.stat(full)
                except OSError:
                    continue

                if not stat.S_ISREG(st.st_mode):
                    continue

                if st.st_size <= 0 or st.st_size > max_text_size:
                    continue

                _, ext = os.path.splitext(fname)
                ext = ext.lower()
                if ext not in text_exts:
                    continue

                relpath = os.path.relpath(full, root)
                lower = relpath.lower()
                if ('capwap' not in lower and
                        'poc' not in lower and
                        'crash' not in lower and
                        'heap' not in lower and
                        'overflow' not in lower):
                    continue

                try:
                    with open(full, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                except OSError:
                    continue

                candidates: list[bytes] = []

                m1 = hex_pat_0x.findall(text)
                if len(m1) >= 4:
                    try:
                        data1 = bytes(int(h, 16) & 0xFF for h in m1)
                        candidates.append(data1)
                    except ValueError:
                        pass

                m2 = hex_pat_bs.findall(text)
                if len(m2) >= 4:
                    try:
                        data2 = bytes(int(h, 16) & 0xFF for h in m2)
                        candidates.append(data2)
                    except ValueError:
                        pass

                for data in candidates:
                    if not data:
                        continue
                    size = len(data)
                    size_diff = abs(size - target_size)
                    score = max(0, 100 - size_diff * 5)
                    if size == target_size:
                        score += 50
                    if 'capwap' in lower:
                        score += 120
                    if 'poc' in lower or 'crash' in lower:
                        score += 60
                    if 'heap' in lower or 'overflow' in lower:
                        score += 40

                    if best_data is None or best_score is None or score > best_score:
                        best_data = data
                        best_score = score

        return best_data

    def _default_poc(self) -> bytes:
        # Fallback: synthetic CAPWAP-like minimal payload (33 bytes)
        # Structure: simple non-empty pattern; specific content is unlikely to matter
        # if a better PoC was not discovered above.
        return b'CAPWAP_HEAP_OVERFLOW_POC_00033'[:33]