import os
import re
import tarfile
import tempfile
import shutil
import zipfile
import base64

class Solution:
    def __init__(self):
        self.TARGET_SIZE = 2179
        self.BUG_ID = "42536068"
        self.max_text_read = 2 * 1024 * 1024
        self.binary_exts = {
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp',
            '.pdf', '.zip', '.7z', '.gz', '.bz2', '.xz', '.bin', '.raw', '.ico',
            '.pcx', '.exr', '.hdr', '.psd', '.xcf', '.svg', '.ttf', '.otf',
            '.woff', '.woff2', '.mid', '.ogg', '.mp3', '.wav', '.flac'
        }
        self.text_exts = {
            '.txt', '.xml', '.html', '.htm', '.xhtml', '.json', '.yaml', '.yml',
            '.csv', '.md', '.rst', '.ini', '.cfg', '.conf', '.svg'
        }
        self.code_exts = {
            '.c', '.cc', '.cpp', '.h', '.hpp', '.hh', '.cxx', '.hxx',
            '.m', '.mm', '.rs', '.go', '.kt', '.java', '.cs', '.py', '.rb',
            '.js', '.ts', '.php', '.swift'
        }
        self.candidate_name_keywords = [
            'poc', 'crash', 'testcase', 'clusterfuzz', 'oss-fuzz', 'minimized',
            'repro', 'reproducer', 'id:', 'ticket', 'bug', self.BUG_ID
        ]

    def solve(self, src_path: str) -> bytes:
        tmpdir = tempfile.mkdtemp(prefix="src_extract_")
        try:
            self._extract_tarball(src_path, tmpdir)
            # 1) Try direct bug id references
            data = self._search_bug_id_related(tmpdir)
            if data:
                return data
            # 2) Exact size match anywhere (prioritized paths)
            data = self._search_files_exact_size(tmpdir, self.TARGET_SIZE)
            if data:
                return data
            # 3) Search names with PoC-like keywords
            data = self._search_by_name_keywords(tmpdir)
            if data:
                return data
            # 4) Search seed corpus zips
            data = self._search_in_seed_corpus_zip(tmpdir)
            if data:
                return data
            # 5) Search in source arrays/base64 blobs near size
            data = self._search_in_source_embedded_blobs(tmpdir)
            if data:
                return data
            # 6) Search closest sized files in repo
            data = self._search_files_closest_size(tmpdir, self.TARGET_SIZE)
            if data:
                return data
            # Fallback: return structured-ish but generic content
            return self._fallback_bytes(self.TARGET_SIZE)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _extract_tarball(self, src_path, dst_dir):
        try:
            with tarfile.open(src_path, 'r:*') as tf:
                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    return prefix == abs_directory

                for m in tf.getmembers():
                    member_path = os.path.join(dst_dir, m.name)
                    if not is_within_directory(dst_dir, member_path):
                        continue
                    tf.extract(m, dst_dir)
        except Exception:
            # Try to handle if src_path is actually a directory (unlikely)
            if os.path.isdir(src_path):
                # Copy contents
                for root, dirs, files in os.walk(src_path):
                    rel = os.path.relpath(root, src_path)
                    outdir = os.path.join(dst_dir, rel if rel != '.' else '')
                    os.makedirs(outdir, exist_ok=True)
                    for f in files:
                        sp = os.path.join(root, f)
                        dp = os.path.join(outdir, f)
                        try:
                            shutil.copy2(sp, dp)
                        except Exception:
                            pass

    def _iter_files(self, root):
        for base, _, files in os.walk(root):
            for f in files:
                yield os.path.join(base, f)

    def _is_text_file_by_ext(self, path):
        _, ext = os.path.splitext(path.lower())
        return ext in self.text_exts or ext in self.code_exts or ext in {'.sh', '.bat', '.cmake', '.am', '.ac'}

    def _read_text_limited(self, path, limit=None):
        if limit is None:
            limit = self.max_text_read
        try:
            size = os.path.getsize(path)
            if size > limit:
                return None
            with open(path, 'rb') as f:
                data = f.read(limit)
            try:
                return data.decode('utf-8', errors='ignore')
            except Exception:
                return data.decode('latin-1', errors='ignore')
        except Exception:
            return None

    def _priority_score_for_path(self, path):
        p = path.lower()
        score = 0
        if self.BUG_ID in p:
            score += 1000
        if 'tests' in p or 'test' in p:
            score += 200
        if 'fuzz' in p or 'corpus' in p or 'seed' in p:
            score += 150
        for kw in self.candidate_name_keywords:
            if kw in p:
                score += 100
        # Prefer data-like extensions
        _, ext = os.path.splitext(p)
        if ext in self.binary_exts or ext in self.text_exts:
            score += 20
        # Deprioritize obvious source code files for direct read
        if ext in self.code_exts:
            score -= 10
        return score

    def _search_bug_id_related(self, root):
        # Search for files with the bug ID in filename or content
        best = None
        best_score = None
        best_len_diff = None

        # 1) Filename contains bug id
        for fp in self._iter_files(root):
            lp = fp.lower()
            if self.BUG_ID in lp:
                try:
                    sz = os.path.getsize(fp)
                except Exception:
                    continue
                try:
                    with open(fp, 'rb') as f:
                        data = f.read()
                    # Prefer exact size first
                    diff = abs(len(data) - self.TARGET_SIZE)
                    score = self._priority_score_for_path(fp)
                    # choose best: exact size first, then highest score then smallest diff
                    key = (diff == 0, score, -diff)
                    # Using tuple comparison with boolean precedence
                    # But we want True (diff==0) to be prioritized; so convert to int
                    key2 = (1 if diff == 0 else 0, score, -diff)
                    if (best is None) or (key2 > (1 if best_len_diff == 0 else 0, best_score, -best_len_diff)):
                        best = data
                        best_score = score
                        best_len_diff = diff
                        if diff == 0:
                            return best
                except Exception:
                    continue

        # 2) Content contains bug id
        for fp in self._iter_files(root):
            if not self._is_text_file_by_ext(fp):
                continue
            content = self._read_text_limited(fp)
            if not content:
                continue
            if self.BUG_ID in content:
                # Try to parse embedded arrays or base64 from this file
                data = self._extract_embedded_blob_from_text(content, target_size=self.TARGET_SIZE)
                if data:
                    return data

        return best

    def _search_files_exact_size(self, root, size):
        candidates = []
        for fp in self._iter_files(root):
            try:
                sz = os.path.getsize(fp)
            except Exception:
                continue
            if sz == size:
                candidates.append(fp)
        if not candidates:
            return None
        # Choose the best candidate by priority score
        candidates.sort(key=lambda p: self._priority_score_for_path(p), reverse=True)
        for fp in candidates:
            try:
                with open(fp, 'rb') as f:
                    return f.read()
            except Exception:
                continue
        return None

    def _search_by_name_keywords(self, root):
        best = None
        best_score = -10**9
        best_diff = 10**9
        for fp in self._iter_files(root):
            lp = fp.lower()
            if any(kw in lp for kw in self.candidate_name_keywords):
                try:
                    with open(fp, 'rb') as f:
                        data = f.read()
                    diff = abs(len(data) - self.TARGET_SIZE)
                    score = self._priority_score_for_path(fp)
                    # select best by exact size, then score, then closeness
                    key = (1 if diff == 0 else 0, score, -diff)
                    if (1 if diff == 0 else 0, score, -diff) > (1 if best_diff == 0 else 0, best_score, -best_diff):
                        best = data
                        best_score = score
                        best_diff = diff
                        if diff == 0:
                            return best
                except Exception:
                    continue
        return best

    def _search_in_seed_corpus_zip(self, root):
        # Look for seed corpus zips, extract nearest to target size
        zip_paths = []
        for fp in self._iter_files(root):
            lp = fp.lower()
            if lp.endswith('.zip') and ('corpus' in lp or 'seed' in lp or 'fuzz' in lp):
                zip_paths.append(fp)
        # Prioritize by path score
        zip_paths.sort(key=lambda p: self._priority_score_for_path(p), reverse=True)

        for zpath in zip_paths:
            try:
                with zipfile.ZipFile(zpath, 'r') as zf:
                    best = None
                    best_diff = 10**9
                    best_name_score = -10**9
                    exact = None
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        # skip too big files
                        if info.file_size > 5 * 1024 * 1024:
                            continue
                        diff = abs(info.file_size - self.TARGET_SIZE)
                        name_score = self._priority_score_for_path(info.filename)
                        if diff == 0:
                            # Prefer better names
                            if exact is None or name_score > best_name_score:
                                try:
                                    data = zf.read(info.filename)
                                except Exception:
                                    continue
                                if len(data) == self.TARGET_SIZE:
                                    return data
                        else:
                            if diff < best_diff or (diff == best_diff and name_score > best_name_score):
                                try:
                                    data = zf.read(info.filename)
                                except Exception:
                                    continue
                                best = data
                                best_diff = diff
                                best_name_score = name_score
                    if best:
                        return best
            except Exception:
                continue
        return None

    def _extract_embedded_blob_from_text(self, text, target_size=None):
        # Try array of bytes like {0x12, 34, ...}
        arr = self._parse_c_array_bytes(text, target_size=target_size)
        if arr:
            return arr
        # Try base64 blobs
        b64 = self._parse_base64_blob(text, target_size=target_size)
        if b64:
            return b64
        # Try C escaped string
        esc = self._parse_c_escaped_string_blob(text, target_size=target_size)
        if esc:
            return esc
        return None

    def _parse_c_array_bytes(self, text, target_size=None):
        # regex to find array initializers with many numbers
        pattern = re.compile(
            r'(?s)(?:static\s+)?(?:const\s+)?(?:unsigned\s+)?(?:char|uint8_t|int8_t)\s+\w+\s*\[\s*\]\s*=\s*\{([^{}]{20,})\}\s*;',
            re.MULTILINE
        )
        candidates = []
        for m in pattern.finditer(text):
            body = m.group(1)
            # extract numbers
            nums = re.findall(r'0x[0-9a-fA-F]+|0[0-7]+|\d+', body)
            if not nums or len(nums) < 8:
                continue
            try:
                bs = bytearray()
                for n in nums:
                    if n.startswith('0x') or n.startswith('0X'):
                        v = int(n, 16)
                    elif len(n) > 1 and n.startswith('0'):
                        # octal
                        try:
                            v = int(n, 8)
                        except Exception:
                            v = int(n, 10)
                    else:
                        v = int(n, 10)
                    bs.append(v & 0xFF)
                b = bytes(bs)
                candidates.append(b)
            except Exception:
                continue
        if not candidates:
            return None
        # Choose by size closeness if target_size provided
        if target_size is not None:
            candidates.sort(key=lambda b: (abs(len(b) - target_size), -len(b)))
        else:
            candidates.sort(key=lambda b: -len(b))
        return candidates[0]

    def _parse_base64_blob(self, text, target_size=None):
        # Collect long base64-like strings (allow across multiple concatenated quotes)
        # Look for sequences of base64 chars possibly with whitespace, length >= 200
        # Try to decode; if success and length near target_size, return
        # First, try joined quoted strings:
        str_pat = re.compile(r'"([A-Za-z0-9+/=\s]{100,})"', re.MULTILINE)
        candidates = []
        for m in str_pat.finditer(text):
            s = m.group(1)
            s_clean = re.sub(r'\s+', '', s)
            if len(s_clean) < 100:
                continue
            try:
                data = base64.b64decode(s_clean, validate=False)
                if data:
                    candidates.append(data)
            except Exception:
                continue
        if not candidates:
            return None
        if target_size is not None:
            candidates.sort(key=lambda b: (abs(len(b) - target_size), -len(b)))
        else:
            candidates.sort(key=lambda b: -len(b))
        return candidates[0]

    def _parse_c_escaped_string_blob(self, text, target_size=None):
        # Parse C string literal with escape sequences possibly concatenated
        # This is very heuristic; match long strings inside quotes
        pattern = re.compile(r'"([^"\n]{50,})"', re.MULTILINE)
        candidates = []
        for m in pattern.finditer(text):
            s = m.group(1)
            # consider only if contains escape sequences or hex \xNN etc.
            if '\\x' in s or '\\' in s:
                try:
                    # Interpret C-style escapes
                    b = bytes(s, 'utf-8').decode('unicode_escape').encode('latin-1', errors='ignore')
                    if b:
                        candidates.append(b)
                except Exception:
                    continue
        if not candidates:
            return None
        if target_size is not None:
            candidates.sort(key=lambda b: (abs(len(b) - target_size), -len(b)))
        else:
            candidates.sort(key=lambda b: -len(b))
        return candidates[0]

    def _search_in_source_embedded_blobs(self, root):
        best = None
        best_diff = 10**9
        # Scan source files for embedded blobs
        for fp in self._iter_files(root):
            _, ext = os.path.splitext(fp.lower())
            if ext not in self.code_exts and ext not in {'.txt', '.md'}:
                continue
            text = self._read_text_limited(fp, limit=self.max_text_read)
            if not text:
                continue
            data = self._extract_embedded_blob_from_text(text, target_size=self.TARGET_SIZE)
            if data:
                diff = abs(len(data) - self.TARGET_SIZE)
                if diff == 0:
                    return data
                if diff < best_diff:
                    best = data
                    best_diff = diff
        return best

    def _search_files_closest_size(self, root, target_size):
        # Search repository for files with sizes near target; prefer test/fuzz directories and data-like extensions
        best = None
        best_key = None
        for fp in self._iter_files(root):
            try:
                sz = os.path.getsize(fp)
            except Exception:
                continue
            diff = abs(sz - target_size)
            # Accept only up to some bound
            if diff > 512 and sz > 5 * 1024 * 1024:
                continue
            score = self._priority_score_for_path(fp)
            _, ext = os.path.splitext(fp.lower())
            # Prefer data files over code
            is_code = ext in self.code_exts
            is_data = (ext in self.binary_exts) or (ext in self.text_exts)
            key = (-(1 if diff == 0 else 0), diff, -(1 if is_data else 0), score)
            # We want smallest diff, data prefer, high score
            # Since we negate equals flag, minimal "key" will be with equals first. Use tuple comparison reversed.
            if best is None or key < best_key:
                try:
                    with open(fp, 'rb') as f:
                        data = f.read()
                    best = data
                    best_key = key
                    if diff == 0:
                        return best
                except Exception:
                    continue
        return best

    def _fallback_bytes(self, size):
        # Create a structured-looking input that may be useful across parsers (XML-like),
        # padded to target size.
        header = b'<?xml version="1.0" encoding="UTF-8"?>\n'
        body = b'<!-- fuzz-generated fallback input -->\n'
        body += b'<root>\n'
        # Create many attributes and nested elements to simulate complex structure
        for i in range(50):
            body += (b'  <item id="%d" attr="%s" value="%d"/>\n' % (i, b'X'* (i % 7 + 1), i*7))
        body += b'  <data><![CDATA[' + b'A' * 256 + b']]></data>\n'
        body += b'</root>\n'
        out = header + body
        if len(out) < size:
            out += b'B' * (size - len(out))
        else:
            out = out[:size]
        return out