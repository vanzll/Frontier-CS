import os
import re
import tarfile
import tempfile
import shutil


class Solution:
    def solve(self, src_path: str) -> bytes:
        base_len = 9
        length = base_len
        size_threshold = 0
        var_conds = {}  # var_name -> list of (idx, op, char)
        tmpdir = None

        def add_cond(var_name, idx, op, ch):
            lst = var_conds.setdefault(var_name, [])
            lst.append((idx, op, ch))

        try:
            if os.path.isdir(src_path):
                root = src_path
            else:
                tmpdir = tempfile.mkdtemp()
                try:
                    with tarfile.open(src_path, "r:*") as tf:
                        tf.extractall(tmpdir)
                    root = tmpdir
                except Exception:
                    return b"A" * base_len

            exts = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp")

            # Regexes
            char_cmp_re = re.compile(
                r"([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(\d+)\s*\]\s*([!=]=)\s*'([^']*)'"
            )
            if_cond_re = re.compile(
                r"if\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*([<>]=?)\s*(\d+)\s*\)"
            )
            memcmp_re1 = re.compile(
                r"(?:memcmp|strncmp)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*\"([^\"\\]*)\"\s*,\s*(\d+)\s*\)"
            )
            memcmp_re2 = re.compile(
                r"(?:memcmp|strncmp)\s*\(\s*\"([^\"\\]*)\"\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)"
            )

            for dirpath, _, filenames in os.walk(root):
                for fname in filenames:
                    if not fname.endswith(exts):
                        continue
                    fpath = os.path.join(dirpath, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                        # Character index comparisons like data[0] != 'A'
                        for m in char_cmp_re.finditer(text):
                            var, idx_str, op, lit = m.groups()
                            if not lit:
                                continue
                            idx = int(idx_str)
                            ch = lit[0]
                            add_cond(var, idx, op, ch)

                        # Size/length gating
                        for m in if_cond_re.finditer(text):
                            var, op, num = m.groups()
                            if "size" not in var.lower() and "len" not in var.lower():
                                continue
                            n = int(num)
                            if op == "<":
                                t = n
                            elif op == "<=":
                                t = n + 1
                            else:
                                continue
                            if 0 < t <= 64 and t > size_threshold:
                                size_threshold = t

                        # memcmp/strncmp patterns: memcmp(data, "TAG", 3)
                        for m in memcmp_re1.finditer(text):
                            var, s, n_str = m.groups()
                            try:
                                n = int(n_str)
                            except ValueError:
                                continue
                            if n <= 0:
                                continue
                            for i in range(min(n, len(s))):
                                add_cond(var, i, "==", s[i])

                        # memcmp/strncmp patterns: memcmp("TAG", data, 3)
                        for m in memcmp_re2.finditer(text):
                            s, var, n_str = m.groups()
                            try:
                                n = int(n_str)
                            except ValueError:
                                continue
                            if n <= 0:
                                continue
                            for i in range(min(n, len(s))):
                                add_cond(var, i, "==", s[i])

                    except Exception:
                        continue

            if size_threshold > length:
                length = size_threshold
        finally:
            if tmpdir is not None and os.path.isdir(tmpdir):
                shutil.rmtree(tmpdir, ignore_errors=True)

        if length < base_len:
            length = base_len

        arr = bytearray(b"A" * length)

        if var_conds:
            def score_var(name, conds):
                lname = name.lower()
                s = len(conds)
                if "data" in lname:
                    s += 20
                if "input" in lname:
                    s += 15
                if "buf" in lname or "str" in lname:
                    s += 10
                max_idx = max((c[0] for c in conds), default=0)
                s += max_idx / 5.0
                return s

            best_name = max(var_conds.items(), key=lambda kv: score_var(kv[0], kv[1]))[0]
            for idx, op, ch in var_conds[best_name]:
                if 0 <= idx < len(arr):
                    # Treat both == and != as equality to satisfy typical gating like if (data[i] != 'X') return;
                    arr[idx] = ord(ch)

        return bytes(arr)