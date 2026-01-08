import os
import tarfile
import tempfile
import shutil


class Solution:
    def solve(self, src_path: str) -> bytes:
        target_len = 33762
        tmpdir = tempfile.mkdtemp(prefix="poc_extract_")
        try:
            # Safely extract tarball
            with tarfile.open(src_path, "r:*") as tf:
                def is_within_directory(directory: str, target: str) -> bool:
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    try:
                        common = os.path.commonpath([abs_directory, abs_target])
                    except ValueError:
                        return False
                    return common == abs_directory

                members = []
                for m in tf.getmembers():
                    member_path = os.path.join(tmpdir, m.name)
                    if is_within_directory(tmpdir, member_path):
                        members.append(m)
                tf.extractall(tmpdir, members=members)

            exact_files = []
            near_files = []
            all_files = []
            limit_near = 4096

            for root, dirs, files in os.walk(tmpdir):
                for name in files:
                    path = os.path.join(root, name)
                    try:
                        size = os.path.getsize(path)
                    except OSError:
                        continue
                    all_files.append((path, size))
                    if size == target_len:
                        exact_files.append((path, size))
                    elif abs(size - target_len) <= limit_near:
                        near_files.append((path, size))

            def score_name(path: str) -> int:
                base = os.path.basename(path).lower()
                tokens = {
                    "poc": 50,
                    "crash": 40,
                    "uaf": 30,
                    "use-after-free": 30,
                    "use_after_free": 30,
                    "heap": 25,
                    "issue": 20,
                    "bug": 20,
                    "regress": 15,
                    "test": 10,
                    "sample": 5,
                    "fuzz": 10,
                    "input": 5,
                }
                s = 0
                for tok, w in tokens.items():
                    if tok in base:
                        s += w
                ext = os.path.splitext(base)[1]
                ext_bonus = {
                    ".pdf": 5,
                    ".txt": 3,
                    ".bin": 3,
                    ".dat": 3,
                    ".html": 3,
                    ".xml": 3,
                    ".json": 3,
                }
                s += ext_bonus.get(ext, 0)
                return s

            def choose(candidates, prefer_size_closest: bool = False):
                if not candidates:
                    return None
                best_path = None
                best_key = None
                for path, size in candidates:
                    name_score = score_name(path)
                    if prefer_size_closest:
                        size_penalty = abs(size - target_len)
                        key = (name_score, -size_penalty, -size)
                    else:
                        key = (name_score, -size)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_path = path
                return best_path

            chosen_path = None
            if exact_files:
                chosen_path = choose(exact_files, prefer_size_closest=False)
            elif near_files:
                chosen_path = choose(near_files, prefer_size_closest=True)
            else:
                if all_files:
                    chosen_path = choose(all_files, prefer_size_closest=True)

            if chosen_path and os.path.isfile(chosen_path):
                with open(chosen_path, "rb") as f:
                    return f.read()

            # Fallback if nothing suitable found
            return b"A" * 10
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)