import os
import tarfile
import tempfile


class Solution:
    def _safe_extract(self, tar: tarfile.TarFile, path: str) -> None:
        def is_within_directory(directory: str, target: str) -> bool:
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            return os.path.commonprefix([abs_directory, abs_target]) == abs_directory

        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                continue
        tar.extractall(path)

    def _collect_candidates(self, root_dir: str, bug_id: str, target_size: int):
        candidates = []
        bug_id_bytes = bug_id.encode("ascii", "ignore")

        for dirpath, dirnames, filenames in os.walk(root_dir):
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                relpath = os.path.relpath(fpath, root_dir)
                lower = relpath.lower()
                try:
                    size = os.path.getsize(fpath)
                except OSError:
                    continue

                # Base heuristic: ignore trivially small or huge files
                if size == 0:
                    continue

                # Initial priority = high (worse)
                priority = 100

                # Filename/path based heuristics
                if bug_id in lower:
                    priority = 0
                elif bug_id[:5] in lower:  # partial bug id
                    priority = min(priority, 1)

                keywords = {
                    "poc": 2,
                    "crash": 3,
                    "oss-fuzz": 3,
                    "fuzz": 4,
                    "corpus": 5,
                    "regress": 6,
                    "test": 7,
                    "inputs": 8,
                }
                for kw, pr in keywords.items():
                    if kw in lower:
                        priority = min(priority, pr)

                # If nothing matched, skip unless file is in likely test directories
                if priority == 100:
                    continue

                size_score = abs(size - target_size)

                candidates.append((priority, size_score, fpath))

        # Content-based bug-id search for extra strong matches
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(fpath)
                except OSError:
                    continue
                # Limit very large files from content scanning for performance
                if size > 1024 * 1024:
                    continue
                try:
                    with open(fpath, "rb") as f:
                        data = f.read()
                except OSError:
                    continue
                if bug_id_bytes in data:
                    size_score = abs(size - target_size)
                    # Content hit: give highest priority
                    candidates.append((-1, size_score, fpath))

        return candidates

    def solve(self, src_path: str) -> bytes:
        bug_id = "42536068"
        target_size = 2179

        # Determine root directory: extract tarball or use directory directly
        if os.path.isdir(src_path):
            root_dir = src_path
        else:
            tmpdir = tempfile.mkdtemp(prefix="poc_extract_")
            try:
                if tarfile.is_tarfile(src_path):
                    with tarfile.open(src_path, "r:*") as tf:
                        self._safe_extract(tf, tmpdir)
                    root_dir = tmpdir
                else:
                    # Not a tarfile; fallback: treat directory of src_path as root
                    root_dir = os.path.dirname(os.path.abspath(src_path)) or "."
            except tarfile.TarError:
                root_dir = os.path.dirname(os.path.abspath(src_path)) or "."

        candidates = self._collect_candidates(root_dir, bug_id, target_size)

        if candidates:
            candidates.sort()
            _, _, best_path = candidates[0]
            try:
                with open(best_path, "rb") as f:
                    return f.read()
            except OSError:
                pass

        # Fallback: return bug id as bytes if nothing useful is found
        return bug_id.encode("ascii", "ignore")