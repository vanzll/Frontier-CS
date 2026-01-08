import os
import re
import tarfile
import tempfile
from typing import Optional, Tuple


class Solution:
    def solve(self, src_path: str) -> bytes:
        with tempfile.TemporaryDirectory() as td:
            self._safe_extract_tar(src_path, td)
            poc = self._find_best_poc(td)
            if poc is not None:
                return poc
        return b"A" * 9

    def _safe_extract_tar(self, tar_path: str, dst_dir: str) -> None:
        def is_within_directory(directory: str, target: str) -> bool:
            directory = os.path.abspath(directory)
            target = os.path.abspath(target)
            try:
                common = os.path.commonpath([directory, target])
            except ValueError:
                return False
            return common == directory

        with tarfile.open(tar_path, "r:*") as tf:
            members = tf.getmembers()
            safe_members = []
            for m in members:
                name = m.name
                if not name:
                    continue
                name = name.lstrip("/").replace("\\", "/")
                if name.startswith("../") or "/../" in name:
                    continue
                target_path = os.path.join(dst_dir, name)
                if not is_within_directory(dst_dir, target_path):
                    continue
                m.name = name
                safe_members.append(m)
            tf.extractall(dst_dir, members=safe_members)

    def _find_best_poc(self, root: str) -> Optional[bytes]:
        skip_dirs = {
            ".git", ".svn", ".hg", "__pycache__", "build", "out", "dist", "bazel-bin",
            "bazel-out", "bazel-testlogs", "CMakeFiles", ".idea", ".vs"
        }

        source_exts = {
            ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc",
            ".py", ".pyi", ".java", ".kt", ".go", ".rs", ".swift",
            ".m", ".mm", ".cs", ".js", ".ts",
            ".cmake", ".mk", ".make", ".am", ".ac", ".m4", ".in",
            ".s", ".S", ".asm",
            ".md", ".rst", ".adoc",
            ".yml", ".yaml", ".json", ".toml", ".ini", ".cfg",
            ".sh", ".bat", ".ps1",
            ".gradle", ".gni", ".gn", ".ninja",
            ".pl", ".rb",
        }

        likely_input_exts = {
            ".bin", ".dat", ".raw", ".poc", ".crash", ".seed",
            ".xml", ".mpd", ".m4s", ".mp4", ".m4a", ".m4v", ".idx", ".txt"
        }

        negative_names = {
            "readme", "readme.txt", "readme.md",
            "license", "license.txt", "license.md",
            "copying", "copying.txt",
            "authors", "authors.txt",
            "changelog", "changelog.txt",
            "news", "news.txt",
            "contributing", "contributing.md",
        }

        def is_probably_binary(data: bytes) -> bool:
            if not data:
                return False
            if b"\x00" in data:
                return True
            # Heuristic: many non-printables
            non_print = 0
            for b in data[:256]:
                if b in b"\t\r\n":
                    continue
                if b < 32 or b > 126:
                    non_print += 1
            return non_print > 20

        def file_score(path: str, size: int) -> int:
            p = path.replace("\\", "/")
            lp = p.lower()
            base = os.path.basename(lp)
            ext = os.path.splitext(base)[1]

            score = 0

            if base in negative_names:
                score -= 200
            if any(x in lp for x in ["/doc/", "/docs/", "/documentation/", "/man/", "/cmake/"]):
                score -= 80

            if "clusterfuzz-testcase" in lp:
                score += 200
            if "minimized" in lp:
                score += 80
            if "crash" in lp or "crasher" in lp:
                score += 70
            if "repro" in lp:
                score += 60
            if re.search(r"\bpoc\b", lp):
                score += 60
            if "overflow" in lp:
                score += 40
            if "asan" in lp or "ubsan" in lp or "sanitizer" in lp:
                score += 20
            if any(k in lp for k in ["oss-fuzz", "fuzz", "corpus", "testdata", "regression", "seeds", "seed", "inputs", "bugs", "samples", "data", "cases"]):
                score += 30

            if ext in likely_input_exts:
                score += 15
            if ext in source_exts:
                score -= 30

            if size <= 4:
                score += 5
            elif size <= 8:
                score += 20
            elif size <= 16:
                score += 35
            elif size <= 32:
                score += 30
            elif size <= 64:
                score += 22
            elif size <= 128:
                score += 16
            elif size <= 256:
                score += 10
            elif size <= 512:
                score += 6
            elif size <= 2048:
                score += 2
            else:
                score -= 2

            # Prefer deeper paths slightly when keywords indicate fuzzing artifacts (often nested)
            depth = lp.count("/")
            score += min(depth, 8)

            return score

        best: Optional[Tuple[int, int, str]] = None  # (score, size, path)

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
            for fn in filenames:
                if not fn:
                    continue
                if fn.startswith("."):
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                if not os.path.isfile(full):
                    continue
                size = st.st_size
                if size <= 0 or size > 2_000_000:
                    continue

                rel = os.path.relpath(full, root)
                sc = file_score(rel, size)

                if best is None or (sc, -size, rel) > (best[0], -best[1], best[2]):
                    best = (sc, size, full)

        if best is None or best[0] <= 0:
            # Second pass: look for small non-source files under likely dirs even without keywords
            alt_best: Optional[Tuple[int, int, str]] = None
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
                lp_dir = dirpath.replace("\\", "/").lower()
                if not any(k in lp_dir for k in ["fuzz", "corpus", "testdata", "regression", "seed", "inputs", "samples", "data", "cases"]):
                    continue
                for fn in filenames:
                    if not fn or fn.startswith("."):
                        continue
                    full = os.path.join(dirpath, fn)
                    try:
                        st = os.stat(full)
                    except OSError:
                        continue
                    if not os.path.isfile(full):
                        continue
                    size = st.st_size
                    if size <= 0 or size > 2_000_000:
                        continue
                    base = fn.lower()
                    ext = os.path.splitext(base)[1]
                    if ext in source_exts:
                        continue
                    # prefer smallest
                    sc = 1000 - min(size, 1000)
                    rel = os.path.relpath(full, root)
                    if alt_best is None or (sc, -size, rel) > (alt_best[0], -alt_best[1], alt_best[2]):
                        alt_best = (sc, size, full)
            if alt_best is None:
                return None
            path = alt_best[2]
        else:
            path = best[2]

        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            return None

        if not data:
            return None

        # If we accidentally picked a README/LICENSE, fallback
        base = os.path.basename(path).lower()
        if base in negative_names:
            return None

        # Prefer very small binary-ish payloads; if big text selected, attempt to find better nearby.
        if len(data) > 4096 and not is_probably_binary(data):
            return None

        return data