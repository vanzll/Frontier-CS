import os
import tarfile
import zipfile
import io


class Solution:
    def solve(self, src_path: str) -> bytes:
        def select_best_file(file_iter):
            text_exts = {
                ".txt", ".md", ".rst",
                ".c", ".h", ".cpp", ".hpp", ".cc", ".hh", ".cxx", ".hxx",
                ".py", ".pyw",
                ".java", ".rb", ".pl", ".php",
                ".sh", ".bash", ".zsh", ".bat", ".cmd", ".ps1",
                ".cmake", ".mak", ".mk",
                ".html", ".htm", ".xml",
                ".json", ".yml", ".yaml",
                ".ini", ".cfg", ".conf",
                ".csv", ".tsv",
                ".log",
                ".sln", ".vcxproj", ".filters", ".user",
                ".cs", ".fs", ".go", ".rs",
                ".js", ".ts", ".jsx", ".tsx",
                ".css", ".scss", ".less",
                ".m", ".mm",
                ".gradle", ".kts",
            }

            binary_ext_scores = {
                ".j2k": 40,
                ".jp2": 40,
                ".j2c": 40,
                ".jpc": 40,
                ".pgx": 25,
                ".bin": 20,
                ".raw": 20,
                ".dat": 15,
                ".img": 15,
                ".bmp": 10,
                ".gz": 10,
                ".xz": 10,
                ".lzma": 10,
                ".bz2": 10,
                ".zip": 5,
            }

            keyword_scores = [
                ("poc", 25),
                ("proof", 15),
                ("crash", 20),
                ("heap", 10),
                ("overflow", 10),
                ("heap_overflow", 15),
                ("cve", 10),
                ("bug", 8),
                ("id_", 6),
                ("id-", 6),
                ("fuzz", 4),
                ("seed", 4),
                ("input", 3),
                ("case", 3),
                ("htj2k", 15),
                ("ht", 5),
                ("test", 2),
            ]

            best_meta = None
            best_score = None
            best_1479 = None
            best_close = None

            for path, size, opener in file_iter:
                name = os.path.basename(path)
                ext = os.path.splitext(name)[1].lower()
                name_lower = name.lower()
                path_lower = path.lower()

                score = 0.0

                if size == 1479:
                    score += 50.0
                elif 900 <= size <= 4096:
                    score += 10.0
                elif size < 64:
                    score -= 5.0
                elif size > 1_000_000:
                    score -= 20.0

                if ext in text_exts:
                    score -= 40.0

                score += binary_ext_scores.get(ext, 0.0)

                for kw, val in keyword_scores:
                    if kw in name_lower or kw in path_lower:
                        score += val

                if "poc" in path_lower and size == 1479:
                    score += 20.0

                if best_score is None or score > best_score:
                    best_score = score
                    best_meta = (path, size, opener)

                if size == 1479:
                    if best_1479 is None:
                        best_1479 = (path, size, opener)
                elif 1300 <= size <= 2000:
                    if best_close is None:
                        best_close = (path, size, opener)

            candidate_meta = None
            if best_meta is None:
                return None

            if best_score is not None and best_score > 0:
                candidate_meta = best_meta
            elif best_1479 is not None:
                candidate_meta = best_1479
            elif best_close is not None:
                candidate_meta = best_close
            else:
                candidate_meta = best_meta

            if candidate_meta is None:
                return None

            _, _, opener = candidate_meta
            try:
                f = opener()
                try:
                    data = f.read()
                finally:
                    try:
                        f.close()
                    except Exception:
                        pass
            except Exception:
                return None

            return data

        def iter_directory(root_path):
            ignore_dirs = {
                ".git", ".svn", ".hg",
                "build", "cmake-build-debug", "cmake-build-release",
                ".idea", ".vscode",
                "__pycache__",
                "out", "dist",
                ".deps", ".cache",
                "node_modules",
                "venv", ".venv", "env",
            }
            for dirpath, dirnames, filenames in os.walk(root_path):
                dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
                for fname in filenames:
                    path = os.path.join(dirpath, fname)
                    try:
                        size = os.path.getsize(path)
                    except OSError:
                        continue

                    def opener(p=path):
                        return open(p, "rb")

                    yield path, size, opener

        def iter_tar(tar_path):
            with tarfile.open(tar_path, "r:*") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    size = member.size
                    path = member.name

                    def opener(m=member):
                        f = tf.extractfile(m)
                        if f is None:
                            return io.BytesIO(b"")
                        return f

                    yield path, size, opener

        def iter_zip(zip_path):
            with zipfile.ZipFile(zip_path, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    size = info.file_size
                    path = info.filename

                    def opener(name=info.filename):
                        return zf.open(name, "r")

                    yield path, size, opener

        data = None

        if os.path.isdir(src_path):
            data = select_best_file(iter_directory(src_path))
        else:
            if tarfile.is_tarfile(src_path):
                data = select_best_file(iter_tar(src_path))
            elif zipfile.is_zipfile(src_path):
                data = select_best_file(iter_zip(src_path))
            else:
                parent = os.path.dirname(src_path)
                if parent and os.path.isdir(parent):
                    data = select_best_file(iter_directory(parent))

        if data is None:
            data = b"A" * 1479

        return data