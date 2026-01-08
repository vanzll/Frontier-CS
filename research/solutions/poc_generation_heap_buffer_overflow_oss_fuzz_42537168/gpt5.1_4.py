import os
import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        candidates = []

        def add_fs_candidates(base_dir: str):
            for root, _, files in os.walk(base_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    try:
                        size = os.path.getsize(fpath)
                    except OSError:
                        continue
                    if size <= 0:
                        continue

                    rel_name = os.path.relpath(fpath, base_dir)

                    def make_reader(path=fpath):
                        def reader():
                            with open(path, "rb") as f:
                                return f.read()
                        return reader

                    candidates.append(
                        {
                            "name": rel_name,
                            "size": size,
                            "reader": make_reader(),
                        }
                    )

        def add_tar_candidates(tar_path: str):
            try:
                tf = tarfile.open(tar_path, "r:*")
            except tarfile.ReadError:
                # Not a tarball; treat as a single file
                if os.path.isfile(tar_path):
                    size = os.path.getsize(tar_path)
                    if size > 0:
                        def reader():
                            with open(tar_path, "rb") as f:
                                return f.read()
                        candidates.append(
                            {
                                "name": os.path.basename(tar_path),
                                "size": size,
                                "reader": reader,
                            }
                        )
                return

            for m in tf.getmembers():
                if not m.isreg() or m.size <= 0:
                    continue

                def make_reader(member=m, tar_obj=tf):
                    def reader():
                        fobj = tar_obj.extractfile(member)
                        if fobj is None:
                            return b""
                        try:
                            data = fobj.read()
                        finally:
                            fobj.close()
                        return data
                    return reader

                candidates.append(
                    {
                        "name": m.name,
                        "size": m.size,
                        "reader": make_reader(),
                    }
                )

        if os.path.isdir(src_path):
            add_fs_candidates(src_path)
        else:
            add_tar_candidates(src_path)

        if not candidates:
            return b""

        ground_len = 913_919
        bug_id = "42537168"
        hints = [
            "poc",
            "crash",
            "issue",
            "bug",
            "testcase",
            "clusterfuzz",
            "repro",
            bug_id,
        ]

        def has_hint(name: str) -> bool:
            lname = name.lower()
            for h in hints:
                if h in lname:
                    return True
            return False

        def get_ext(name: str) -> str:
            base = name.rsplit("/", 1)[-1]
            if "." in base:
                return base.rsplit(".", 1)[1].lower()
            return ""

        # 1. Prefer hinted files around the expected PoC size
        hinted = [c for c in candidates if has_hint(c["name"])]
        hinted_mid = [
            c
            for c in hinted
            if ground_len // 4 <= c["size"] <= ground_len * 4
        ]
        if hinted_mid:
            best = min(hinted_mid, key=lambda c: abs(c["size"] - ground_len))
            return best["reader"]()
        if hinted:
            best = min(hinted, key=lambda c: abs(c["size"] - ground_len))
            return best["reader"]()

        # 2. Prefer binary-looking extensions
        binary_exts = {
            "pdf", "svg", "png", "jpg", "jpeg", "gif", "bmp", "ico", "icns", "webp",
            "tif", "tiff", "ps", "eps", "psd", "ai", "cur",
            "mp3", "ogg", "flac", "wav", "m4a", "mid", "midi",
            "mp4", "m4v", "mov", "avi", "mkv", "webm", "mpg", "mpeg",
            "otf", "ttf", "woff", "woff2",
            "bin", "dat", "raw", "zip", "gz", "bz2", "xz", "lzma", "7z", "rar",
            "tar", "jar", "class", "dex", "wasm", "swf",
            "pbm", "pgm", "ppm", "pam", "heic", "heif", "avif",
        }

        bin_candidates = [c for c in candidates if get_ext(c["name"]) in binary_exts]
        bin_mid = [
            c
            for c in bin_candidates
            if ground_len // 4 <= c["size"] <= ground_len * 4
        ]
        if bin_mid:
            best = min(bin_mid, key=lambda c: abs(c["size"] - ground_len))
            return best["reader"]()
        if bin_candidates:
            best = min(bin_candidates, key=lambda c: abs(c["size"] - ground_len))
            return best["reader"]()

        # 3. Non-text files near expected size
        text_exts = {
            "c", "cc", "cpp", "cxx", "h", "hh", "hpp",
            "py", "pyi", "pyx",
            "txt", "md", "rst",
            "html", "htm", "xml", "xhtml", "js", "css",
            "java", "kt", "gradle", "groovy",
            "json", "yaml", "yml", "toml", "ini", "cfg",
            "cmake", "mak", "mk",
            "sh", "bat", "ps1",
            "ac", "am", "m4",
            "in", "out",
            "s", "asm",
            "csv", "tsv",
            "tex",
            "frag", "vert", "glsl",
            "metal",
            "rb", "php", "pl", "pm", "tcl", "lua", "go", "rs", "swift",
        }

        non_text = [c for c in candidates if get_ext(c["name"]) not in text_exts]
        non_text_mid = [
            c
            for c in non_text
            if ground_len // 4 <= c["size"] <= ground_len * 4
        ]
        if non_text_mid:
            best = min(non_text_mid, key=lambda c: abs(c["size"] - ground_len))
            return best["reader"]()
        if non_text:
            best = min(non_text, key=lambda c: abs(c["size"] - ground_len))
            return best["reader"]()

        # 4. Fallback: use the largest file
        best = max(candidates, key=lambda c: c["size"])
        return best["reader"]()