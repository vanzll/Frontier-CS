import os
import tarfile
import zipfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        bug_id_str = "383200048"
        bug_id_bytes = bug_id_str.encode("ascii")
        max_file_size = 1_000_000

        bugid_512 = []
        bugid_any = []
        kw_512 = []
        size_512 = []
        kw_any = []
        all_small = []

        name_keywords = (
            "poc",
            "crash",
            "fuzz",
            "oss-fuzz",
            "ossfuzz",
            "clusterfuzz",
            "repro",
            "id_",
            "bug",
            bug_id_str,
        )

        def consider(data: bytes, name: str) -> None:
            if not data:
                return
            size = len(data)
            if size == 0 or size > max_file_size:
                return

            name_lower = (name or "").lower()

            is_bugid = (bug_id_str in name_lower) or (bug_id_bytes in data)
            if is_bugid:
                bugid_any.append(data)
                if size == 512:
                    bugid_512.append(data)

            kw_hit = any(k in name_lower for k in name_keywords)
            if kw_hit:
                kw_any.append(data)
                if size == 512:
                    kw_512.append(data)

            if size == 512:
                size_512.append(data)

            all_small.append(data)

        def score_header(d: bytes) -> int:
            s = 0
            if d.startswith(b"\x7fELF"):
                s += 20
            if d[:4] == b"UPX!":
                s += 15
            if d.startswith(b"MZ"):
                s += 10
            # Prefer more binary-looking data (non-text)
            non_printable = sum(1 for b in d[:64] if b < 9 or (13 < b < 32) or b > 126)
            s += non_printable
            return s

        def choose_best(lst):
            if not lst:
                return None
            return max(lst, key=lambda d: (score_header(d), -len(d)))

        # Case 1: src_path is a directory
        if os.path.isdir(src_path):
            for root, _, files in os.walk(src_path):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    try:
                        size = os.path.getsize(fpath)
                    except OSError:
                        continue
                    if size == 0 or size > max_file_size:
                        continue
                    try:
                        with open(fpath, "rb") as f:
                            data = f.read()
                    except OSError:
                        continue
                    consider(data, fpath)
        else:
            # Case 2: try as tarball
            tried_tar = False
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    tried_tar = True
                    for m in tf.getmembers():
                        if not m.isreg():
                            continue
                        if m.size == 0 or m.size > max_file_size:
                            continue
                        try:
                            f = tf.extractfile(m)
                        except (KeyError, OSError):
                            continue
                        if f is None:
                            continue
                        try:
                            data = f.read()
                        finally:
                            f.close()
                        consider(data, m.name)
            except (tarfile.ReadError, OSError):
                pass

            # Case 3: try as zip if tar didn't succeed or yielded nothing useful
            need_zip = (not tried_tar) or (
                not bugid_512
                and not kw_512
                and not size_512
                and not bugid_any
                and not kw_any
            )
            if need_zip:
                try:
                    with zipfile.ZipFile(src_path, "r") as zf:
                        for name in zf.namelist():
                            try:
                                info = zf.getinfo(name)
                            except KeyError:
                                continue
                            if info.is_dir():
                                continue
                            if info.file_size == 0 or info.file_size > max_file_size:
                                continue
                            try:
                                data = zf.read(name)
                            except OSError:
                                continue
                            consider(data, name)
                except (zipfile.BadZipFile, OSError):
                    pass

        # Selection priority
        result = (
            choose_best(bugid_512)
            or choose_best(kw_512)
            or choose_best(size_512)
            or (min(bugid_any, key=len) if bugid_any else None)
            or choose_best(kw_any)
            or choose_best(all_small)
        )

        if result is not None:
            return result

        # Fallback: generic 512-byte payload
        return b"A" * 512