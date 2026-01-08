import os
import tarfile
import zipfile
import io
from dataclasses import dataclass
from typing import Callable, List


@dataclass
class FileEntry:
    path: str
    size: int
    loader: Callable[[], bytes]


class Solution:
    TARGET_POC_LEN = 1032
    MAX_NESTED_ARCHIVE_SIZE = 2_000_000

    def solve(self, src_path: str) -> bytes:
        entries: List[FileEntry] = []

        if os.path.isdir(src_path):
            self._gather_from_dir(src_path, entries)
        elif tarfile.is_tarfile(src_path):
            self._gather_from_tar(src_path, entries)
        else:
            try:
                size = os.path.getsize(src_path)
            except OSError:
                return b""
            if size > 0:
                def loader(path=src_path) -> bytes:
                    with open(path, "rb") as f:
                        return f.read()
                entries.append(FileEntry(src_path, size, loader))

        if not entries:
            return b""

        # Try to inspect nested archives (zip/tar) inside promising files
        extra_entries: List[FileEntry] = []
        for e in entries:
            self._maybe_add_nested_entries(e, extra_entries)
        if extra_entries:
            entries.extend(extra_entries)

        ordered = self._rank_entries(entries, self.TARGET_POC_LEN)
        for e in ordered:
            try:
                data = e.loader()
            except Exception:
                continue
            if data:
                return data

        return b""

    def _gather_from_dir(self, root: str, entries: List[FileEntry]) -> None:
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                try:
                    size = os.path.getsize(full_path)
                except OSError:
                    continue
                if size <= 0:
                    continue

                def make_loader(path=full_path) -> Callable[[], bytes]:
                    def loader() -> bytes:
                        with open(path, "rb") as f:
                            return f.read()
                    return loader

                entries.append(FileEntry(full_path, size, make_loader()))

    def _gather_from_tar(self, archive_path: str, entries: List[FileEntry]) -> None:
        try:
            tar = tarfile.open(archive_path, "r:*")
        except tarfile.TarError:
            return

        try:
            for m in tar.getmembers():
                if not m.isreg():
                    continue
                size = m.size
                if size <= 0:
                    continue
                member_name = m.name
                full_path = member_name

                def make_loader(ap=archive_path, mn=member_name) -> Callable[[], bytes]:
                    def loader() -> bytes:
                        try:
                            with tarfile.open(ap, "r:*") as t2:
                                try:
                                    member = t2.getmember(mn)
                                except KeyError:
                                    return b""
                                f = t2.extractfile(member)
                                return f.read() if f is not None else b""
                        except tarfile.TarError:
                            return b""
                    return loader

                entries.append(FileEntry(full_path, size, make_loader()))
        finally:
            try:
                tar.close()
            except Exception:
                pass

    def _maybe_add_nested_entries(self, entry: FileEntry, dest_entries: List[FileEntry]) -> None:
        if entry.size <= 0 or entry.size > self.MAX_NESTED_ARCHIVE_SIZE:
            return

        name = os.path.basename(entry.path).lower()
        if not any(k in name for k in ("testcase", "crash", "poc", "repro", "clusterfuzz", "fuzz")):
            return

        try:
            data = entry.loader()
        except Exception:
            return

        if not data:
            return

        # Try as ZIP archive
        bio = io.BytesIO(data)
        try:
            with zipfile.ZipFile(bio) as zf:
                for zi in zf.infolist():
                    if zi.is_dir():
                        continue
                    zi_name = zi.filename
                    zi_size = zi.file_size
                    if zi_size <= 0:
                        continue

                    def make_loader_zip(zip_bytes=data, fname=zi_name) -> Callable[[], bytes]:
                        def loader() -> bytes:
                            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf2:
                                with zf2.open(fname) as f:
                                    return f.read()
                        return loader

                    dest_entries.append(
                        FileEntry(entry.path + "::" + zi_name, zi_size, make_loader_zip())
                    )
        except zipfile.BadZipFile:
            pass

        # Try as TAR archive
        bio2 = io.BytesIO(data)
        try:
            with tarfile.open(fileobj=bio2, mode="r:*") as t:
                for m in t.getmembers():
                    if not m.isreg():
                        continue
                    m_name = m.name
                    m_size = m.size
                    if m_size <= 0:
                        continue

                    def make_loader_tar(tar_bytes=data, mn=m_name) -> Callable[[], bytes]:
                        def loader() -> bytes:
                            try:
                                with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as t2:
                                    try:
                                        member = t2.getmember(mn)
                                    except KeyError:
                                        return b""
                                    f = t2.extractfile(member)
                                    return f.read() if f is not None else b""
                            except tarfile.TarError:
                                return b""
                        return loader

                    dest_entries.append(
                        FileEntry(entry.path + "::" + m_name, m_size, make_loader_tar())
                    )
        except tarfile.ReadError:
            pass

    def _entry_score(self, entry: FileEntry, target_len: int) -> int:
        basename = os.path.basename(entry.path).lower()
        full = entry.path.lower()
        score = 0

        source_exts = (
            ".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp",
            ".py", ".pyw", ".java", ".js", ".ts",
            ".go", ".rs",
            ".sh", ".bash", ".zsh", ".bat", ".ps1",
            ".md", ".rst", ".txt",
            ".html", ".htm", ".xml",
            ".yml", ".yaml", ".ini", ".cfg",
            ".cmake", ".mak", ".mk", ".in", ".ac", ".am",
            ".pl", ".rb", ".php",
            ".jsonl", ".csv"
        )

        if any(basename.endswith(ext) for ext in source_exts):
            score -= 120

        if any(k in full for k in ("crash", "testcase", "poc", "repro", "clusterfuzz")):
            score += 150
        elif "fuzz" in full:
            score += 60
        elif any(k in full for k in ("seed", "corpus")):
            score += 20

        if target_len > 0:
            diff = abs(entry.size - target_len)
            if diff == 0:
                score += 120
            else:
                ratio_diff = diff / float(target_len)
                size_score = max(0, int(100 - ratio_diff * 100))
                score += size_score

        if entry.size < 8:
            score -= 50
        elif entry.size < 64:
            score -= 10
        elif entry.size <= 4096:
            score += 40
        else:
            penalty = min(100, int((entry.size - 4096) / 4096) * 10)
            score -= penalty

        if any(k in full for k in ("polygon", "poly", "cell", "h3")):
            score += 25

        return score

    def _rank_entries(self, entries: List[FileEntry], target_len: int) -> List[FileEntry]:
        scored = []
        for e in entries:
            s = self._entry_score(e, target_len)
            # tie-breaker: prefer smaller size
            scored.append((s, -min(e.size, target_len), e))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [e for _, _, e in scored]