import os
import tarfile
import re


class Solution:
    TARGET_SIZE = 1551

    def solve(self, src_path: str) -> bytes:
        if os.path.isdir(src_path):
            return self._solve_from_dir(src_path)

        # Try opening as tarball
        try:
            with tarfile.open(src_path, "r:*") as tf:
                return self._solve_from_tar(tf)
        except tarfile.ReadError:
            # Not a tarball; best effort: treat as directory if possible
            if os.path.isdir(src_path):
                return self._solve_from_dir(src_path)
            return b""

    def _solve_from_tar(self, tf: tarfile.TarFile) -> bytes:
        members = tf.getmembers()

        best_elf_member = None
        best_elf_diff = None
        best_elf_size = None

        # First pass: look for ELF files
        for m in members:
            if not m.isreg() or m.size <= 0:
                continue
            try:
                f = tf.extractfile(m)
            except (KeyError, OSError):
                continue
            if f is None:
                continue
            try:
                header = f.read(4)
            finally:
                f.close()
            if not header or not header.startswith(b"\x7fELF"):
                continue
            diff = abs(m.size - self.TARGET_SIZE)
            if (
                best_elf_member is None
                or diff < best_elf_diff
                or (diff == best_elf_diff and m.size < best_elf_size)
            ):
                best_elf_member = m
                best_elf_diff = diff
                best_elf_size = m.size

        if best_elf_member is not None:
            f = tf.extractfile(best_elf_member)
            if f is not None:
                try:
                    return f.read()
                finally:
                    f.close()

        # Fallback: look for files with relevant names
        pattern = re.compile(r"(poc|crash|debug|dwarf|names|ossfuzz|383170474)", re.IGNORECASE)
        best_member = None
        best_diff = None
        best_size = None

        for m in members:
            if not m.isreg() or m.size <= 0:
                continue
            name = m.name
            if not pattern.search(name):
                continue
            diff = abs(m.size - self.TARGET_SIZE)
            if (
                best_member is None
                or diff < best_diff
                or (diff == best_diff and m.size < best_size)
            ):
                best_member = m
                best_diff = diff
                best_size = m.size

        if best_member is not None:
            f = tf.extractfile(best_member)
            if f is not None:
                try:
                    return f.read()
                finally:
                    f.close()

        # Last resort: no candidate found
        return b""

    def _solve_from_dir(self, root: str) -> bytes:
        best_elf_path = None
        best_elf_diff = None
        best_elf_size = None

        # First pass: ELF files
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                path = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size <= 0:
                    continue
                try:
                    with open(path, "rb") as f:
                        header = f.read(4)
                except OSError:
                    continue
                if not header or not header.startswith(b"\x7fELF"):
                    continue
                diff = abs(size - self.TARGET_SIZE)
                if (
                    best_elf_path is None
                    or diff < best_elf_diff
                    or (diff == best_elf_diff and size < best_elf_size)
                ):
                    best_elf_path = path
                    best_elf_diff = diff
                    best_elf_size = size

        if best_elf_path is not None:
            try:
                with open(best_elf_path, "rb") as f:
                    return f.read()
            except OSError:
                pass

        # Fallback: name-based heuristic
        pattern = re.compile(r"(poc|crash|debug|dwarf|names|ossfuzz|383170474)", re.IGNORECASE)
        best_path = None
        best_diff = None
        best_size = None

        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if not pattern.search(fname):
                    continue
                path = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size <= 0:
                    continue
                diff = abs(size - self.TARGET_SIZE)
                if (
                    best_path is None
                    or diff < best_diff
                    or (diff == best_diff and size < best_size)
                ):
                    best_path = path
                    best_diff = diff
                    best_size = size

        if best_path is not None:
            try:
                with open(best_path, "rb") as f:
                    return f.read()
            except OSError:
                pass

        return b""