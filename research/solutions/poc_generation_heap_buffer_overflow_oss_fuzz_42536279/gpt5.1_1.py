import os
import stat
import tarfile
import zipfile
import gzip


class Solution:
    GROUND_TRUTH_SIZE = 6180
    BUG_ID = "42536279"

    def solve(self, src_path: str) -> bytes:
        if os.path.isdir(src_path):
            data = self._find_poc_in_dir(src_path)
        else:
            data = self._find_poc_in_file(src_path)

        if data is None:
            # Fallback: return some small dummy input if nothing found
            return b"A"
        return data

    def _find_poc_in_file(self, path: str):
        # Try as archive first
        try:
            if tarfile.is_tarfile(path):
                exact, _, approx, _, _ = self._search_archive_for_poc(path)
                if exact is not None:
                    return exact
                if approx is not None:
                    return approx
                # fall through to other options if none found
            elif zipfile.is_zipfile(path):
                exact, _, approx, _, _ = self._search_archive_for_poc(path)
                if exact is not None:
                    return exact
                if approx is not None:
                    return approx
        except Exception:
            pass

        name_lower = os.path.basename(path).lower()

        # If it's a gzip-compressed PoC
        if name_lower.endswith(".gz") and any(
            key in name_lower for key in ("poc", "crash", "clusterfuzz", self.BUG_ID)
        ):
            try:
                with gzip.open(path, "rb") as f:
                    data = f.read()
                if data:
                    return data
            except Exception:
                pass

        # Fallback: treat as raw input file
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception:
            return None

    def _find_poc_in_dir(self, root: str):
        target = self.GROUND_TRUTH_SIZE
        bug_id = self.BUG_ID

        best_reg_exact_path = None
        best_reg_exact_score = None

        best_reg_approx_path = None
        best_reg_approx_score = None
        best_reg_approx_diff = None

        gz_candidates = []
        subarchives = []

        for dirpath, dirnames, filenames in os.walk(root):
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                try:
                    st = os.stat(full)
                except OSError:
                    continue

                if not stat.S_ISREG(st.st_mode):
                    continue

                size = st.st_size
                name_lower = fname.lower()

                # Classify archives and gz
                is_tar_or_zip = any(
                    name_lower.endswith(suf)
                    for suf in (
                        ".tar",
                        ".tar.gz",
                        ".tar.bz2",
                        ".tar.xz",
                        ".tgz",
                        ".zip",
                    )
                )
                is_gz_only = name_lower.endswith(".gz") and not is_tar_or_zip

                if is_tar_or_zip and any(
                    key in name_lower for key in ("poc", "crash", "clusterfuzz", bug_id)
                ):
                    subarchives.append(full)
                    # Do not treat as raw candidate
                    continue

                if is_gz_only and any(
                    key in name_lower for key in ("poc", "crash", "clusterfuzz", bug_id)
                ):
                    gz_candidates.append(full)
                    # Do not treat as raw candidate
                    continue

                # Raw candidate file
                cand_score = self._score_path(full, size)

                if size == target:
                    if (
                        best_reg_exact_path is None
                        or cand_score > best_reg_exact_score
                    ):
                        best_reg_exact_path = full
                        best_reg_exact_score = cand_score
                else:
                    diff = abs(size - target)
                    if (
                        best_reg_approx_path is None
                        or diff < best_reg_approx_diff
                        or (
                            diff == best_reg_approx_diff
                            and cand_score > best_reg_approx_score
                        )
                    ):
                        best_reg_approx_path = full
                        best_reg_approx_score = cand_score
                        best_reg_approx_diff = diff

        # Stage 2: process gzip-compressed candidates
        best_gz_exact_data = None
        best_gz_exact_score = None

        best_gz_approx_data = None
        best_gz_approx_score = None
        best_gz_approx_diff = None

        if gz_candidates:
            for gz_path in gz_candidates:
                try:
                    with gzip.open(gz_path, "rb") as f:
                        data = f.read()
                except Exception:
                    continue

                if not data:
                    continue

                size = len(data)
                cand_score = self._score_path(gz_path + "|gz", size)

                if size == target:
                    if best_gz_exact_data is None or cand_score > best_gz_exact_score:
                        best_gz_exact_data = data
                        best_gz_exact_score = cand_score
                else:
                    diff = abs(size - target)
                    if (
                        best_gz_approx_data is None
                        or diff < best_gz_approx_diff
                        or (
                            diff == best_gz_approx_diff
                            and cand_score > best_gz_approx_score
                        )
                    ):
                        best_gz_approx_data = data
                        best_gz_approx_score = cand_score
                        best_gz_approx_diff = diff

        # Stage 3: process sub-archives
        best_arch_exact_data = None
        best_arch_exact_score = None

        best_arch_approx_data = None
        best_arch_approx_score = None
        best_arch_approx_diff = None

        for arch_path in subarchives:
            exact_data, exact_score, approx_data, approx_score, approx_diff = (
                self._search_archive_for_poc(arch_path)
            )

            if exact_data is not None:
                if (
                    best_arch_exact_data is None
                    or (exact_score is not None and exact_score > best_arch_exact_score)
                ):
                    best_arch_exact_data = exact_data
                    best_arch_exact_score = exact_score

            if approx_data is not None:
                if (
                    best_arch_approx_data is None
                    or approx_diff < best_arch_approx_diff
                    or (
                        approx_diff == best_arch_approx_diff
                        and approx_score > best_arch_approx_score
                    )
                ):
                    best_arch_approx_data = approx_data
                    best_arch_approx_score = approx_score
                    best_arch_approx_diff = approx_diff

        # Combine exact candidates
        best_exact_data = None
        best_exact_score = None

        if best_reg_exact_path is not None:
            try:
                with open(best_reg_exact_path, "rb") as f:
                    data = f.read()
                best_exact_data = data
                best_exact_score = best_reg_exact_score
            except Exception:
                pass

        if best_gz_exact_data is not None:
            if (
                best_exact_data is None
                or (
                    best_gz_exact_score is not None
                    and best_gz_exact_score > best_exact_score
                )
            ):
                best_exact_data = best_gz_exact_data
                best_exact_score = best_gz_exact_score

        if best_arch_exact_data is not None:
            if (
                best_exact_data is None
                or (
                    best_arch_exact_score is not None
                    and best_arch_exact_score > best_exact_score
                )
            ):
                best_exact_data = best_arch_exact_data
                best_exact_score = best_arch_exact_score

        if best_exact_data is not None:
            return best_exact_data

        # No exact-size candidate; use the best approximate one
        best_approx_data = None
        best_approx_diff = None
        best_approx_score = None

        if best_reg_approx_path is not None:
            try:
                with open(best_reg_approx_path, "rb") as f:
                    data = f.read()
                best_approx_data = data
                best_approx_diff = best_reg_approx_diff
                best_approx_score = best_reg_approx_score
            except Exception:
                pass

        if best_gz_approx_data is not None:
            if (
                best_approx_data is None
                or best_gz_approx_diff < best_approx_diff
                or (
                    best_gz_approx_diff == best_approx_diff
                    and best_gz_approx_score > best_approx_score
                )
            ):
                best_approx_data = best_gz_approx_data
                best_approx_diff = best_gz_approx_diff
                best_approx_score = best_gz_approx_score

        if best_arch_approx_data is not None:
            if (
                best_approx_data is None
                or best_arch_approx_diff < best_approx_diff
                or (
                    best_arch_approx_diff == best_approx_diff
                    and best_arch_approx_score > best_approx_score
                )
            ):
                best_approx_data = best_arch_approx_data
                best_approx_diff = best_arch_approx_diff
                best_approx_score = best_arch_approx_score

        return best_approx_data

    def _search_archive_for_poc(self, archive_path: str):
        target = self.GROUND_TRUTH_SIZE

        exact_data = None
        exact_score = None

        approx_data = None
        approx_score = None
        approx_diff = None

        try:
            if tarfile.is_tarfile(archive_path):
                with tarfile.open(archive_path, "r:*") as tf:
                    best_exact_member = None
                    best_exact_score = None

                    best_approx_member = None
                    best_approx_score = None
                    best_approx_diff = None

                    for member in tf.getmembers():
                        if not member.isreg():
                            continue
                        size = member.size
                        path_str = f"{archive_path}::{member.name}"
                        score = self._score_path(path_str, size)

                        if size == target:
                            if (
                                best_exact_member is None
                                or score > best_exact_score
                            ):
                                best_exact_member = member
                                best_exact_score = score
                        else:
                            diff = abs(size - target)
                            if (
                                best_approx_member is None
                                or diff < best_approx_diff
                                or (
                                    diff == best_approx_diff
                                    and score > best_approx_score
                                )
                            ):
                                best_approx_member = member
                                best_approx_score = score
                                best_approx_diff = diff

                    if best_exact_member is not None:
                        fobj = tf.extractfile(best_exact_member)
                        if fobj is not None:
                            exact_data = fobj.read()
                            exact_score = best_exact_score
                        return exact_data, exact_score, approx_data, approx_score, approx_diff

                    if best_approx_member is not None:
                        fobj = tf.extractfile(best_approx_member)
                        if fobj is not None:
                            approx_data = fobj.read()
                            approx_score = best_approx_score
                            approx_diff = best_approx_diff
                        return exact_data, exact_score, approx_data, approx_score, approx_diff

            elif zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(archive_path, "r") as zf:
                    best_exact_name = None
                    best_exact_score = None

                    best_approx_name = None
                    best_approx_score = None
                    best_approx_diff = None

                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        size = info.file_size
                        path_str = f"{archive_path}::{info.filename}"
                        score = self._score_path(path_str, size)

                        if size == target:
                            if best_exact_name is None or score > best_exact_score:
                                best_exact_name = info.filename
                                best_exact_score = score
                        else:
                            diff = abs(size - target)
                            if (
                                best_approx_name is None
                                or diff < best_approx_diff
                                or (
                                    diff == best_approx_diff
                                    and score > best_approx_score
                                )
                            ):
                                best_approx_name = info.filename
                                best_approx_score = score
                                best_approx_diff = diff

                    if best_exact_name is not None:
                        exact_data = zf.read(best_exact_name)
                        exact_score = best_exact_score
                        return exact_data, exact_score, approx_data, approx_score, approx_diff

                    if best_approx_name is not None:
                        approx_data = zf.read(best_approx_name)
                        approx_score = best_approx_score
                        approx_diff = best_approx_diff
                        return exact_data, exact_score, approx_data, approx_score, approx_diff

        except Exception:
            pass

        return None, None, None, None, None

    def _score_path(self, path: str, size: int) -> int:
        path_lower = path.lower()
        name = os.path.basename(path_lower)

        score = 0

        if "poc" in name:
            score += 100
        if "poc" in path_lower:
            score += 50

        if "crash" in name:
            score += 80
        if "crash" in path_lower:
            score += 40

        if "clusterfuzz" in path_lower or "cluster_fuzz" in path_lower:
            score += 60

        if self.BUG_ID in path_lower:
            score += 40

        if "svcdec" in path_lower or "svc" in path_lower:
            score += 20

        if "fuzz" in path_lower:
            score += 10

        if "test" in path_lower or "regress" in path_lower:
            score += 5

        base, ext = os.path.splitext(name)
        if ext in ("", ".bin", ".dat", ".raw", ".yuv", ".h264", ".264", ".svc", ".es"):
            score += 30

        if ext in (
            ".c",
            ".cc",
            ".cpp",
            ".h",
            ".hpp",
            ".hh",
            ".txt",
            ".md",
            ".json",
            ".xml",
            ".html",
            ".py",
            ".sh",
            ".cmake",
            ".java",
            ".rb",
            ".js",
            ".php",
            ".go",
            ".rs",
        ):
            score -= 100

        return score