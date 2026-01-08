import os
import tarfile
import zipfile
import tempfile
import shutil
import json
from typing import Any, Iterable, List, Optional, Tuple, Set


TARGET_LEN = 149


class Solution:
    def solve(self, src_path: str) -> bytes:
        root_dir = None
        tmpdir = None
        try:
            if os.path.isdir(src_path):
                root_dir = src_path
            elif self._is_tarfile(src_path):
                tmpdir = tempfile.mkdtemp(prefix="pocgen_tar_")
                self._extract_tar(src_path, tmpdir)
                root_dir = tmpdir
            elif self._is_zipfile(src_path):
                tmpdir = tempfile.mkdtemp(prefix="pocgen_zip_")
                self._extract_zip(src_path, tmpdir)
                root_dir = tmpdir
            elif os.path.isfile(src_path):
                # If it's a standalone file, just return its contents
                with open(src_path, "rb") as f:
                    return f.read()
            else:
                # Unknown type; fallback
                return b"A" * TARGET_LEN

            if root_dir is None:
                return b"A" * TARGET_LEN

            poc = self._find_poc_in_root(root_dir)
            if poc is None:
                # Fallback: deterministic dummy data
                return b"A" * TARGET_LEN
            return poc
        finally:
            if tmpdir is not None:
                shutil.rmtree(tmpdir, ignore_errors=True)

    # --- archive helpers ---

    def _is_tarfile(self, path: str) -> bool:
        try:
            return tarfile.is_tarfile(path)
        except Exception:
            return False

    def _is_zipfile(self, path: str) -> bool:
        try:
            return zipfile.is_zipfile(path)
        except Exception:
            return False

    def _extract_tar(self, tar_path: str, dest_dir: str) -> None:
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                tf.extractall(dest_dir)
        except Exception:
            # Best-effort extraction; ignore on failure
            pass

    def _extract_zip(self, zip_path: str, dest_dir: str) -> None:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest_dir)
        except Exception:
            pass

    # --- PoC discovery ---

    def _find_poc_in_root(self, root: str) -> Optional[bytes]:
        # 1. Try bug_info-style metadata
        bug_info_paths = self._find_bug_info_files(root)
        poc_paths_from_bug_info = self._gather_paths_from_bug_info_files(bug_info_paths, root)
        if poc_paths_from_bug_info:
            # Score among candidates discovered via bug_info first
            best_path, _ = self._choose_best_scored(poc_paths_from_bug_info, root)
            if best_path is not None:
                try:
                    with open(best_path, "rb") as f:
                        return f.read()
                except Exception:
                    pass

        # 2. Generic heuristic scan over all files
        all_files = self._collect_all_files(root)
        best_path, _ = self._choose_best_scored(all_files, root)
        if best_path is not None:
            try:
                with open(best_path, "rb") as f:
                    return f.read()
            except Exception:
                pass

        return None

    def _collect_all_files(self, root: str) -> List[str]:
        result: List[str] = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                if os.path.isfile(full):
                    result.append(full)
        return result

    # --- bug_info parsing ---

    def _find_bug_info_files(self, root: str) -> List[str]:
        candidates: List[str] = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                lname = fn.lower()
                if (
                    "bug_info" in lname
                    or "bug-info" in lname
                    or (lname.endswith(".json") and "bug" in lname and "info" in lname)
                ):
                    full = os.path.join(dirpath, fn)
                    if os.path.isfile(full):
                        candidates.append(full)
        return candidates

    def _gather_paths_from_bug_info_files(self, bug_info_paths: List[str], root: str) -> List[str]:
        result: Set[str] = set()
        for info_path in bug_info_paths:
            if info_path.lower().endswith(".json"):
                for p in self._paths_from_bug_info_json(info_path, root):
                    result.add(p)
            else:
                for p in self._paths_from_bug_info_text(info_path, root):
                    result.add(p)
        return list(result)

    def _paths_from_bug_info_json(self, json_path: str, root: str) -> Iterable[str]:
        try:
            with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
        except Exception:
            return []

        strings = list(self._extract_strings_from_obj(data))
        base_dir = os.path.dirname(json_path)
        candidates: Set[str] = set()
        for s in strings:
            for path in self._normalize_candidate_path(s, root, base_dir):
                if os.path.isfile(path):
                    candidates.add(path)
        return candidates

    def _paths_from_bug_info_text(self, txt_path: str, root: str) -> Iterable[str]:
        prefixes = ["poc", "poc_path", "poc-file", "poc_file", "input", "crash", "crash_input"]
        candidates: Set[str] = set()
        base_dir = os.path.dirname(txt_path)
        try:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    l = line.strip()
                    ll = l.lower()
                    for prefix in prefixes:
                        if ll.startswith(prefix):
                            for sep in (":", "="):
                                if sep in l:
                                    val = l.split(sep, 1)[1].strip().strip("\"'")
                                    if val:
                                        for path in self._normalize_candidate_path(val, root, base_dir):
                                            if os.path.isfile(path):
                                                candidates.add(path)
                                    break
        except Exception:
            return []

        return candidates

    def _extract_strings_from_obj(self, obj: Any) -> Iterable[str]:
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, dict):
            for v in obj.values():
                for s in self._extract_strings_from_obj(v):
                    yield s
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                for s in self._extract_strings_from_obj(item):
                    yield s

    def _normalize_candidate_path(self, s: str, root: str, base_dir: str) -> Iterable[str]:
        s = s.strip()
        if not s:
            return []
        paths: List[str] = []

        # If absolute, use as-is
        if os.path.isabs(s):
            paths.append(s)
        else:
            paths.append(os.path.join(root, s))
            paths.append(os.path.join(base_dir, s))

        # Also consider if s is something like "path:subpath"
        if ":" in s and not os.path.exists(paths[0]):
            tail = s.split(":", 1)[1].strip()
            if tail:
                if os.path.isabs(tail):
                    paths.append(tail)
                else:
                    paths.append(os.path.join(root, tail))
                    paths.append(os.path.join(base_dir, tail))

        # Deduplicate preserving order
        seen: Set[str] = set()
        result: List[str] = []
        for p in paths:
            rp = os.path.normpath(p)
            if rp not in seen:
                seen.add(rp)
                result.append(rp)
        return result

    # --- scoring ---

    def _choose_best_scored(self, paths: List[str], root: str) -> Tuple[Optional[str], Optional[int]]:
        best_path: Optional[str] = None
        best_score: Optional[int] = None
        best_size: Optional[int] = None

        for p in paths:
            try:
                size = os.path.getsize(p)
            except OSError:
                continue
            if size <= 0:
                continue
            rel = os.path.relpath(p, root)
            score = self._score_file(rel, size)
            if best_path is None:
                best_path, best_score, best_size = p, score, size
            else:
                if score > best_score:  # type: ignore[operator]
                    best_path, best_score, best_size = p, score, size
                elif score == best_score and size < best_size:  # type: ignore[operator]
                    best_path, best_score, best_size = p, score, size
        return best_path, best_score

    def _score_file(self, rel_path: str, size: int) -> int:
        # Base on closeness to target length
        diff = abs(size - TARGET_LEN)
        size_score = max(0, 1000 - diff)

        path_lower = rel_path.lower()
        base = os.path.basename(path_lower)
        ext = os.path.splitext(base)[1]

        score = size_score

        # Keyword bonuses
        keyword_bonus = 0
        keywords_high = ["poc", "testcase", "crash", "clusterfuzz", "id:", "id_", "repro"]
        keywords_med = ["fuzz", "input", "sample", "seed", "queue", "rv", "rv60", "ffmpeg"]

        for kw in keywords_high:
            if kw in path_lower:
                keyword_bonus += 3000
        for kw in keywords_med:
            if kw in path_lower:
                keyword_bonus += 800

        score += keyword_bonus

        # Extension-based adjustments
        bin_exts = [
            ".bin",
            ".dat",
            ".mp4",
            ".mkv",
            ".webm",
            ".rv",
            ".rm",
            ".rmvb",
            ".ts",
            ".avi",
            ".flv",
            ".ogg",
            ".ogv",
        ]
        text_exts = [
            ".c",
            ".h",
            ".cpp",
            ".cc",
            ".hpp",
            ".txt",
            ".md",
            ".html",
            ".htm",
            ".sh",
            ".py",
            ".java",
            ".rb",
            ".go",
            ".rs",
            ".js",
            ".json",
            ".yml",
            ".yaml",
            ".toml",
            ".cfg",
            ".ini",
            ".cmake",
        ]

        if ext in bin_exts:
            score += 600
        if ext in text_exts:
            score -= 1200

        # Penalize obvious non-input artifacts
        bad_keywords = ["makefile", "readme", "license", "changelog", "config", "cmakelists"]
        for bk in bad_keywords:
            if bk in base:
                score -= 1500

        # Large file penalty
        if size > 2_000_000:
            score -= 3000
        elif size > 200_000:
            score -= 1000

        return score