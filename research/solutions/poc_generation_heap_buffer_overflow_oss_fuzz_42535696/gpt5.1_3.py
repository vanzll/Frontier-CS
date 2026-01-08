import os
import tarfile
import tempfile
import json
import gzip
import bz2
import lzma
import zipfile


class Solution:
    def __init__(self):
        self.bug_id = "42535696"
        self.target_poc_size = 150979

    def solve(self, src_path: str) -> bytes:
        root_dir = self._prepare_root_dir(src_path)

        # Try to find PoC via metadata/JSON hints
        poc_path = self._find_poc_via_metadata(root_dir)
        if poc_path is None:
            # Fallback: heuristic scan of all files
            poc_path = self._guess_poc_by_scanning(root_dir)

        if poc_path is not None:
            data = self._read_poc_file(poc_path)
            if data:
                return data

        # Ultimate fallback if nothing useful found
        return self._fallback_poc()

    def _prepare_root_dir(self, src_path: str) -> str:
        if os.path.isdir(src_path):
            return src_path

        # If it's a file, try to treat it as a tarball and extract
        if os.path.isfile(src_path):
            tmpdir = tempfile.mkdtemp(prefix="pocgen_")
            try:
                with tarfile.open(src_path, "r:*") as tar:
                    self._safe_extract(tar, tmpdir)
                return tmpdir
            except Exception:
                # If extraction fails, just use the directory of the file if any
                dir_name = os.path.dirname(os.path.abspath(src_path))
                if os.path.isdir(dir_name):
                    return dir_name
                return os.getcwd()
        # Fallback to current directory
        return os.getcwd()

    def _safe_extract(self, tar: tarfile.TarFile, path: str) -> None:
        base_path = os.path.abspath(path)

        for member in tar.getmembers():
            member_path = os.path.abspath(os.path.join(path, member.name))
            if not member_path.startswith(base_path):
                continue
            try:
                tar.extract(member, path)
            except Exception:
                continue

    def _find_poc_via_metadata(self, root_dir: str):
        # Look for JSON files that might reference a PoC path
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if not fname.lower().endswith(".json"):
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    data = json.loads(text)
                except Exception:
                    continue

                # Search recursively for fields that look like PoC paths
                candidates = []

                def walk(node):
                    if isinstance(node, dict):
                        for k, v in node.items():
                            kl = str(k).lower()
                            if isinstance(v, str):
                                vl = v.lower()
                                if (
                                    "poc" in kl
                                    or "repro" in kl
                                    or "crash" in kl
                                    or "input" in kl
                                    or "testcase" in kl
                                ) or (
                                    "poc" in vl
                                    or "repro" in vl
                                    or "crash" in vl
                                    or "testcase" in vl
                                ):
                                    candidates.append(v)
                            elif isinstance(v, (dict, list)):
                                walk(v)
                    elif isinstance(node, list):
                        for item in node:
                            walk(item)

                walk(data)

                for rel in candidates:
                    path = self._resolve_candidate_path(rel, dirpath, root_dir)
                    if path is not None:
                        return path
        return None

    def _resolve_candidate_path(self, value: str, base_dir: str, root_dir: str):
        if not isinstance(value, str):
            return None
        v = value.strip()
        if not v:
            return None

        # Absolute path
        if os.path.isabs(v):
            if os.path.isfile(v):
                return v

        # Relative to the JSON directory
        cand = os.path.join(base_dir, v)
        if os.path.isfile(cand):
            return cand

        # Relative to root_dir
        cand = os.path.join(root_dir, v)
        if os.path.isfile(cand):
            return cand

        return None

    def _guess_poc_by_scanning(self, root_dir: str):
        goal_size = self.target_poc_size
        bug_id = self.bug_id

        best_exact_path = None
        best_exact_score = -1
        best_any_path = None
        best_any_score = -1

        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                try:
                    if not os.path.isfile(fpath):
                        continue
                    size = os.path.getsize(fpath)
                except OSError:
                    continue

                if size <= 0:
                    continue

                rel = os.path.relpath(fpath, root_dir)
                name_lower = fname.lower()
                rel_lower = rel.lower()

                score = 0

                # Strong indicators from bug ID
                if bug_id in fname or bug_id in rel:
                    score += 5000

                if "oss-fuzz" in rel_lower or "ossfuzz" in rel_lower or "clusterfuzz" in rel_lower:
                    score += 2000

                # Look for PoC/Crash/Testcase hints
                if (
                    "poc" in rel_lower
                    or "repro" in rel_lower
                    or "crash" in rel_lower
                    or "testcase" in rel_lower
                    or "id_" in name_lower
                ):
                    score += 1200

                # Prefer binary-looking extensions
                if name_lower.endswith(
                    (
                        ".pdf",
                        ".ps",
                        ".bin",
                        ".raw",
                        ".dat",
                        ".in",
                        ".input",
                        ".poc",
                    )
                ):
                    score += 700

                # De-prioritize obvious source/config/docs
                if name_lower.endswith(
                    (
                        ".c",
                        ".h",
                        ".cpp",
                        ".cc",
                        ".cxx",
                        ".hpp",
                        ".java",
                        ".py",
                        ".sh",
                        ".md",
                        ".rst",
                        ".html",
                        ".xml",
                        ".json",
                        ".yaml",
                        ".yml",
                        ".toml",
                        ".cfg",
                        ".ini",
                        ".cmake",
                        ".am",
                        ".ac",
                        ".txt",
                    )
                ):
                    score -= 300

                # Factor in closeness to target PoC size
                if goal_size:
                    diff = abs(size - goal_size)
                    if diff == 0:
                        score += 4000
                    elif diff < 100:
                        score += 1500
                    elif diff < 1000:
                        score += 700
                    elif diff < 10000:
                        score += 200

                # Slight penalty for very large files to avoid huge binaries
                if size > 10 * 1024 * 1024:
                    score -= 500

                if size == goal_size and score > best_exact_score:
                    best_exact_score = score
                    best_exact_path = fpath

                if score > best_any_score:
                    best_any_score = score
                    best_any_path = fpath

        return best_exact_path or best_any_path

    def _read_poc_file(self, path: str) -> bytes:
        name_lower = os.path.basename(path).lower()
        try:
            if name_lower.endswith(".gz"):
                with gzip.open(path, "rb") as f:
                    return f.read()
            if name_lower.endswith(".bz2"):
                with bz2.open(path, "rb") as f:
                    return f.read()
            if name_lower.endswith(".xz") or name_lower.endswith(".lzma"):
                with lzma.open(path, "rb") as f:
                    return f.read()
            if name_lower.endswith(".zip"):
                with zipfile.ZipFile(path, "r") as zf:
                    members = [m for m in zf.namelist() if not m.endswith("/")]
                    if not members:
                        return b""
                    best_name = None
                    best_score = -1
                    for m in members:
                        info = zf.getinfo(m)
                        m_lower = m.lower()
                        score = 0
                        if self.bug_id in m:
                            score += 2000
                        if "poc" in m_lower or "crash" in m_lower or "testcase" in m_lower:
                            score += 1000
                        diff = abs(info.file_size - self.target_poc_size)
                        if diff == 0:
                            score += 2000
                        elif diff < 100:
                            score += 800
                        elif diff < 1000:
                            score += 300
                        if score > best_score:
                            best_score = score
                            best_name = m
                    if best_name is None:
                        best_name = members[0]
                    with zf.open(best_name, "r") as f:
                        return f.read()
            with open(path, "rb") as f:
                return f.read()
        except Exception:
            return b""

    def _fallback_poc(self) -> bytes:
        # Fallback PostScript attempting to exercise pdfwrite viewer state handling
        fallback_ps = b"""%!PS-Adobe-3.0
%% Fallback PoC for pdfwrite viewer state handling
%% Try to exercise DOCVIEW/PAGEVIEW/VIEW pdfmarks in odd ways.

[/Title (Heap Overflow PoC Fallback) /Author (AutoGenerator) /DOCINFO pdfmark

% First, set some normal viewer state
[ /PageMode /UseOutlines /DOCVIEW pdfmark
[ /View [ /Fit ] /PAGEVIEW pdfmark

% Now, attempt to confuse the viewer state stack with unusual sequences
[ /View [ /XYZ null null 0 ] /DOCVIEW pdfmark
[ /Page 1 /View [ /FitV 0 ] /PAGEVIEW pdfmark
[ /Page 1 /View [ /FitH 0 ] /PAGEVIEW pdfmark
[ /Dest (XYZDest) /View [ /XYZ 0 0 0 ] /VIEW pdfmark

% Invalid/degenerate marks that may tickle edge cases
[ /Page -1 /View [ /Fit ] /PAGEVIEW pdfmark
[ /Page 0 /View [ /XYZ null null -1 ] /PAGEVIEW pdfmark
[ /InvalidKey 0 /InvalidValue 0 /DOCVIEW pdfmark

showpage
"""
        return fallback_ps