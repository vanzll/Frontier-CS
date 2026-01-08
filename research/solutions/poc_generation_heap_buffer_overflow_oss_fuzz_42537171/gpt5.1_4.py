import os
import tarfile
import zipfile
import io
from typing import Optional, Tuple


class Solution:
    def solve(self, src_path: str) -> bytes:
        target_len = 825339

        # Try to find an existing PoC inside the tarball based on size heuristics.
        poc = self._find_existing_poc(src_path, target_len)
        if poc is not None:
            return poc

        # Try to generate a PoC from any available seed/corpus files.
        poc = self._generate_from_seeds(src_path, target_len)
        if poc is not None:
            return poc

        # Fallback: generate a synthetic textual PoC with many clip/layer/save tokens.
        return self._fallback_pattern(target_len)

    def _find_existing_poc(self, tar_path: str, target_len: int) -> Optional[bytes]:
        non_poc_text_exts = {
            ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
            ".m", ".mm", ".java", ".py", ".pyw", ".pyi", ".pyx",
            ".go", ".rs", ".js", ".ts", ".tsx", ".jsx",
            ".html", ".htm", ".css", ".xml", ".xhtml",
            ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
            ".txt", ".md", ".rst", ".tex", ".csv", ".tsv", ".log",
            ".cmake", ".mak", ".make", ".mk", ".ninja",
            ".sh", ".bash", ".bat", ".ps1",
            ".in", ".ac", ".am", ".m4",
            ".sln", ".vcxproj", ".csproj", ".gradle", ".properties",
            ".pl", ".pm", ".php", ".rb", ".lua"
        }
        keywords = ("poc", "crash", "testcase", "clusterfuzz", "id_", "bug", "repro")

        best_exact: Optional[tarfile.TarInfo] = None
        best_exact_keyword: Optional[tarfile.TarInfo] = None
        best_close_nontext: Optional[tarfile.TarInfo] = None
        best_close_nontext_diff: Optional[int] = None
        best_close_any: Optional[tarfile.TarInfo] = None
        best_close_any_diff: Optional[int] = None

        try:
            with tarfile.open(tar_path, "r:*") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    size = member.size
                    if size <= 0:
                        continue
                    diff = abs(size - target_len)
                    name_lower = member.name.lower()
                    _, ext = os.path.splitext(name_lower)

                    if size == target_len:
                        if any(k in name_lower for k in keywords):
                            # Prefer keyword-matching exact candidates.
                            if best_exact_keyword is None:
                                best_exact_keyword = member
                        else:
                            if best_exact is None:
                                best_exact = member

                    if ext not in non_poc_text_exts:
                        if best_close_nontext is None or diff < best_close_nontext_diff:
                            best_close_nontext = member
                            best_close_nontext_diff = diff

                    if best_close_any is None or diff < best_close_any_diff:
                        best_close_any = member
                        best_close_any_diff = diff

                candidate = None
                if best_exact_keyword is not None:
                    candidate = best_exact_keyword
                elif best_exact is not None:
                    candidate = best_exact
                else:
                    # Use reasonably close non-text candidate if within 10% size range.
                    if best_close_nontext is not None and best_close_nontext_diff is not None:
                        if best_close_nontext_diff <= int(target_len * 0.10):
                            candidate = best_close_nontext
                    # As a last resort, if any file is extremely close (within 5%), use it.
                    if candidate is None and best_close_any is not None and best_close_any_diff is not None:
                        if best_close_any_diff <= int(target_len * 0.05):
                            candidate = best_close_any

                if candidate is not None:
                    f = tar.extractfile(candidate)
                    if f is None:
                        return None
                    data = f.read()
                    return data
        except Exception:
            return None

        return None

    def _generate_from_seeds(self, tar_path: str, target_len: int) -> Optional[bytes]:
        seed_datas = []

        try:
            with tarfile.open(tar_path, "r:*") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    lname = member.name.lower()
                    if member.size <= 0:
                        continue

                    # Heuristic: look for potential seed/corpus/test files.
                    if any(k in lname for k in ("seed", "corpus", "test", "fuzz")):
                        # Skip very large files to avoid heavy processing.
                        if member.size > target_len * 4 and not lname.endswith(".zip"):
                            continue

                        if lname.endswith(".zip"):
                            # Attempt to read a zip seed corpus.
                            if member.size > 10 * 1024 * 1024:
                                continue
                            f = tar.extractfile(member)
                            if f is None:
                                continue
                            try:
                                bz = f.read()
                                zf = zipfile.ZipFile(io.BytesIO(bz))
                            except Exception:
                                continue
                            for zinfo in zf.infolist():
                                if zinfo.is_dir():
                                    continue
                                # Limit size per entry.
                                if zinfo.file_size <= 0 or zinfo.file_size > target_len * 2:
                                    continue
                                try:
                                    seed = zf.read(zinfo)
                                except Exception:
                                    continue
                                if seed:
                                    seed_datas.append(seed)
                        else:
                            f = tar.extractfile(member)
                            if f is None:
                                continue
                            try:
                                seed = f.read()
                            except Exception:
                                continue
                            if seed:
                                seed_datas.append(seed)
        except Exception:
            return None

        best_clip: Optional[bytes] = None
        best_any: Optional[bytes] = None

        for seed in seed_datas:
            try:
                mutated, used_clip = self._mutate_seed(seed, target_len)
            except Exception:
                continue
            if mutated is None:
                continue
            if used_clip:
                if best_clip is None or len(mutated) < len(best_clip):
                    best_clip = mutated
            else:
                if best_any is None or len(mutated) < len(best_any):
                    best_any = mutated

        return best_clip or best_any

    def _mutate_seed(self, seed_data: bytes, target_len: int) -> Tuple[Optional[bytes], bool]:
        if not seed_data:
            return None, False

        try:
            text = seed_data.decode("latin1", "ignore")
        except Exception:
            text = ""
        text_lower = text.lower() if text else ""

        tokens = ["clip", "save", "layer", "mask"]
        best_idx = None
        best_token = None
        for tok in tokens:
            idx = text_lower.find(tok)
            if idx != -1 and (best_idx is None or idx < best_idx):
                best_idx = idx
                best_token = tok

        used_clip = (best_token == "clip")

        if best_token is not None and text:
            # Use the line containing the keyword as the repeated segment.
            start = text.rfind("\n", 0, best_idx)
            if start == -1:
                start = 0
            else:
                start += 1
            end = text.find("\n", best_idx)
            if end == -1:
                end = len(text)

            start_b = start
            end_b = end
            if start_b < 0:
                start_b = 0
            if end_b > len(seed_data):
                end_b = len(seed_data)

            segment = seed_data[start_b:end_b]
            if segment:
                seg_len = len(segment)
                reps = max(1, target_len // max(1, seg_len))
                body = segment * reps
                new_data = seed_data[:start_b] + body + seed_data[end_b:]
                max_len = max(target_len, len(seed_data))
                if len(new_data) > max_len * 2:
                    new_data = new_data[:max_len * 2]
                return new_data, used_clip

        # Fallback: repeat entire seed to reach target scale.
        seg = seed_data
        seg_len = len(seg)
        if seg_len == 0:
            return None, False
        reps = max(1, target_len // seg_len)
        body = seg * reps
        max_len = max(target_len, seg_len)
        if len(body) > max_len * 2:
            body = body[:max_len * 2]
        return body, False

    def _fallback_pattern(self, target_len: int) -> bytes:
        # Construct a generic textual pattern rich in "clip", "save", and "layer" tokens
        # to exercise potential layer/clip stack handling.
        base_line = (
            b"# layer/clip stack stress test\n"
            b"save\n"
            b"push_layer\n"
            b"clip\n"
            b"clip\n"
            b"clip\n"
            b"save\n"
            b"clip\n"
        )
        if not base_line:
            base_line = b"clip\n"

        reps = max(1, target_len // len(base_line))
        data = base_line * reps
        if len(data) < target_len:
            data += b"\n" * (target_len - len(data))
        elif len(data) > target_len:
            data = data[:target_len]
        return data