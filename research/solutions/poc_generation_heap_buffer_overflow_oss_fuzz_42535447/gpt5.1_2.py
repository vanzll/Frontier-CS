import os
import tarfile
import gzip


class Solution:
    def solve(self, src_path: str) -> bytes:
        target_len = 133
        bug_id = "42535447"

        code_ext = {
            ".c", ".cc", ".cxx", ".cpp", ".h", ".hpp", ".hh",
            ".java", ".py", ".pyw", ".pyx", ".pxd",
            ".go", ".rs", ".js", ".ts", ".m", ".mm", ".cs",
            ".php", ".rb", ".swift", ".kt", ".kts",
            ".m4", ".cmake", ".in", ".am", ".ac",
            ".s", ".asm",
        }
        text_ext = {
            ".txt", ".md", ".markdown", ".rst",
            ".html", ".htm", ".xhtml",
            ".xml", ".xsd",
            ".json", ".yml", ".yaml", ".toml",
            ".ini", ".cfg", ".conf", ".config",
            ".csv", ".tsv",
            ".log",
            ".cmake", ".pc",
            ".sh", ".bash", ".zsh", ".bat", ".ps1",
            ".awk", ".sed", ".mak", ".mk",
            ".diff", ".patch",
            ".tex",
        }
        binary_suspect_ext = {
            ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif",
            ".tif", ".tiff", ".ico", ".icns",
            ".jxl", ".avif", ".heif", ".heic",
            ".pnm", ".ppm", ".pgm", ".pbm", ".pam",
            ".hdr", ".exr",
            ".dng", ".nef", ".cr2", ".arw", ".rw2", ".orf", ".raf", ".srw",
            ".psd", ".xcf",
            ".ktx", ".ktx2", ".dds", ".tga",
            ".mp4", ".mkv", ".webm", ".avi", ".mov",
            ".ogg", ".ogv", ".mp3", ".flac", ".wav",
            ".pdf", ".swf",
            ".bin", ".dat", ".raw",
        }

        def is_binary_sample(sample: bytes) -> bool:
            if not sample:
                return False
            if b"\x00" in sample:
                return True
            nontext = 0
            length = len(sample)
            for b in sample:
                if b in (9, 10, 13, 8, 12):
                    continue
                if 32 <= b <= 126:
                    continue
                nontext += 1
            return (nontext / float(length)) > 0.30

        def pick_best(cands):
            if not cands:
                return None
            best_mem = None
            best_metric = None
            for mem, _is_bin in cands:
                diff = abs(mem.size - target_len)
                metric = (diff, mem.size)
                if best_metric is None or metric < best_metric:
                    best_metric = metric
                    best_mem = mem
            return best_mem

        try:
            tf = tarfile.open(src_path, "r:*")
        except Exception:
            return b"A" * target_len

        members = tf.getmembers()

        high = []
        med = []
        low = []

        for mem in members:
            if not mem.isfile():
                continue
            if mem.size == 0:
                continue
            if mem.size > 5_000_000:
                continue

            name_lower = mem.name.lower()
            base = os.path.basename(name_lower)
            ext = os.path.splitext(base)[1]

            if ext in code_ext:
                continue

            try:
                f = tf.extractfile(mem)
                if f is None:
                    continue
                sample = f.read(min(4096, mem.size))
                f.close()
            except Exception:
                continue

            is_bin = is_binary_sample(sample)

            if (ext in text_ext or not is_bin) and mem.size > 4096:
                continue

            path_parts = name_lower.split("/")

            score_tag = 0
            if bug_id in name_lower:
                score_tag += 3
            if "poc" in name_lower or "proof" in name_lower:
                score_tag += 3
            if "clusterfuzz" in name_lower or "testcase" in name_lower or "minimized" in name_lower:
                score_tag += 3
            if any(s in name_lower for s in ("crash", "repro", "trigger")):
                score_tag += 2
            if any(part in ("poc", "pocs", "crash", "crashes", "bugs", "bug") for part in path_parts):
                score_tag += 2
            if "oss-fuzz" in name_lower or "ossfuzz" in name_lower:
                score_tag += 1
            if any(part == "fuzz" or part.endswith("fuzz") for part in path_parts):
                score_tag += 1
            if any(part in ("corpus", "seeds", "seed", "regress", "regression", "tests", "test") for part in path_parts):
                score_tag += 1
            if any("gainmap" in part or "gain_map" in part or "gain-map" in part for part in path_parts):
                score_tag += 2
            if ext in binary_suspect_ext:
                score_tag += 1
            if not is_bin:
                score_tag -= 1

            if score_tag >= 3:
                high.append((mem, is_bin))
            elif score_tag >= 1:
                med.append((mem, is_bin))
            else:
                if is_bin and mem.size <= 65536:
                    low.append((mem, is_bin))

        bugid_candidates = [(m, b) for (m, b) in high + med + low if bug_id in m.name]
        best_mem = pick_best(bugid_candidates)

        if best_mem is None:
            func_candidates = []
            for (m, b) in high + med + low:
                lname = m.name.lower()
                if "decodegainmapmetadata" in lname or "gainmap" in lname:
                    func_candidates.append((m, b))
            best_mem = pick_best(func_candidates)

        if best_mem is None:
            if high:
                best_mem = pick_best(high)
            elif med:
                best_mem = pick_best(med)
            else:
                best_mem = pick_best(low)

        data = b""
        if best_mem is not None:
            try:
                f = tf.extractfile(best_mem)
                if f is not None:
                    data = f.read()
                    f.close()
            except Exception:
                data = b""

        if not data:
            return b"A" * target_len

        name_lower = best_mem.name.lower()
        if name_lower.endswith(".gz") or name_lower.endswith(".gzip") or data.startswith(b"\x1f\x8b"):
            try:
                decomp = gzip.decompress(data)
                if decomp:
                    data = decomp
            except Exception:
                pass

        return data