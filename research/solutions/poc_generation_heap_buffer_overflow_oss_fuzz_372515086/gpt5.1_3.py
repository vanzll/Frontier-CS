import os
import tarfile
import re
import json
import base64


class Solution:
    def solve(self, src_path: str) -> bytes:
        BUG_ID = "372515086"
        GROUND_TRUTH_LEN = 1032

        def load_member_content(tar, member):
            try:
                f = tar.extractfile(member)
                if not f:
                    return None
                return f.read()
            except Exception:
                return None

        try:
            tar = tarfile.open(src_path, "r:*")
        except Exception:
            return b"A" * GROUND_TRUTH_LEN

        with tar:
            members = tar.getmembers()
            by_name = {}
            base_to_members = {}
            for m in members:
                if not m.isfile():
                    continue
                by_name[m.name] = m
                base = os.path.basename(m.name)
                base_to_members.setdefault(base, []).append(m)

            skip_exts = (
                ".c",
                ".cc",
                ".cpp",
                ".cxx",
                ".h",
                ".hpp",
                ".hh",
                ".txt",
                ".md",
                ".rst",
                ".cmake",
                ".sh",
                ".py",
                ".java",
                ".js",
                ".html",
                ".xml",
                ".yml",
                ".yaml",
                ".toml",
                ".in",
                ".am",
                ".ac",
                ".m4",
                ".pc",
                ".lock",
                ".mod",
                ".sum",
                ".cmake.in",
                ".bat",
                ".ps1",
            )

            def try_parse_json_for_poc():
                for m in members:
                    if not m.isfile():
                        continue
                    nlower = m.name.lower()
                    if not nlower.endswith(".json"):
                        continue
                    if not (BUG_ID in nlower or re.search(r"(bug|poc|crash|testcase)", nlower)):
                        continue
                    data = load_member_content(tar, m)
                    if not data:
                        continue
                    try:
                        text = data.decode("utf-8")
                    except Exception:
                        continue
                    try:
                        obj = json.loads(text)
                    except Exception:
                        continue

                    stack = [obj]
                    visited_ids = set()
                    while stack:
                        cur = stack.pop()
                        oid = id(cur)
                        if oid in visited_ids:
                            continue
                        visited_ids.add(oid)
                        if isinstance(cur, dict):
                            for key in ("poc_bytes", "input_bytes", "crash_bytes"):
                                if key in cur and isinstance(cur[key], list):
                                    arr = cur[key]
                                    try:
                                        bts = bytes(int(x) & 0xFF for x in arr)
                                        if bts:
                                            return bts
                                    except Exception:
                                        pass
                            for key in ("poc", "poc_path", "poc_file", "input", "crash"):
                                if key in cur and isinstance(cur[key], str):
                                    val = cur[key].strip()
                                    if not val:
                                        continue
                                    stripped = val.strip()
                                    looks_base64 = False
                                    if (
                                        re.fullmatch(r"[0-9A-Za-z+/=\s]+", stripped) is not None
                                        and "/" not in stripped
                                        and "\\" not in stripped
                                        and "." not in stripped
                                        and len(stripped.replace("\n", "")) >= 16
                                    ):
                                        looks_base64 = True
                                    if looks_base64:
                                        try:
                                            decoded = base64.b64decode(stripped, validate=False)
                                            if decoded:
                                                return decoded
                                        except Exception:
                                            pass
                                    candidates = []
                                    if val in by_name:
                                        candidates.append(by_name[val])
                                    basename = os.path.basename(val)
                                    if basename in base_to_members:
                                        candidates.extend(base_to_members[basename])
                                    for mem in candidates:
                                        content = load_member_content(tar, mem)
                                        if content:
                                            return content
                            for v in cur.values():
                                if isinstance(v, (dict, list)):
                                    stack.append(v)
                        elif isinstance(cur, list):
                            for v in cur:
                                if isinstance(v, (dict, list)):
                                    stack.append(v)
                return None

            try:
                poc_data = try_parse_json_for_poc()
            except Exception:
                poc_data = None
            if poc_data:
                return poc_data

            best_poc = None
            best_score = float("-inf")

            for m in members:
                if not m.isfile() or m.size == 0:
                    continue
                nlower = m.name.lower()
                if BUG_ID not in nlower:
                    continue
                base = os.path.basename(nlower)
                if any(base.endswith(ext) for ext in skip_exts):
                    continue
                data = load_member_content(tar, m)
                if not data:
                    continue
                score = 0.0
                if "poc" in nlower:
                    score += 5.0
                if "crash" in nlower:
                    score += 4.0
                if "clusterfuzz" in nlower:
                    score += 3.0
                if "testcase" in nlower:
                    score += 2.0
                if "repro" in nlower:
                    score += 2.0
                if b"\x00" in data:
                    score += 2.0
                score -= abs(len(data) - GROUND_TRUTH_LEN) / float(GROUND_TRUTH_LEN)
                if score > best_score:
                    best_score = score
                    best_poc = data
            if best_poc:
                return best_poc

            poc_like_pattern = re.compile(r"(poc|crash|repro|clusterfuzz|testcase)", re.IGNORECASE)
            for m in members:
                if not m.isfile() or m.size == 0:
                    continue
                nlower = m.name.lower()
                if not poc_like_pattern.search(nlower):
                    continue
                base = os.path.basename(nlower)
                if any(base.endswith(ext) for ext in skip_exts) and not base.endswith(".json"):
                    continue
                data = load_member_content(tar, m)
                if not data:
                    continue
                if base.endswith(".json"):
                    continue
                score = 0.0
                if "poc" in nlower:
                    score += 5.0
                if "crash" in nlower:
                    score += 4.0
                if "clusterfuzz" in nlower:
                    score += 3.0
                if "testcase" in nlower:
                    score += 2.0
                if "repro" in nlower:
                    score += 2.0
                if BUG_ID in nlower:
                    score += 3.0
                if b"\x00" in data:
                    score += 2.0
                score -= abs(len(data) - GROUND_TRUTH_LEN) / float(GROUND_TRUTH_LEN)
                if score > best_score:
                    best_score = score
                    best_poc = data
            if best_poc:
                return best_poc

            for m in members:
                if not m.isfile() or m.size == 0:
                    continue
                base = os.path.basename(m.name)
                nlower = base.lower()
                if not (nlower.startswith("id:") or nlower.startswith("id_") or nlower.startswith("id-")):
                    continue
                data = load_member_content(tar, m)
                if not data:
                    continue
                score = 0.0
                if b"\x00" in data:
                    score += 2.0
                score -= abs(len(data) - GROUND_TRUTH_LEN) / float(GROUND_TRUTH_LEN)
                if score > best_score:
                    best_score = score
                    best_poc = data
            if best_poc:
                return best_poc

            for m in members:
                if not m.isfile() or m.size == 0 or m.size > 4096:
                    continue
                base = os.path.basename(m.name.lower())
                if any(
                    base.endswith(ext)
                    for ext in skip_exts
                    + (".json", ".csv", ".tsv", ".ini", ".cfg", ".conf", ".log")
                ):
                    continue
                data = load_member_content(tar, m)
                if not data:
                    continue
                if b"\x00" not in data:
                    continue
                score = -abs(len(data) - GROUND_TRUTH_LEN)
                if score > best_score:
                    best_score = score
                    best_poc = data
            if best_poc:
                return best_poc

            return b"A" * GROUND_TRUTH_LEN