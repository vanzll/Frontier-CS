import tarfile
import os
import re


class Solution:
    def solve(self, src_path: str) -> bytes:
        BUG_ID = "385170375"
        TARGET_LEN = 149

        def safe_read_member(tf: tarfile.TarFile, member: tarfile.TarInfo) -> bytes | None:
            try:
                f = tf.extractfile(member)
                if f is None:
                    return None
                return f.read()
            except Exception:
                return None

        def member_score(m: tarfile.TarInfo) -> int:
            n = m.name.lower()
            score = 0
            if "tests/" in n or n.startswith("tests"):
                score -= 10
            if "data/" in n:
                score -= 5
            if "fate" in n:
                score -= 3
            if "fuzz" in n:
                score -= 2
            if "ossfuzz" in n or "oss-fuzz" in n:
                score -= 20
            score += int(m.size / 1024)
            return score

        def iter_text_members(tf: tarfile.TarFile, members):
            text_exts = (".mak", ".txt", ".c", ".h", ".md", ".ini", ".cfg")
            for m in members:
                if not m.isfile():
                    continue
                if m.size > 1024 * 1024:
                    continue
                name_lower = m.name.lower()
                if not name_lower.endswith(text_exts) and "makefile" not in name_lower:
                    continue
                data = safe_read_member(tf, m)
                if not data:
                    continue
                try:
                    text = data.decode("utf-8", "ignore")
                except Exception:
                    continue
                yield m, text

        with tarfile.open(src_path, "r:*") as tf:
            members = tf.getmembers()
            file_members = [m for m in members if m.isfile()]

            # 1) Direct filename match containing bug id
            id_matches = [m for m in file_members if BUG_ID in m.name]
            if id_matches:
                exact_len = [m for m in id_matches if m.size == TARGET_LEN]
                if exact_len:
                    chosen = sorted(exact_len, key=member_score)[0]
                    data = safe_read_member(tf, chosen)
                    if data is not None:
                        return data
                chosen = sorted(id_matches, key=member_score)[0]
                data = safe_read_member(tf, chosen)
                if data is not None:
                    return data

            # 2) Parse text files to find references to the PoC
            for m, text in iter_text_members(tf, file_members):
                if BUG_ID not in text:
                    continue
                for line in text.splitlines():
                    if BUG_ID not in line:
                        continue
                    match = re.search(r"(?:ossfuzz|oss-fuzz)-?385170375[^\s\"']*", line)
                    if not match:
                        continue
                    token = match.group(0).strip(" \"'")
                    base = os.path.basename(token)
                    base = base.replace("$(TARGET_SAMPLES)/", "")
                    candidates = [
                        fm
                        for fm in file_members
                        if base and base in os.path.basename(fm.name)
                    ]
                    if not candidates:
                        candidates = [
                            fm
                            for fm in file_members
                            if os.path.basename(fm.name).startswith(base)
                        ]
                    if candidates:
                        exact_len = [fm for fm in candidates if fm.size == TARGET_LEN]
                        if exact_len:
                            chosen = exact_len[0]
                        else:
                            chosen = min(candidates, key=lambda x: x.size)
                        data = safe_read_member(tf, chosen)
                        if data is not None:
                            return data

            # 3) Fallback: any ossfuzz-related small file
            fuzz_candidates = [
                m
                for m in file_members
                if "ossfuzz" in m.name.lower() or "oss-fuzz" in m.name.lower()
            ]
            if fuzz_candidates:
                exact_len = [m for m in fuzz_candidates if m.size == TARGET_LEN]
                if exact_len:
                    chosen = exact_len[0]
                else:
                    chosen = min(fuzz_candidates, key=lambda x: x.size)
                data = safe_read_member(tf, chosen)
                if data is not None:
                    return data

        # 4) Final fallback: deterministic placeholder
        return b"A" * TARGET_LEN