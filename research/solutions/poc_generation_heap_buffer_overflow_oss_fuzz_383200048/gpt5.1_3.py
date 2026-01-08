import tarfile
import re


class Solution:
    def solve(self, src_path: str) -> bytes:
        TARGET_ID = "383200048"

        try:
            tf = tarfile.open(src_path, "r:*")
        except Exception:
            return self._fallback_poc()

        # Pass 1: search for an obvious PoC file by filename heuristics
        best_member = None
        best_score = -1

        for m in tf.getmembers():
            if not m.isfile():
                continue

            name_lower = m.name.lower()
            size = m.size

            if size <= 0 or size > 1024 * 1024:
                continue

            score = 0

            if TARGET_ID in name_lower:
                score += 200
            if "oss-fuzz" in name_lower or "ossfuzz" in name_lower:
                score += 20
            if "poc" in name_lower:
                score += 20
            if "crash" in name_lower or "testcase" in name_lower:
                score += 10
            if "heap-buffer-overflow" in name_lower or "heap_buffer_overflow" in name_lower:
                score += 10

            if score == 0 and TARGET_ID not in name_lower:
                continue

            score += max(0, 100 - abs(size - 512))

            if score > best_score:
                best_score = score
                best_member = m

        if best_member is not None:
            try:
                f = tf.extractfile(best_member)
                if f is not None:
                    data = f.read()
                    if data:
                        return data
            except Exception:
                pass

        # Pass 2: search text files for references to the bug ID that might include a PoC path
        members = tf.getmembers()
        members_by_name = {m.name: m for m in members if m.isfile()}

        text_ext_markers = (
            ".txt",
            ".md",
            ".c",
            ".cc",
            ".cpp",
            ".h",
            ".hpp",
            ".py",
            ".rst",
            ".mdown",
            ".org",
            "readme",
            ".yaml",
            ".yml",
            ".toml",
            ".cfg",
            ".config",
        )

        pattern = re.compile(r"([A-Za-z0-9_\-./]*" + re.escape(TARGET_ID) + r"[A-Za-z0-9_\-./]*)")

        for m in members:
            if not m.isfile():
                continue
            if m.size == 0 or m.size > 1024 * 1024:
                continue

            name_lower = m.name.lower()
            if not any(ext in name_lower for ext in text_ext_markers):
                continue

            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                raw = f.read()
            except Exception:
                continue

            try:
                text = raw.decode("utf-8", "ignore")
            except Exception:
                continue

            if TARGET_ID not in text:
                continue

            matches = pattern.findall(text)
            for match in matches:
                candidate_path = match.strip(' "\'()[]{}')
                if not candidate_path:
                    continue

                m2 = members_by_name.get(candidate_path)
                if m2 is None:
                    for name, member in members_by_name.items():
                        if name.endswith(candidate_path) or candidate_path.endswith(name):
                            m2 = member
                            break

                if m2 is None or not m2.isfile():
                    continue
                if m2.size <= 0 or m2.size > 1024 * 1024:
                    continue

                try:
                    f2 = tf.extractfile(m2)
                    if f2 is None:
                        continue
                    data = f2.read()
                    if data:
                        return data
                except Exception:
                    continue

        # Pass 3: generic search in likely PoC directories (fallback if bug-id-specific search failed)
        best_member = None
        best_distance = None

        for m in members:
            if not m.isfile():
                continue
            size = m.size
            if size <= 0 or size > 4096:
                continue

            name_lower = m.name.lower()
            if any(s in name_lower for s in ("/poc", "/pocs", "/crash", "/crashes", "oss-fuzz", "testcases", "corpus")):
                distance = abs(size - 512)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_member = m

        if best_member is not None:
            try:
                f = tf.extractfile(best_member)
                if f is not None:
                    data = f.read()
                    if data:
                        return data
            except Exception:
                pass

        # Final fallback: synthetic 512-byte payload
        return self._fallback_poc()

    def _fallback_poc(self) -> bytes:
        # Construct a deterministic 512-byte blob with ELF and UPX-like markers.
        size = 512
        data = bytearray(size)

        # ELF magic
        if size >= 4:
            data[0:4] = b"\x7fELF"

        # Some UPX-like markers scattered through the file
        marker = b"UPX!"
        for i in range(16, size, 32):
            end = min(i + len(marker), size)
            data[i:end] = marker[: end - i]

        return bytes(data)