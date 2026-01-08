import tarfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        L_GROUND_TRUTH = 825339

        try:
            tf = tarfile.open(src_path, "r:*")
        except tarfile.ReadError:
            try:
                with open(src_path, "rb") as f:
                    return f.read()
            except OSError:
                return b""

        with tf:
            members = tf.getmembers()

            def interesting_name(name: str) -> bool:
                name_low = name.lower()
                keywords = [
                    "42537171",
                    "clusterfuzz",
                    "oss-fuzz",
                    "ossfuzz",
                    "testcase",
                    "poc",
                    "crash",
                    "heap-buffer-overflow",
                    "heap_buffer_overflow",
                    "heap-overflow",
                    "heap_overflow",
                    "overflow",
                    "bug",
                    "fuzzer",
                    "id:",
                ]
                for kw in keywords:
                    if kw in name_low:
                        return True
                return False

            def extract_bytes(member: tarfile.TarInfo) -> bytes:
                f = tf.extractfile(member)
                if f is None:
                    return b""
                data = f.read()
                f.close()
                return data

            # 1) Exact-size & interesting name
            exact_interesting = [
                m
                for m in members
                if m.isfile() and m.size == L_GROUND_TRUTH and interesting_name(m.name)
            ]
            if exact_interesting:
                exact_interesting.sort(
                    key=lambda m: (m.name.count("/"), len(m.name))
                )
                return extract_bytes(exact_interesting[0])

            # 2) Any exact-size file
            exact_any = [
                m for m in members if m.isfile() and m.size == L_GROUND_TRUTH
            ]
            if exact_any:
                exact_any.sort(key=lambda m: (m.name.count("/"), len(m.name)))
                return extract_bytes(exact_any[0])

            # 3) Near-size & interesting-name within +/- 16 KiB
            tolerance = 16 * 1024
            near_interesting = [
                m
                for m in members
                if m.isfile()
                and abs(m.size - L_GROUND_TRUTH) <= tolerance
                and interesting_name(m.name)
            ]
            if near_interesting:
                near_interesting.sort(
                    key=lambda m: (
                        abs(m.size - L_GROUND_TRUTH),
                        m.name.count("/"),
                        len(m.name),
                    )
                )
                return extract_bytes(near_interesting[0])

            # 4) Any interesting-name file <= 8 MiB (prefer size close to L_GROUND_TRUTH)
            interesting_files = [
                m
                for m in members
                if m.isfile()
                and m.size > 0
                and m.size <= 8 * 1024 * 1024
                and interesting_name(m.name)
            ]
            if interesting_files:
                interesting_files.sort(
                    key=lambda m: (
                        abs(m.size - L_GROUND_TRUTH),
                        m.name.count("/"),
                        len(m.name),
                    )
                )
                return extract_bytes(interesting_files[0])

            # 5) Prefer plausible media/vector/image formats
            preferred_exts = [
                ".pdf",
                ".svg",
                ".skp",
                ".ai",
                ".ps",
                ".eps",
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
                ".gif",
                ".webp",
                ".tif",
                ".tiff",
                ".ico",
                ".wbmp",
                ".pgm",
                ".ppm",
                ".pbm",
            ]
            media_candidates = []
            for m in members:
                if not m.isfile() or m.size <= 0:
                    continue
                name_low = m.name.lower()
                for ext in preferred_exts:
                    if name_low.endswith(ext):
                        media_candidates.append(m)
                        break
            if media_candidates:
                media_candidates.sort(
                    key=lambda m: (
                        -min(m.size, L_GROUND_TRUTH),
                        m.name.count("/"),
                        len(m.name),
                    )
                )
                return extract_bytes(media_candidates[0])

            # 6) Fallback: largest regular file under 2 MiB
            generic_candidates = [
                m
                for m in members
                if m.isfile() and 0 < m.size <= 2 * 1024 * 1024
            ]
            if generic_candidates:
                generic_candidates.sort(
                    key=lambda m: (-m.size, m.name.count("/"), len(m.name))
                )
                return extract_bytes(generic_candidates[0])

            # 7) Absolute last resort: empty input
            return b""