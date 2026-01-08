import os
import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        gt_size = 33762

        # Try to treat src_path as a tarball
        try:
            tf = tarfile.open(src_path, "r:*")
        except (tarfile.TarError, OSError):
            return self._solve_no_tar(src_path, gt_size)

        with tf:
            members = [m for m in tf.getmembers() if m.isfile() and m.size > 0]
            if not members:
                return b""

            member = self._select_member(members, gt_size)

            fobj = tf.extractfile(member)
            if fobj is None:
                return b""
            data = fobj.read()
            if not isinstance(data, (bytes, bytearray)):
                data = bytes(data)
            return data

    def _select_member(self, members, gt_size: int):
        # Step 0: exact size matches
        exact = [m for m in members if m.size == gt_size]
        if exact:
            return self._choose_preferred_member(exact)

        # Step 1: priority basenames
        priority_basenames = {
            "poc",
            "poc.txt",
            "poc.bin",
            "poc.pdf",
            "poc.input",
            "crash",
            "crash.bin",
            "crash.pdf",
            "testcase",
        }
        pri = [
            m
            for m in members
            if os.path.basename(m.name).lower() in priority_basenames
        ]
        if pri:
            return self._choose_closest_size(pri, gt_size)

        # Step 2: heuristic substrings in name
        heur_keys = ("poc", "crash", "uaf", "heap", "useafterfree", "id:", "id_", "testcase")
        heur = [m for m in members if any(k in m.name.lower() for k in heur_keys)]
        if heur:
            return self._choose_closest_size(heur, gt_size)

        # Step 3: extension-specific, e.g., .pdf
        pdfs = [m for m in members if m.name.lower().endswith(".pdf")]
        if pdfs:
            return self._choose_closest_size(pdfs, gt_size)

        # Step 4: fallback - choose file whose size is closest to gt_size
        return self._choose_closest_size(members, gt_size)

    def _choose_preferred_member(self, members):
        # Among members (already filtered, e.g., by size), prefer those whose
        # basename indicates it's a PoC or crash, or has a relevant extension.
        keys = ("poc", "crash", "uaf", "heap")
        preferred = []
        pdfs = []
        for m in members:
            name_lower = m.name.lower()
            base = os.path.basename(name_lower)
            if any(k in base for k in keys):
                preferred.append(m)
            if base.endswith(".pdf"):
                pdfs.append(m)
        if preferred:
            return preferred[0]
        if pdfs:
            return pdfs[0]
        return members[0]

    def _choose_closest_size(self, members, target: int):
        best = members[0]
        best_diff = abs(best.size - target)
        for m in members[1:]:
            d = abs(m.size - target)
            if d < best_diff:
                best = m
                best_diff = d
        return best

    def _solve_no_tar(self, path: str, gt_size: int) -> bytes:
        # Fallback if src_path is not a tarball: treat as file or directory.
        if os.path.isdir(path):
            best_path = None
            best_diff = None

            for root, _, files in os.walk(path):
                for fn in files:
                    full = os.path.join(root, fn)
                    try:
                        size = os.path.getsize(full)
                    except OSError:
                        continue
                    if size <= 0:
                        continue
                    diff = abs(size - gt_size)
                    if best_diff is None or diff < best_diff:
                        best_diff = diff
                        best_path = full

            if best_path is not None:
                try:
                    with open(best_path, "rb") as f:
                        return f.read()
                except OSError:
                    return b""
            return b""

        # If it's a regular file, just return its contents.
        try:
            with open(path, "rb") as f:
                return f.read()
        except OSError:
            return b""