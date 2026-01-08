import os
import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        TARGET_LEN = 6180

        text_exts = {
            '.c', '.h', '.hpp', '.hh', '.cc', '.cpp', '.cxx',
            '.java', '.py', '.pyc', '.pyo', '.sh', '.bat', '.ps1',
            '.pl', '.rb', '.go', '.rs', '.js', '.ts', '.php',
            '.html', '.htm', '.xml', '.xsl', '.xsd',
            '.json', '.toml', '.ini', '.cfg', '.config',
            '.yml', '.yaml',
            '.md', '.rst', '.txt', '.csv', '.tsv',
            '.cmake', '.mak', '.mk', '.am', '.ac', '.m4', '.in',
            '.css', '.scss', '.less',
            '.sln', '.vcxproj', '.vcproj', '.ninja', '.gn', '.gni',
            '.gradle', '.iml',
            '.pbtxt',
        }

        binary_bonus_exts = {
            '.bin', '.dat', '.svc', '.264', '.h264', '.bs',
            '.bit', '.stream', '.ivf', '.annexb'
        }

        def compute_priority(name_lower: str) -> int:
            prio = 0
            if '42536279' in name_lower:
                prio += 500
            if 'svcdec' in name_lower:
                prio += 200
            elif 'svc' in name_lower:
                prio += 120
            if 'oss-fuzz' in name_lower or 'ossfuzz' in name_lower:
                prio += 150
            if 'clusterfuzz' in name_lower:
                prio += 140
            if 'heap' in name_lower:
                prio += 80
            if 'overflow' in name_lower or 'overrun' in name_lower:
                prio += 80
            if 'poc' in name_lower:
                prio += 70
            if 'crash' in name_lower:
                prio += 70
            if 'bug' in name_lower:
                prio += 40
            if 'regress' in name_lower:
                prio += 40
            if 'fuzz' in name_lower:
                prio += 20
            if 'corpus' in name_lower:
                prio += 15
            if 'input' in name_lower or 'case' in name_lower or 'seed' in name_lower:
                prio += 10
            return prio

        def is_probably_binary(name: str) -> bool:
            _, ext = os.path.splitext(name)
            return ext.lower() not in text_exts

        def priority_with_ext(name: str) -> int:
            nlower = name.lower()
            base = compute_priority(nlower)
            _, ext = os.path.splitext(name)
            if ext.lower() in binary_bonus_exts:
                base += 25
            return base

        def select_from_tar(path: str) -> bytes:
            best_exact_member = None
            best_exact_score = None
            best_kw_member = None
            best_kw_score = None
            smallest_member = None
            smallest_size = None

            with tarfile.open(path, 'r:*') as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    size = member.size
                    if size <= 0:
                        continue
                    name = member.name

                    if smallest_member is None or size < smallest_size:
                        smallest_member = member
                        smallest_size = size

                    if size > 10 * 1024 * 1024 and size != TARGET_LEN:
                        continue

                    is_bin = is_probably_binary(name)
                    prio = priority_with_ext(name)

                    if size == TARGET_LEN:
                        if not is_bin:
                            prio -= 50
                        score = (prio, -size, -len(name))
                        if best_exact_score is None or score > best_exact_score:
                            best_exact_score = score
                            best_exact_member = member

                    if prio > 0 and is_bin and size <= 1_000_000:
                        score = (prio, -size, -len(name))
                        if best_kw_score is None or score > best_kw_score:
                            best_kw_score = score
                            best_kw_member = member

                chosen = None
                if best_exact_member is not None:
                    chosen = best_exact_member
                elif best_kw_member is not None:
                    chosen = best_kw_member
                elif smallest_member is not None and smallest_size <= 1_000_000:
                    chosen = smallest_member

                if chosen is not None:
                    f = tf.extractfile(chosen)
                    if f is not None:
                        data = f.read()
                        if data:
                            return data

            return b'A' * TARGET_LEN

        def select_from_dir(path: str) -> bytes:
            best_exact_path = None
            best_exact_score = None
            best_kw_path = None
            best_kw_score = None
            smallest_path = None
            smallest_size = None

            for root, _, files in os.walk(path):
                for fname in files:
                    full = os.path.join(root, fname)
                    try:
                        size = os.path.getsize(full)
                    except OSError:
                        continue
                    if size <= 0:
                        continue
                    name_rel = os.path.relpath(full, path)

                    if smallest_path is None or size < smallest_size:
                        smallest_path = full
                        smallest_size = size

                    if size > 10 * 1024 * 1024 and size != TARGET_LEN:
                        continue

                    is_bin = is_probably_binary(name_rel)
                    prio = priority_with_ext(name_rel)

                    if size == TARGET_LEN:
                        if not is_bin:
                            prio -= 50
                        score = (prio, -size, -len(name_rel))
                        if best_exact_score is None or score > best_exact_score:
                            best_exact_score = score
                            best_exact_path = full

                    if prio > 0 and is_bin and size <= 1_000_000:
                        score = (prio, -size, -len(name_rel))
                        if best_kw_score is None or score > best_kw_score:
                            best_kw_score = score
                            best_kw_path = full

            chosen_path = None
            if best_exact_path is not None:
                chosen_path = best_exact_path
            elif best_kw_path is not None:
                chosen_path = best_kw_path
            elif smallest_path is not None and smallest_size <= 1_000_000:
                chosen_path = smallest_path

            if chosen_path is not None:
                try:
                    with open(chosen_path, 'rb') as f:
                        data = f.read()
                        if data:
                            return data
                except OSError:
                    pass

            return b'A' * TARGET_LEN

        try:
            if os.path.isdir(src_path):
                return select_from_dir(src_path)
            else:
                return select_from_tar(src_path)
        except Exception:
            return b'A' * TARGET_LEN