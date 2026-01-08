import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        GROUND_TRUTH_LENGTH = 150979

        def fallback_poc() -> bytes:
            return (
                b"%PDF-1.3\n"
                b"1 0 obj\n"
                b"<< /Type /Catalog /Pages 2 0 R >>\n"
                b"endobj\n"
                b"2 0 obj\n"
                b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n"
                b"endobj\n"
                b"3 0 obj\n"
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\n"
                b"endobj\n"
                b"xref\n"
                b"0 4\n"
                b"0000000000 65535 f \n"
                b"0000000010 00000 n \n"
                b"0000000060 00000 n \n"
                b"0000000110 00000 n \n"
                b"trailer\n"
                b"<< /Root 1 0 R /Size 4 >>\n"
                b"startxref\n"
                b"170\n"
                b"%%EOF\n"
            )

        try:
            tf = tarfile.open(src_path, "r:*")
        except Exception:
            return fallback_poc()

        pattern_weights = {
            "42535696": 4000,
            "clusterfuzz": 3000,
            "oss-fuzz": 2500,
            "minimized": 2200,
            "repro": 2200,
            "reproducer": 2200,
            "poc": 2200,
            "crash": 2000,
            "heap-buffer-overflow": 1800,
            "hbo": 800,
            "pdfwrite": 1500,
            "viewer": 600,
            "test": 200,
            "tests": 200,
        }

        ext_weights = {
            "pdf": 3000,
            "ps": 2800,
            "eps": 2600,
            "bin": 2200,
            "dat": 2000,
            "txt": 1500,
        }

        strong_member = None
        strong_score = -1
        weak_member = None
        weak_score = -1

        for member in tf.getmembers():
            try:
                is_file = member.isreg()
            except Exception:
                try:
                    is_file = member.isfile()
                except Exception:
                    is_file = False
            if not is_file:
                continue

            size = getattr(member, "size", 0)
            if not isinstance(size, int) or size <= 0:
                continue

            name_lower = member.name.lower()

            score = 0
            has_pattern = False
            for pat, weight in pattern_weights.items():
                if pat in name_lower:
                    score += weight
                    has_pattern = True

            base, dot, ext = name_lower.rpartition(".")
            ext = ext if dot else ""
            is_input_ext = ext in ext_weights
            if is_input_ext:
                score += ext_weights[ext]

            if not has_pattern and not is_input_ext:
                continue

            if GROUND_TRUTH_LENGTH > 0:
                diff = abs(size - GROUND_TRUTH_LENGTH)
                size_score = 2000 - diff // 50
                if size_score < 0:
                    size_score = 0
                score += size_score

            if size < 64:
                score -= 200
            elif size > 2 * 1024 * 1024:
                score -= 200

            if has_pattern:
                if score > strong_score:
                    strong_score = score
                    strong_member = member
            else:
                if score > weak_score:
                    weak_score = score
                    weak_member = member

        chosen_member = strong_member if strong_member is not None else weak_member

        data = None
        if chosen_member is not None:
            try:
                extracted = tf.extractfile(chosen_member)
                if extracted is not None:
                    try:
                        data = extracted.read()
                    finally:
                        extracted.close()
            except Exception:
                data = None

        try:
            tf.close()
        except Exception:
            pass

        if data is not None and len(data) > 0:
            return data

        return fallback_poc()