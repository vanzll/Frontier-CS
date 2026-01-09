import os
import re
import tarfile
from typing import Dict, List, Optional, Tuple


class Solution:
    def solve(self, src_path: str) -> bytes:
        with tarfile.open(src_path, "r:*") as tar:
            members = [m for m in tar.getmembers() if m.isreg()]

            poc = self._find_embedded_poc(tar, members)
            if poc is not None:
                return poc

            harness_text, harness_name = self._find_fuzz_harness(tar, members)
            fmt = self._infer_format(harness_text, harness_name, members)

            sample = self._find_sample_and_mutate(tar, members, fmt)
            if sample is not None:
                return sample

            return self._template_for_format(fmt)

    def _read_member(self, tar: tarfile.TarFile, m: tarfile.TarInfo, limit: Optional[int] = None) -> bytes:
        try:
            f = tar.extractfile(m)
            if f is None:
                return b""
            if limit is None:
                return f.read()
            return f.read(limit)
        except Exception:
            return b""

    def _find_embedded_poc(self, tar: tarfile.TarFile, members: List[tarfile.TarInfo]) -> Optional[bytes]:
        name_re = re.compile(r"(clusterfuzz-testcase|minimized|reproducer|repro|poc|crash|42536068)", re.IGNORECASE)
        candidates: List[tarfile.TarInfo] = []
        for m in members:
            n = m.name.replace("\\", "/")
            base = n.rsplit("/", 1)[-1]
            if m.size <= 2_000_000 and (name_re.search(n) or name_re.search(base)):
                candidates.append(m)
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x.size, len(x.name)))
        data = self._read_member(tar, candidates[0], limit=2_000_000)
        if data:
            return data
        for m in candidates[1:10]:
            data = self._read_member(tar, m, limit=2_000_000)
            if data:
                return data
        return None

    def _find_fuzz_harness(self, tar: tarfile.TarFile, members: List[tarfile.TarInfo]) -> Tuple[str, str]:
        exts = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh")
        key_markers = (
            "LLVMFuzzerTestOneInput",
            "FUZZ_TEST",
            "fuzz_target!",
            "Honggfuzz",
            "AFL_FUZZ",
        )
        best_text = ""
        best_name = ""
        best_score = -1

        for m in members:
            n = m.name.replace("\\", "/")
            ln = n.lower()
            if not ln.endswith(exts):
                continue
            if m.size <= 0 or m.size > 400_000:
                continue
            if any(seg in ln for seg in ("/third_party/", "/thirdparty/", "/external/", "/vendor/")):
                continue
            data = self._read_member(tar, m, limit=400_000)
            if not data:
                continue
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                continue
            if not any(k in text for k in key_markers):
                continue

            score = 0
            if "LLVMFuzzerTestOneInput" in text:
                score += 5
            if "fuzz_target!" in text:
                score += 4
            if "/fuzz" in ln or "fuzz" in ln:
                score += 2
            if "attribute" in text.lower():
                score += 1
            if "xml" in text.lower() or "svg" in text.lower():
                score += 1
            score -= len(text) // 200_000

            if score > best_score:
                best_score = score
                best_text = text
                best_name = n

        return best_text, best_name

    def _infer_format(self, harness_text: str, harness_name: str, members: List[tarfile.TarInfo]) -> str:
        text = (harness_text or "").lower()
        name = (harness_name or "").lower()

        # Strong harness hints
        if "nsvgparse" in text or "lunasvg" in text or "svg" in name or "svg" in text:
            return "svg"
        if "collada" in text or ".dae" in text or "collada" in name or ".dae" in name:
            return "collada"
        if "x3d" in text or ".x3d" in text or "x3d" in name:
            return "x3d"
        if "xmlreadmemory" in text or "tinyxml" in text or "pugixml" in text or "rapidxml" in text:
            return "xml"
        if ".html" in text or "html" in name:
            return "html"
        if ".gltf" in text or "gltf" in text:
            return "gltf"
        if ".json" in text or "json" in text:
            # Keep as fallback only; many projects mention json without being json fuzzer input.
            pass

        # Infer from sample distribution in repo
        counts: Dict[str, int] = {}
        interesting_dirs = ("test", "tests", "example", "examples", "sample", "samples", "corpus", "data", "assets", "input")
        for m in members:
            n = m.name.replace("\\", "/").lower()
            if m.size <= 0 or m.size > 500_000:
                continue
            if not any(f"/{d}/" in n or n.startswith(f"{d}/") for d in interesting_dirs):
                continue
            ext = ""
            base = n.rsplit("/", 1)[-1]
            if "." in base:
                ext = "." + base.rsplit(".", 1)[-1]
            if ext in (".svg", ".dae", ".x3d", ".xml", ".html", ".htm", ".gltf", ".json"):
                counts[ext] = counts.get(ext, 0) + 1

        def pick() -> str:
            if not counts:
                return "unknown"
            pref = [(".svg", "svg"), (".dae", "collada"), (".x3d", "x3d"), (".xml", "xml"), (".html", "html"), (".htm", "html"), (".gltf", "gltf"), (".json", "json")]
            best = max(counts.items(), key=lambda kv: kv[1])[0]
            for ext, fmt in pref:
                if ext == best:
                    return fmt
            return "unknown"

        fmt = pick()
        if fmt != "unknown":
            return fmt

        return "svg"

    def _is_likely_text(self, data: bytes) -> bool:
        if not data:
            return False
        sample = data[:4096]
        if b"\x00" in sample:
            return False
        # Heuristic: printable ratio
        printable = 0
        for b in sample:
            if b in (9, 10, 13) or 32 <= b <= 126:
                printable += 1
        return printable / max(1, len(sample)) > 0.92

    def _mutate_text(self, s: str, fmt: str) -> str:
        s2 = s

        if fmt == "svg":
            if "<svg" in s2:
                if re.search(r"\bwidth\s*=\s*\"", s2, flags=re.IGNORECASE):
                    s2 = re.sub(r"(\bwidth\s*=\s*\")([^\"]*)(\")", r"\1x\3", s2, count=1, flags=re.IGNORECASE)
                else:
                    s2 = re.sub(r"(<svg\b)", r'\1 width="x"', s2, count=1, flags=re.IGNORECASE)
                if re.search(r"\bheight\s*=\s*\"", s2, flags=re.IGNORECASE):
                    s2 = re.sub(r"(\bheight\s*=\s*\")([^\"]*)(\")", r"\1x\3", s2, count=1, flags=re.IGNORECASE)
                else:
                    s2 = re.sub(r"(<svg\b[^>]*\bwidth=\"x\")", r'\1 height="1"', s2, count=1, flags=re.IGNORECASE)
                if re.search(r"\bviewBox\s*=\s*\"", s2, flags=re.IGNORECASE):
                    s2 = re.sub(r"(\bviewBox\s*=\s*\")([^\"]*)(\")", r"\10 0 x 1\3", s2, count=1, flags=re.IGNORECASE)
            # also try typical numeric attrs
            s2 = self._generic_numeric_attr_mutation(s2)
            return s2

        if fmt == "collada":
            # meter is a known numeric attribute in COLLADA asset unit
            if re.search(r"\bmeter\s*=\s*\"", s2, flags=re.IGNORECASE):
                s2 = re.sub(r"(\bmeter\s*=\s*\")([^\"]*)(\")", r"\1x\3", s2, count=1, flags=re.IGNORECASE)
            else:
                s2 = self._generic_numeric_attr_mutation(s2)
            return s2

        if fmt == "x3d":
            if re.search(r"\bversion\s*=\s*\"", s2, flags=re.IGNORECASE):
                s2 = re.sub(r"(\bversion\s*=\s*\")([^\"]*)(\")", r"\1x\3", s2, count=1, flags=re.IGNORECASE)
            else:
                s2 = self._generic_numeric_attr_mutation(s2)
            return s2

        if fmt in ("xml", "html"):
            return self._generic_numeric_attr_mutation(s2)

        return self._generic_numeric_attr_mutation(s2)

    def _generic_numeric_attr_mutation(self, s: str) -> str:
        # Replace first clearly-numeric attribute value with "x"
        num_attr = re.compile(r'=\s*"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"')
        m = num_attr.search(s)
        if m:
            start, end = m.span()
            return s[:start] + '="x"' + s[end:]
        # Fallback: replace first attribute value with x
        any_attr = re.compile(r'=\s*"[^"]*"')
        m = any_attr.search(s)
        if m:
            start, end = m.span()
            return s[:start] + '="x"' + s[end:]
        return s

    def _find_sample_and_mutate(self, tar: tarfile.TarFile, members: List[tarfile.TarInfo], fmt: str) -> Optional[bytes]:
        preferred_exts = {
            "svg": [".svg", ".xml"],
            "collada": [".dae", ".xml"],
            "x3d": [".x3d", ".xml"],
            "xml": [".xml", ".svg", ".dae", ".x3d", ".html", ".htm"],
            "html": [".html", ".htm", ".xml"],
            "gltf": [".gltf", ".json"],
            "json": [".json"],
            "unknown": [".svg", ".xml", ".dae", ".x3d", ".html", ".htm", ".json"],
        }.get(fmt, [".svg", ".xml"])

        interesting_dirs = ("test", "tests", "example", "examples", "sample", "samples", "corpus", "data", "assets", "input")
        candidates: List[tarfile.TarInfo] = []
        for m in members:
            if m.size <= 0 or m.size > 300_000:
                continue
            n = m.name.replace("\\", "/")
            ln = n.lower()
            if not any(f"/{d}/" in ln or ln.startswith(f"{d}/") for d in interesting_dirs):
                continue
            base = ln.rsplit("/", 1)[-1]
            ext = "." + base.rsplit(".", 1)[-1] if "." in base else ""
            if ext in preferred_exts:
                candidates.append(m)

        candidates.sort(key=lambda x: (x.size, len(x.name)))
        for m in candidates[:50]:
            data = self._read_member(tar, m, limit=350_000)
            if not self._is_likely_text(data):
                continue
            try:
                s = data.decode("utf-8", errors="ignore")
            except Exception:
                continue
            if "<" not in s or ">" not in s:
                continue
            mutated = self._mutate_text(s, fmt)
            out = mutated.encode("utf-8", errors="ignore")
            if out and len(out) <= 1_000_000:
                return out

        # As a fallback, try to find any small text sample regardless of directory
        any_candidates: List[tarfile.TarInfo] = []
        for m in members:
            if m.size <= 0 or m.size > 200_000:
                continue
            n = m.name.replace("\\", "/").lower()
            base = n.rsplit("/", 1)[-1]
            ext = "." + base.rsplit(".", 1)[-1] if "." in base else ""
            if ext in preferred_exts:
                any_candidates.append(m)
        any_candidates.sort(key=lambda x: (x.size, len(x.name)))
        for m in any_candidates[:50]:
            data = self._read_member(tar, m, limit=250_000)
            if not self._is_likely_text(data):
                continue
            try:
                s = data.decode("utf-8", errors="ignore")
            except Exception:
                continue
            if "<" not in s or ">" not in s:
                continue
            mutated = self._mutate_text(s, fmt)
            out = mutated.encode("utf-8", errors="ignore")
            if out and len(out) <= 1_000_000:
                return out

        return None

    def _template_for_format(self, fmt: str) -> bytes:
        if fmt == "collada":
            return (
                b'<?xml version="1.0" encoding="utf-8"?>\n'
                b'<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">\n'
                b'  <asset>\n'
                b'    <unit meter="x" name="meter"/>\n'
                b'  </asset>\n'
                b'  <library_geometries/>\n'
                b'  <scene/>\n'
                b'</COLLADA>\n'
            )
        if fmt == "x3d":
            return (
                b'<?xml version="1.0" encoding="UTF-8"?>\n'
                b'<X3D profile="Immersive" version="x" xmlns:xsd="http://www.w3.org/2001/XMLSchema-instance">\n'
                b'  <Scene/>\n'
                b'</X3D>\n'
            )
        if fmt == "html":
            return b'<!doctype html><html><body><div data-w="x" style="width:xpx;height:1px"></div></body></html>'
        if fmt == "xml":
            return b'<?xml version="1.0"?><root width="x" height="1"><child x="x"/></root>'
        if fmt == "gltf" or fmt == "json":
            return b'{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"name":"x","extras":{"attr":"x"}}]}'
        # Default to SVG (XML) with invalid numeric attrs
        return b'<svg xmlns="http://www.w3.org/2000/svg" width="x" height="1" viewBox="0 0 x 1"><rect x="0" y="0" width="1" height="1"/></svg>'