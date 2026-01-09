import os
import re
import io
import tarfile
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional


_SOURCE_EXTS = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc", ".inl", ".ipp", ".mm", ".m",
    ".rs", ".go", ".java", ".kt", ".swift",
}
_XML_EXTS = {
    ".xml", ".svg", ".dae", ".x3d", ".kml", ".gml", ".osm", ".urdf", ".sdf", ".rdf", ".plist", ".opf",
    ".xaml", ".xbm", ".xul", ".xsd", ".xsl", ".xslt", ".wsdl", ".dita", ".gpx",
}

_RE_QUERY_ATTR = re.compile(
    r"""\bQuery(?P<ty>Int|Unsigned|Int64|Unsigned64|Bool|Float|Double)Attribute\s*\(\s*"(?P<an>[^"]+)"\s*,"""
)
_RE_ATTR = re.compile(r"""\bAttribute\s*\(\s*"(?P<an>[^"]+)"\s*(?:,|\))""")
_RE_GETATTR = re.compile(r"""\bgetAttribute\s*\(\s*"(?P<an>[^"]+)"\s*\)""", re.IGNORECASE)
_RE_FIRSTCHILD = re.compile(r"""\bFirstChildElement\s*\(\s*"(?P<en>[^"]+)"\s*\)""")
_RE_DOC_FIRSTCHILD = re.compile(r"""\bdoc\s*\.\s*FirstChildElement\s*\(\s*"(?P<en>[^"]+)"\s*\)""")
_RE_XML_DECL = re.compile(r"^\s*<\?xml\b", re.IGNORECASE)
_RE_XML_START = re.compile(r"^\s*<")


def _is_probably_text(b: bytes) -> bool:
    if not b:
        return False
    if b"\x00" in b:
        return False
    sample = b[:4096]
    # allow UTF-8 and general text; reject very high control density
    ctrl = sum(1 for ch in sample if ch < 9 or (13 < ch < 32))
    return ctrl <= max(4, len(sample) // 200)


def _valid_xml_name(s: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][0-9A-Za-z_\-:\.]*$", s))


class _FS:
    def list_files(self) -> List[Tuple[str, int]]:
        raise NotImplementedError

    def read_file(self, relpath: str, max_bytes: Optional[int] = None) -> bytes:
        raise NotImplementedError


class _DirFS(_FS):
    def __init__(self, root: str):
        self.root = root
        self._files = None

    def list_files(self) -> List[Tuple[str, int]]:
        if self._files is not None:
            return self._files
        out = []
        for base, _, files in os.walk(self.root):
            for fn in files:
                p = os.path.join(base, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                rel = os.path.relpath(p, self.root)
                rel = rel.replace("\\", "/")
                out.append((rel, int(st.st_size)))
        self._files = out
        return out

    def read_file(self, relpath: str, max_bytes: Optional[int] = None) -> bytes:
        p = os.path.join(self.root, relpath)
        with open(p, "rb") as f:
            if max_bytes is None:
                return f.read()
            return f.read(max_bytes + 1)


class _TarFS(_FS):
    def __init__(self, tar_path: str):
        self.tar_path = tar_path
        self._members: Dict[str, tarfile.TarInfo] = {}
        self._files: List[Tuple[str, int]] = []
        with tarfile.open(self.tar_path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                name = m.name
                name = name.lstrip("./")
                name = name.replace("\\", "/")
                if not name or name.startswith("../") or "/../" in name:
                    continue
                self._members[name] = m
                self._files.append((name, int(m.size)))

    def list_files(self) -> List[Tuple[str, int]]:
        return self._files

    def read_file(self, relpath: str, max_bytes: Optional[int] = None) -> bytes:
        m = self._members.get(relpath)
        if m is None:
            return b""
        with tarfile.open(self.tar_path, "r:*") as tf:
            f = tf.extractfile(m)
            if f is None:
                return b""
            data = f.read() if max_bytes is None else f.read(max_bytes + 1)
            return data


def _choose_fs(src_path: str) -> _FS:
    if os.path.isdir(src_path):
        return _DirFS(src_path)
    if os.path.isfile(src_path) and tarfile.is_tarfile(src_path):
        return _TarFS(src_path)
    # fallback: treat as directory if possible, else empty
    return _DirFS(src_path) if os.path.isdir(src_path) else _DirFS(os.path.dirname(src_path) or ".")


def _scan_sources(fs: _FS) -> Tuple[Counter, Counter, Counter, Counter, List[Tuple[str, str]]]:
    attr_freq = Counter()
    attr_type_freq = Counter()
    elem_freq = Counter()
    root_freq = Counter()
    pairs = []

    files = fs.list_files()
    for rel, sz in files:
        ext = os.path.splitext(rel.lower())[1]
        if ext not in _SOURCE_EXTS:
            continue
        if sz <= 0 or sz > 2_000_000:
            continue
        b = fs.read_file(rel, max_bytes=1_000_000)
        if not b or len(b) > 1_000_000:
            continue
        if not _is_probably_text(b):
            continue
        try:
            txt = b.decode("latin1", errors="ignore")
        except Exception:
            continue

        # element names
        for m in _RE_FIRSTCHILD.finditer(txt):
            en = m.group("en")
            if en and len(en) <= 64:
                elem_freq[en] += 1

        for m in _RE_DOC_FIRSTCHILD.finditer(txt):
            en = m.group("en")
            if en and len(en) <= 64:
                root_freq[en] += 1

        # attribute names
        for m in _RE_QUERY_ATTR.finditer(txt):
            an = m.group("an")
            ty = m.group("ty")
            if an and len(an) <= 64:
                attr_freq[an] += 3  # prioritize Query*Attribute usage
                attr_type_freq[(an, ty)] += 1

            # try to infer nearby element name
            start = m.start()
            window = txt[max(0, start - 600):start]
            prevs = list(_RE_FIRSTCHILD.finditer(window))
            if prevs:
                en = prevs[-1].group("en")
                if en and an and len(en) <= 64 and len(an) <= 64:
                    pairs.append((en, an))

        # also collect other attribute getters
        for m in _RE_ATTR.finditer(txt):
            an = m.group("an")
            if an and len(an) <= 64:
                attr_freq[an] += 1

        for m in _RE_GETATTR.finditer(txt):
            an = m.group("an")
            if an and len(an) <= 64:
                attr_freq[an] += 1

    return attr_freq, attr_type_freq, elem_freq, root_freq, pairs


def _rank_bad_attrs(attr_freq: Counter, attr_type_freq: Counter, max_k: int = 6) -> List[str]:
    type_weight = {
        "Int": 20,
        "Unsigned": 18,
        "Int64": 16,
        "Unsigned64": 14,
        "Bool": 12,
        "Float": 10,
        "Double": 8,
    }
    score = Counter()
    for (an, ty), c in attr_type_freq.items():
        score[an] += c * type_weight.get(ty, 1)

    for an, c in attr_freq.items():
        score[an] += min(10, c)

    ranked = [an for an, _ in score.most_common()]
    ranked = [a for a in ranked if _valid_xml_name(a)]

    if not ranked:
        ranked = ["width", "height", "x", "y", "z", "size", "count", "index", "offset", "length"]

    # Ensure common numeric-ish ones are tried early
    common = ["width", "height", "x", "y", "z", "r", "g", "b", "a", "size", "count", "index", "offset", "length", "scale", "rotation"]
    for c in common[::-1]:
        if c in ranked:
            ranked.remove(c)
            ranked.insert(0, c)

    out = []
    for a in ranked:
        if a not in out:
            out.append(a)
        if len(out) >= max_k:
            break
    return out


def _looks_like_xml_bytes(b: bytes, rel: str) -> bool:
    if not b:
        return False
    if len(b) < 8:
        return False
    if b"\x00" in b[:4096]:
        return False
    # quick start check
    head = b[:512].decode("latin1", errors="ignore")
    if _RE_XML_DECL.search(head) or _RE_XML_START.search(head):
        # ensure it has at least one tag close
        if b">" in b[:4096] and b"<" in b[:4096]:
            return True
    # extension hint
    ext = os.path.splitext(rel.lower())[1]
    if ext in _XML_EXTS and b"<" in b[:4096] and b">" in b[:4096]:
        return True
    return False


def _select_xml_sample(fs: _FS, bad_attrs: List[str]) -> Optional[Tuple[str, bytes]]:
    files = fs.list_files()
    best = None
    best_score = None

    # Build a fast regex to detect any of the attributes
    attrs_for_regex = [a for a in bad_attrs if _valid_xml_name(a)]
    if attrs_for_regex:
        alt = "|".join(re.escape(a) for a in attrs_for_regex[:24])
        any_attr_re = re.compile(r"(?:\b(?:" + alt + r")\b\s*=\s*['\"])")
    else:
        any_attr_re = None

    for rel, sz in files:
        if sz <= 0 or sz > 300_000:
            continue
        ext = os.path.splitext(rel.lower())[1]
        # Prefer likely XML files
        if ext not in _XML_EXTS and ext not in {".txt", ".dat", ".in", ".out"} and "xml" not in rel.lower() and "svg" not in rel.lower():
            continue
        b = fs.read_file(rel, max_bytes=300_000)
        if not b or len(b) > 300_000:
            continue
        if not _is_probably_text(b):
            continue
        if not _looks_like_xml_bytes(b, rel):
            continue
        txt = b.decode("latin1", errors="ignore")
        if any_attr_re is not None:
            matches = len(any_attr_re.findall(txt))
        else:
            matches = 0

        # score: prefer files containing attributes of interest, then smaller
        score = matches * 200000 - len(b)
        if best is None or score > best_score:
            best = (rel, b)
            best_score = score

    # Fallback: any XML-looking file
    if best is None:
        for rel, sz in files:
            if sz <= 0 or sz > 80_000:
                continue
            b = fs.read_file(rel, max_bytes=80_000)
            if not b or len(b) > 80_000:
                continue
            if not _is_probably_text(b):
                continue
            if not _looks_like_xml_bytes(b, rel):
                continue
            best = (rel, b)
            break

    return best


def _mutate_xml(xml_bytes: bytes, bad_attrs: List[str], inject_elems: List[Tuple[str, List[str]]]) -> bytes:
    txt = xml_bytes.decode("latin1", errors="ignore")

    # Corrupt a few attributes (limited replacements to keep file mostly valid)
    for an in bad_attrs:
        if not _valid_xml_name(an):
            continue
        pat = re.compile(r'(\b' + re.escape(an) + r'\b\s*=\s*)(["\'])([^"\']*)(\2)')
        # replace up to 3 occurrences of each attribute
        def repl(m):
            return m.group(1) + m.group(2) + "x" + m.group(2)
        txt, _ = pat.subn(repl, txt, count=3)

    # Try injecting a triggering element near end, before final closing tag
    if inject_elems:
        inject_parts = []
        for en, attrs in inject_elems[:3]:
            if not _valid_xml_name(en):
                continue
            atts = []
            for a in attrs:
                if _valid_xml_name(a):
                    atts.append(f'{a}="x"')
            if not atts:
                atts = ['x="x"', 'y="x"', 'width="x"', 'height="x"']
            inject_parts.append("<" + en + " " + " ".join(atts[:10]) + "/>")
        injection = "\n" + "\n".join(inject_parts) + "\n"
        m = re.search(r"(</[^>]+>\s*)\Z", txt)
        if m:
            txt = txt[:m.start()] + injection + txt[m.start():]

    out = txt.encode("latin1", errors="ignore")
    if out:
        return out
    return xml_bytes


def _build_synthetic_xml(root_name: str, element_names: List[str], bad_attrs: List[str]) -> bytes:
    rn = root_name if _valid_xml_name(root_name) else "root"
    elems = [e for e in element_names if _valid_xml_name(e)]
    if not elems:
        elems = ["svg", "rect", "path", "item", "node", "object"]

    # If root resembles svg-ish, add namespace; otherwise keep plain.
    if rn.lower() == "svg":
        root_open = '<svg xmlns="http://www.w3.org/2000/svg" width="x" height="x">'
        root_close = "</svg>"
        rn = "svg"
    else:
        root_open = f"<{rn}>"
        root_close = f"</{rn}>"

    attrs = [a for a in bad_attrs if _valid_xml_name(a)]
    if not attrs:
        attrs = ["x", "y", "width", "height", "size", "count", "index"]

    # Create a few elements with invalid numeric conversions
    parts = ['<?xml version="1.0"?>', root_open]
    for i, en in enumerate(elems[:8]):
        a_subset = attrs[i % len(attrs):] + attrs[:i % len(attrs)]
        a_subset = a_subset[:6]
        attr_str = " ".join(f'{a}="x"' for a in a_subset)
        parts.append(f"<{en} {attr_str}/>")
    parts.append(root_close)
    return ("\n".join(parts) + "\n").encode("utf-8", errors="ignore")


class Solution:
    def solve(self, src_path: str) -> bytes:
        fs = _choose_fs(src_path)

        attr_freq, attr_type_freq, elem_freq, root_freq, pairs = _scan_sources(fs)
        bad_attrs = _rank_bad_attrs(attr_freq, attr_type_freq, max_k=6)

        # Choose likely root
        root_name = None
        if root_freq:
            root_name = root_freq.most_common(1)[0][0]
        else:
            for cand in ["svg", "COLLADA", "robot", "sdf", "model", "plist"]:
                if cand in elem_freq:
                    root_name = cand
                    break
        if not root_name:
            root_name = "root"

        # Build injection element suggestions from pairs
        pair_counter = Counter((en, an) for en, an in pairs if _valid_xml_name(en) and _valid_xml_name(an))
        elem_to_attrs = defaultdict(list)
        for (en, an), _c in pair_counter.most_common(50):
            if an not in elem_to_attrs[en]:
                elem_to_attrs[en].append(an)

        inject_elems = []
        for en, _cnt in elem_freq.most_common(20):
            if en in elem_to_attrs and _valid_xml_name(en):
                inject_elems.append((en, elem_to_attrs[en] + bad_attrs))
        if not inject_elems:
            # try best element names even without pairs
            for en, _cnt in elem_freq.most_common(10):
                if _valid_xml_name(en):
                    inject_elems.append((en, bad_attrs))

        sample = _select_xml_sample(fs, bad_attrs)
        if sample is not None:
            _, xml_bytes = sample
            mutated = _mutate_xml(xml_bytes, bad_attrs, inject_elems)
            if mutated:
                return mutated

        # Fallback: synthetic XML
        element_names = [en for en, _ in elem_freq.most_common(30)]
        return _build_synthetic_xml(root_name, element_names, bad_attrs)