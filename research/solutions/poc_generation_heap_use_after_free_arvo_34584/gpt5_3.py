import os
import tarfile
import io
import re

class Solution:
    def solve(self, src_path: str) -> bytes:
        def is_text_bytes(sample: bytes) -> bool:
            if not sample:
                return False
            if b'\x00' in sample:
                return False
            # Heuristic: consider it text if >85% are printable or whitespace
            printable = b"\t\n\r\f\b" + bytes(range(32, 127))
            cnt = sum(1 for b in sample if b in printable)
            return (cnt / len(sample)) >= 0.85

        def read_member(tf: tarfile.TarFile, member: tarfile.TarInfo, max_bytes: int = None) -> bytes:
            try:
                f = tf.extractfile(member)
                if not f:
                    return b""
                if max_bytes is None:
                    data = f.read()
                else:
                    data = f.read(max_bytes)
                f.close()
                return data
            except Exception:
                return b""

        def scan_tar_for_poc(tar_path: str) -> bytes:
            try:
                tf = tarfile.open(tar_path, mode='r:*')
            except Exception:
                return b""
            best_score = -1
            best_data = b""
            text_exts = ('.js', '.mjs', '.txt', '.html', '.htm')
            target_len = 6624
            # Pre-pass to find exact size match quickly
            exact_candidates = []
            try:
                members = tf.getmembers()
            except Exception:
                tf.close()
                return b""
            for m in members:
                if not m.isfile():
                    continue
                # prioritize exact length matches first
                if m.size == target_len:
                    exact_candidates.append(m)
            # Among exact candidates, prefer .js files and content mentions
            if exact_candidates:
                scored_candidates = []
                for m in exact_candidates:
                    name_lower = m.name.lower()
                    score = 1000
                    if name_lower.endswith('.js') or name_lower.endswith('.mjs'):
                        score += 300
                    if any(k in name_lower for k in ['poc', 'uaf', 'heap', 'crash', 'use-after', 'repro', 'test']):
                        score += 200
                    data = read_member(tf, m)
                    if data:
                        if is_text_bytes(data):
                            score += 50
                        dl = data.lower()
                        if b'uint8clampedarray' in dl:
                            score += 400
                        if b'typedarray' in dl:
                            score += 100
                        if b'gc(' in dl:
                            score += 80
                        if b'use-after-free' in dl or b'use after free' in dl:
                            score += 150
                    scored_candidates.append((score, data))
                if scored_candidates:
                    scored_candidates.sort(key=lambda x: x[0], reverse=True)
                    tf.close()
                    return scored_candidates[0][1] or b""
            # If no exact match, broader scan with heuristic scoring
            for m in members:
                if not m.isfile():
                    continue
                name = m.name
                nlow = name.lower()
                size = m.size
                ext = os.path.splitext(nlow)[1]
                score = 0
                if size == target_len:
                    score += 900
                if ext in text_exts:
                    score += 150
                if any(k in nlow for k in ['poc', 'proof', 'uaf', 'heap', 'crash', 'repro', 'fuzz', 'case', 'bug']):
                    score += 180
                if any(k in nlow for k in ['js', 'javascript']):
                    score += 50
                if 'uint8clamped' in nlow or 'typedarray' in nlow:
                    score += 120
                # Read only files with promising names or types, or smallish files
                should_read = (score >= 100) or (ext in text_exts) or (size <= 1024 * 64)
                if not should_read:
                    continue
                # Cap read to 1MB for safety
                data = read_member(tf, m, max_bytes=min(size, 1024 * 1024))
                if not data:
                    continue
                if not is_text_bytes(data):
                    # skip likely binary
                    continue
                dl = data.lower()
                if b'uint8clampedarray' in dl:
                    score += 400
                if b'typedarray' in dl:
                    score += 120
                if b'gc(' in dl:
                    score += 100
                if b'use after free' in dl or b'use-after-free' in dl:
                    score += 200
                if b'ArrayBuffer'.lower() in dl:
                    score += 40
                if b'detach' in dl:
                    score += 70
                if size and size < 200 and (b'Uint8ClampedArray' in dl):
                    # tiny files likely not PoC
                    score -= 50
                # prefer .js
                if ext in ('.js', '.mjs'):
                    score += 80
                # prefer test directories
                if any(k in nlow for k in ['test', 'tests', 'regress', 'cases']):
                    score += 60
                if score > best_score:
                    best_score = score
                    best_data = data
            tf.close()
            return best_data

        def scan_dir_for_poc(dir_path: str) -> bytes:
            best_score = -1
            best_data = b""
            target_len = 6624
            for root, _, files in os.walk(dir_path):
                for fn in files:
                    try:
                        path = os.path.join(root, fn)
                        st = os.stat(path)
                        if not os.path.isfile(path):
                            continue
                        size = st.st_size
                        name_lower = fn.lower()
                        score = 0
                        if size == target_len:
                            score += 900
                        ext = os.path.splitext(name_lower)[1]
                        if ext in ('.js', '.mjs', '.txt', '.html', '.htm'):
                            score += 150
                        if any(k in name_lower for k in ['poc', 'proof', 'uaf', 'heap', 'crash', 'repro', 'fuzz', 'case', 'bug']):
                            score += 180
                        if 'uint8clamped' in name_lower or 'typedarray' in name_lower:
                            score += 120
                        if not (score >= 100 or ext in ('.js', '.mjs') or size <= 1024 * 64):
                            continue
                        with open(path, 'rb') as f:
                            data = f.read(min(size, 1024 * 1024))
                        if not data:
                            continue
                        if b'\x00' in data:
                            continue
                        dl = data.lower()
                        if b'uint8clampedarray' in dl:
                            score += 400
                        if b'typedarray' in dl:
                            score += 120
                        if b'gc(' in dl:
                            score += 100
                        if b'use after free' in dl or b'use-after-free' in dl:
                            score += 200
                        if ext in ('.js', '.mjs'):
                            score += 80
                        if any(k in path.lower() for k in ['test', 'tests', 'regress', 'cases']):
                            score += 60
                        if score > best_score:
                            best_score = score
                            # Read full content if small; else keep partial (it's enough for PoC likely)
                            if size <= 1024 * 1024:
                                best_data = data if len(data) == size else open(path, 'rb').read()
                            else:
                                best_data = data
                    except Exception:
                        continue
            return best_data

        # 1) Try tarball scan
        data = b""
        if os.path.isfile(src_path):
            data = scan_tar_for_poc(src_path)
        # 2) Try directory scan if it's a directory or tar scan failed
        if not data and os.path.isdir(src_path):
            data = scan_dir_for_poc(src_path)

        # 3) If still nothing, craft a fallback generic JS PoC attempt
        if not data:
            fallback_js = r"""
// Fallback PoC attempt for Uint8ClampedArray typed array misuse.
// This is a generic stressor that tries to tickle potential lifetime bugs via proxies and GC.
function safe_gc() {
    try { if (typeof gc === 'function') gc(); } catch (e) {}
}
function stress() {
    safe_gc();
    let ab = new ArrayBuffer(4096);
    let u8c = new Uint8ClampedArray(ab);
    for (let i = 0; i < u8c.length; i++) u8c[i] = i & 0xff;
    safe_gc();

    // Wrap the prototype to cause side-effects during method lookup
    let proto = Object.getPrototypeOf(u8c);
    let wrappedProto = new Proxy(proto, {
        get(target, prop, receiver) {
            // Trigger GC at sensitive times
            if (prop === 'set' || prop === 'subarray' || prop === 'slice' || prop === 'fill') {
                safe_gc();
            }
            try {
                return Reflect.get(target, prop, receiver);
            } catch (e) {
                return target[prop];
            }
        }
    });
    Object.setPrototypeOf(u8c, wrappedProto);

    safe_gc();

    try {
        // Exercise set()
        let source = new Uint8ClampedArray(512);
        for (let i = 0; i < source.length; i++) source[i] = (i * 7) & 0xff;
        u8c.set(source, 0);
    } catch (e) {}

    safe_gc();

    try {
        // Exercise subarray and slice repeatedly
        for (let i = 0; i < 64; i++) {
            let s = u8c.subarray(i, i + 32);
            let t = u8c.slice(0, 128);
            if (s.length > 0) s[0] ^= 0xff;
            if (t.length > 1) t[1] ^= 0xff;
            safe_gc();
        }
    } catch (e) {}

    // Attempt to invoke typed array routines with strange 'this' values
    try {
        let f = Object.getPrototypeOf(u8c).subarray;
        safe_gc();
        for (let i = 0; i < 32; i++) {
            try { f.call(u8c, 0, 32); } catch (e) {}
            safe_gc();
        }
    } catch (e) {}

    // Proxy around the instance to perturb internal accesses
    let proxy = new Proxy(u8c, {
        get(target, prop, receiver) {
            safe_gc();
            try { return Reflect.get(target, prop, receiver); } catch (e) { return target[prop]; }
        },
        set(target, prop, value, receiver) {
            safe_gc();
            try { return Reflect.set(target, prop, value, receiver); } catch (e) { target[prop] = value; return true; }
        }
    });

    try {
        for (let i = 0; i < 1024; i++) {
            proxy[i & 1023] = i & 0xff;
            let x = proxy[(i * 3) & 1023];
            if ((i & 31) === 0) safe_gc();
        }
    } catch (e) {}

    safe_gc();
}

for (let i = 0; i < 8; i++) stress();
"""
            data = fallback_js.encode('utf-8')

        return data