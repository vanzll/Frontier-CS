import os
from typing import Optional


class Solution:
    def solve(self, src_path: str) -> bytes:
        # 32 bytes total
        return b'.loc 1 1 0\n.file 4294967289 "x"\n'