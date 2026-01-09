import os
import tarfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        header = b"media100_to_mjpegb\n"
        total_len = 1025
        payload_len = max(0, total_len - len(header))
        payload = b"\x00" * payload_len
        return header + payload