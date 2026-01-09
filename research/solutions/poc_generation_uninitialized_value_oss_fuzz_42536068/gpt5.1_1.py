import io

class Solution:
    def solve(self, src_path: str) -> bytes:
        values = [
            "",
            " ",
            "+",
            "-",
            "nan",
            "NaN",
            "inf",
            "Infinity",
            "1e9999",
            "0x",
            "0xG",
            "1e-",
            "18446744073709551616",
            "-9223372036854775809",
            "9223372036854775808",
            "999999999999999999999999999999999999",
            "0.0.0",
            "--123",
            "++123",
            "123abc",
            "abc123",
            "0xFFFFFFFFFFFFFFFFF",
        ]

        out = io.StringIO()
        out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        out.write('<root>\n')
        for i, v in enumerate(values):
            v_esc = (
                v.replace("&", "&amp;")
                 .replace('"', "&quot;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;")
            )
            out.write(
                f'  <element id="{i}" '
                f'int="{v_esc}" '
                f'unsigned="{v_esc}" '
                f'int64="{v_esc}" '
                f'uint64="{v_esc}" '
                f'float="{v_esc}" '
                f'double="{v_esc}" '
                f'bool="{v_esc}" />\n'
            )
        out.write('</root>\n')

        return out.getvalue().encode("utf-8")