import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        sdp_lines = [
            "v=0",
            "o=- 0 0 IN IP4 127.0.0.1",
            "s=-",
            "c=IN IP4 127.0.0.1",
            "t=0 0",
            "m=audio 9 RTP/AVP 96",
            "a=mid:0",
            "a=rtpmap:96 opus/48000/2",
            "a=fmtp:96 minptime=10;useinbandfec=1;usedtx=1;x-google-start-bitrate=16000;maxaveragebitrate=510000;dtx=1;stereo=1;sprop-maxcapturerate=48000;sprop-stereo=1;sprop-parameter-sets="
        ]
        sdp_body = "\r\n".join(sdp_lines)
        content_length = len(sdp_body.encode("ascii", errors="ignore"))

        sip_headers = [
            "INVITE sip:bob@example.com SIP/2.0",
            "Via: SIP/2.0/UDP 127.0.0.1:5060;branch=z9hG4bK776asdhds",
            "Max-Forwards: 70",
            "From: \"Alice\" <sip:alice@example.com>;tag=1928301774",
            "To: <sip:bob@example.com>",
            "Call-ID: a84b4c76e66710@localhost",
            "CSeq: 314159 INVITE",
            "Contact: <sip:alice@127.0.0.1:5060>",
            "Content-Type: application/sdp",
            f"Content-Length: {content_length}"
        ]
        sip_message = "\r\n".join(sip_headers) + "\r\n\r\n" + sdp_body
        return sip_message.encode("ascii", errors="ignore")