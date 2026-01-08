import os
import sys
import subprocess
import tempfile
import tarfile
import glob
import shutil

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability by compiling the target
        with AddressSanitizer and running a fuzzer.
        """
        # Create a temporary directory for the build and execution
        with tempfile.TemporaryDirectory() as work_dir:
            # Extract the source code
            try:
                with tarfile.open(src_path) as tar:
                    tar.extractall(path=work_dir)
            except Exception:
                return b""

            # Find source root (folder containing CMakeLists.txt)
            source_root = work_dir
            for root, dirs, files in os.walk(work_dir):
                if "CMakeLists.txt" in files:
                    source_root = root
                    break

            # Identify the specific fuzzer source file
            fuzzer_src = None
            fuzzer_name = None
            candidates = []
            
            # Recursively find files containing LLVMFuzzerTestOneInput
            for root, dirs, files in os.walk(source_root):
                for f in files:
                    if f.endswith((".c", ".cc", ".cpp")):
                        path = os.path.join(root, f)
                        try:
                            with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                                content = fp.read()
                                if "LLVMFuzzerTestOneInput" in content:
                                    candidates.append((f, path))
                        except IOError:
                            pass
            
            # Prioritize fuzzer related to the vulnerability (polygonToCells/polyfill)
            for name, path in candidates:
                lower_name = name.lower()
                if "polygon" in lower_name or "polyfill" in lower_name:
                    fuzzer_name = name
                    fuzzer_src = path
                    break
            
            # Fallback to the first available fuzzer if specific one not found
            if not fuzzer_src and candidates:
                fuzzer_name = candidates[0][0]
                fuzzer_src = candidates[0][1]

            if not fuzzer_src:
                return b""

            # Build with CMake
            build_dir = os.path.join(source_root, "build_fuzz")
            os.makedirs(build_dir, exist_ok=True)
            
            env = os.environ.copy()
            # Enforce clang and AddressSanitizer + Fuzzer
            env["CC"] = "clang"
            env["CXX"] = "clang++"
            flags = "-fsanitize=address,fuzzer -g -O1"
            env["CFLAGS"] = flags
            env["CXXFLAGS"] = flags
            
            # Configure CMake
            # Enable fuzzers and disable unnecessary targets
            subprocess.run(
                ["cmake", "-DBUILD_FUZZERS=ON", "-DBUILD_TESTING=OFF", "-DENABLE_DOCS=OFF", ".."],
                cwd=build_dir, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            
            # Build
            subprocess.run(
                ["make", "-j8"],
                cwd=build_dir, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            
            # Locate the compiled fuzzer executable
            fuzzer_exe = None
            stem = os.path.splitext(fuzzer_name)[0]
            
            # Search for the executable matching the fuzzer name
            for root, dirs, files in os.walk(build_dir):
                for f in files:
                    path = os.path.join(root, f)
                    if os.access(path, os.X_OK) and not os.path.isdir(path):
                        if stem in f:
                            fuzzer_exe = path
                            break
                if fuzzer_exe: break
            
            # Fallback search for any executable with "fuzz" in name
            if not fuzzer_exe:
                for root, dirs, files in os.walk(build_dir):
                    for f in files:
                        path = os.path.join(root, f)
                        if os.access(path, os.X_OK) and "fuzz" in f and "." not in f:
                            fuzzer_exe = path
                            break
                    if fuzzer_exe: break
            
            # Fallback: Manual compilation if CMake failed to output binary
            if not fuzzer_exe:
                src_files = []
                include_dirs = []
                lib_dir = os.path.join(source_root, "src/h3lib/lib")
                inc_dir = os.path.join(source_root, "src/h3lib/include")
                
                if os.path.exists(lib_dir) and os.path.exists(inc_dir):
                    for f in os.listdir(lib_dir):
                        if f.endswith(".c"):
                            src_files.append(os.path.join(lib_dir, f))
                    
                    out_bin = os.path.join(work_dir, "manual_fuzzer")
                    compile_cmd = [
                        "clang", "-fsanitize=address,fuzzer", "-O1", 
                        "-I", inc_dir,
                        fuzzer_src
                    ] + src_files + ["-o", out_bin]
                    
                    if subprocess.run(compile_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                        fuzzer_exe = out_bin

            if not fuzzer_exe:
                return b""

            # Run the fuzzer to find the crash
            # Limit time to 60 seconds (ground truth is relatively short, should trigger quickly)
            try:
                subprocess.run(
                    [fuzzer_exe, "-max_total_time=60", "-detect_leaks=0", "-rss_limit_mb=2048"],
                    cwd=work_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=70
                )
            except subprocess.TimeoutExpired:
                pass

            # Collect crash artifacts
            crashes = glob.glob(os.path.join(work_dir, "crash-*"))
            if crashes:
                # Return the smallest crash file to maximize score
                crashes.sort(key=os.path.getsize)
                with open(crashes[0], "rb") as f:
                    return f.read()

            return b""