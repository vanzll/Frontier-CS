import glob
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        Method:
        1. Extract the source code.
        2. Locate the fuzz harness (containing LLVMFuzzerTestOneInput) and other source files.
        3. Compile the fuzzer using clang++ with MemorySanitizer (MSAN) since the vulnerability is an "Uninitialized Value".
        4. Run the fuzzer for a short duration to find a crashing input.
        5. Minimize the crashing input to improve the score.
        6. Return the crash data.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Extract source
            try:
                with tarfile.open(src_path) as tar:
                    tar.extractall(path=temp_dir)
            except Exception:
                # If extraction fails, we can't proceed
                return b""

            # 2. Locate source files
            sources = []
            harness_file = None
            include_dirs = set()

            for root, dirs, files in os.walk(temp_dir):
                for f in files:
                    file_path = os.path.join(root, f)
                    
                    if f.endswith(('.h', '.hpp')):
                        include_dirs.add(root)
                    
                    elif f.endswith(('.cpp', '.cc', '.cxx', '.c')):
                        # Identify if this is the harness or a source file
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as fp:
                                content = fp.read()
                                if "LLVMFuzzerTestOneInput" in content:
                                    harness_file = file_path
                                elif "int main" in content:
                                    # Skip files with main() that are not the harness (e.g. unit tests)
                                    continue
                                else:
                                    sources.append(file_path)
                        except IOError:
                            pass
            
            if not harness_file:
                # Without a harness, we cannot fuzz
                return b""

            include_dirs.add(os.path.dirname(harness_file))
            fuzzer_bin = os.path.join(temp_dir, "fuzzer_bin")

            # 3. Build the fuzzer
            # Try MSAN first (most appropriate for Uninitialized Value)
            compile_cmd = [
                "clang++",
                "-g", "-O1",
                "-fsanitize=memory",
                "-fsanitize-memory-track-origins",
                "-fsanitize=fuzzer",
                "-o", fuzzer_bin,
                harness_file
            ] + sources + [f"-I{d}" for d in include_dirs]

            build_success = False
            try:
                subprocess.run(compile_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                build_success = True
            except subprocess.CalledProcessError:
                # Fallback to ASAN if MSAN fails to build (e.g., environment issues)
                try:
                    compile_cmd[3] = "-fsanitize=address"
                    # Remove the track-origins flag which is specific to MSAN
                    compile_cmd = [arg for arg in compile_cmd if "track-origins" not in arg]
                    subprocess.run(compile_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    build_success = True
                except subprocess.CalledProcessError:
                    pass
            
            if not build_success:
                return b""

            # 4. Fuzz
            corpus_dir = os.path.join(temp_dir, "corpus")
            artifacts_dir = os.path.join(temp_dir, "artifacts")
            os.makedirs(corpus_dir, exist_ok=True)
            os.makedirs(artifacts_dir, exist_ok=True)

            # Create a simple XML seed
            with open(os.path.join(corpus_dir, "seed.xml"), "w") as f:
                f.write("<root><e a='1'/></root>")

            fuzz_cmd = [
                fuzzer_bin,
                corpus_dir,
                "-max_total_time=45",  # Run for 45 seconds
                "-artifact_prefix=" + artifacts_dir + "/",
            ]

            try:
                subprocess.run(fuzz_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

            # 5. Check for crashes and minimize
            crashes = glob.glob(os.path.join(artifacts_dir, "crash-*"))
            crashes += glob.glob(os.path.join(artifacts_dir, "timeout-*"))

            if not crashes:
                return b""

            # Pick the smallest crash initially
            crashes.sort(key=os.path.getsize)
            best_crash = crashes[0]

            # Minimize the crash to get a higher score
            minimized_path = best_crash + ".min"
            minimize_cmd = [
                fuzzer_bin,
                "-minimize_crash=1",
                "-max_total_time=15",
                "-exact_artifact_path=" + minimized_path,
                best_crash
            ]

            try:
                subprocess.run(minimize_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(minimized_path):
                    with open(minimized_path, "rb") as f:
                        return f.read()
            except Exception:
                pass

            # If minimization failed, return the original crash
            with open(best_crash, "rb") as f:
                return f.read()