import sys
import os
import subprocess
import tempfile
import time
import random
import glob
import shutil

class Solution:
    def solve(self, src_path: str) -> bytes:
        base_tmp = tempfile.mkdtemp()
        try:
            # Extract source
            subprocess.check_call(['tar', 'xf', src_path, '-C', base_tmp], 
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Locate root directory containing CMakeLists.txt
            root_dir = base_tmp
            for root, dirs, files in os.walk(base_tmp):
                if 'CMakeLists.txt' in files and 'src' in dirs:
                    root_dir = root
                    break
            
            # Attempt to configure with CMake to generate configuration headers (h3api.h, etc.)
            try:
                subprocess.run(['cmake', '.'], cwd=root_dir, 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

            # Locate the specific fuzz harness
            fuzz_source = None
            # Prioritize files explicitly mentioning polygonToCellsExperimental
            for root, dirs, files in os.walk(root_dir):
                for f in files:
                    if f.endswith('.c') or f.endswith('.cc') or f.endswith('.cpp'):
                        path = os.path.join(root, f)
                        try:
                            with open(path, 'r', errors='ignore') as fd:
                                content = fd.read()
                                if 'LLVMFuzzerTestOneInput' in content:
                                    if 'polygonToCellsExperimental' in content:
                                        fuzz_source = path
                                        break
                                    if 'polygonToCells' in content and not fuzz_source:
                                        fuzz_source = path
                        except: pass
                if fuzz_source and 'polygonToCellsExperimental' in open(fuzz_source, errors='ignore').read():
                    break
            
            if not fuzz_source:
                # If no specific harness found, we can't reliably reproduce
                return b''

            # Locate Library Source Files
            lib_src = []
            include_dirs = set()
            
            # Find include directories
            for root, dirs, files in os.walk(root_dir):
                if any(f.endswith('.h') for f in files):
                    include_dirs.add(root)

            # Find source files for the library
            for root, dirs, files in os.walk(root_dir):
                for f in files:
                    if f.endswith('.c'):
                        full_path = os.path.join(root, f)
                        # Exclude tests, apps, fuzzers, and examples
                        if any(x in full_path for x in ['/test/', '/apps/', '/fuzzers/', '/examples/', 'main.c']):
                            continue
                        
                        # Double check for main function to avoid linker errors
                        try:
                            with open(full_path, 'r', errors='ignore') as fd:
                                if 'int main(' in fd.read():
                                    continue
                        except: continue
                        
                        lib_src.append(full_path)

            # Prepare for Compilation
            fuzzer_bin = os.path.join(base_tmp, 'fuzzer_bin')
            # Essential flags for reproduction
            cflags = ['-g', '-O2', '-D_GNU_SOURCE', '-fsanitize=address']
            libs = ['-lm']
            
            built_with_libfuzzer = False
            
            # Attempt 1: Compile with -fsanitize=fuzzer (libFuzzer)
            cmd = ['clang'] + cflags + ['-fsanitize=fuzzer'] 
            for inc in include_dirs:
                cmd.extend(['-I', inc])
            cmd.append(fuzz_source)
            cmd.extend(lib_src)
            cmd.extend(['-o', fuzzer_bin])
            cmd.extend(libs)
            
            try:
                subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                built_with_libfuzzer = True
            except subprocess.CalledProcessError:
                # Attempt 2: Fallback to manual driver if libFuzzer linking fails
                driver_path = os.path.join(base_tmp, 'driver.c')
                with open(driver_path, 'w') as f:
                    f.write(r'''
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size);
int main(int argc, char **argv) {
    if (argc < 2) return 0;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = (uint8_t*)malloc(sz);
    fread(data, 1, sz, f);
    fclose(f);
    LLVMFuzzerTestOneInput(data, sz);
    free(data);
    return 0;
}
''')
                cmd = ['clang'] + cflags 
                for inc in include_dirs:
                    cmd.extend(['-I', inc])
                cmd.append(driver_path)
                cmd.append(fuzz_source)
                cmd.extend(lib_src)
                cmd.extend(['-o', fuzzer_bin])
                cmd.extend(libs)
                try:
                    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except subprocess.CalledProcessError:
                    return b''

            # Run Fuzzing to find the PoC
            env = os.environ.copy()
            # Set ASAN options to exit with specific code on error
            env['ASAN_OPTIONS'] = 'halt_on_error=1:exitcode=77'
            
            if built_with_libfuzzer:
                # Run libFuzzer
                cmd = [
                    fuzzer_bin,
                    '-artifact_prefix=' + base_tmp + '/',
                    '-max_total_time=60',  # 60 seconds budget
                    '-print_final_stats=1'
                ]
                corpus_dir = os.path.join(base_tmp, 'corpus')
                os.makedirs(corpus_dir, exist_ok=True)
                cmd.append(corpus_dir)
                
                subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Check for crash artifacts
                for f in os.listdir(base_tmp):
                    if f.startswith('crash-') or f.startswith('leak-'):
                        with open(os.path.join(base_tmp, f), 'rb') as fd:
                            return fd.read()
            else:
                # Manual fuzzing loop
                start_t = time.time()
                while time.time() - start_t < 60:
                    # Generate random input
                    # Ground truth is 1032, cover range around it
                    l = random.randint(100, 1500)
                    data = os.urandom(l)
                    
                    inp_path = os.path.join(base_tmp, 'input.bin')
                    with open(inp_path, 'wb') as f:
                        f.write(data)
                    
                    proc = subprocess.run([fuzzer_bin, inp_path], env=env, 
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    # Check for crash
                    if proc.returncode == 77 or b'AddressSanitizer' in proc.stderr:
                        return data

            return b''

        finally:
            shutil.rmtree(base_tmp, ignore_errors=True)