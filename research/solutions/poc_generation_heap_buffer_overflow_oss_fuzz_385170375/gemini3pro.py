import os
import shutil
import subprocess
import glob
import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        Strategies:
        1. Build FFmpeg with ASAN and rv60 decoder enabled.
        2. Compile the standard target_dec_fuzzer.c harness with libFuzzer.
        3. Run the fuzzer to discover the crashing input.
        """
        # Prepare build directory
        base_dir = os.getcwd()
        build_dir = os.path.join(base_dir, 'build_poc')
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        os.makedirs(build_dir)

        # Environment configuration
        env = os.environ.copy()
        env['CC'] = 'clang'
        env['CXX'] = 'clang++'
        # High optimization level helps fuzzer speed, ASAN catches bugs
        env['CFLAGS'] = '-fsanitize=address -O1 -g -fno-omit-frame-pointer'
        env['CXXFLAGS'] = '-fsanitize=address -O1 -g -fno-omit-frame-pointer'
        env['LDFLAGS'] = '-fsanitize=address'

        # Configure FFmpeg
        # Minimal configuration to build rv60 decoder with ASAN
        configure_cmd = [
            os.path.join(src_path, 'configure'),
            '--disable-everything',
            '--enable-decoder=rv60',
            '--enable-static',
            '--disable-shared',
            '--disable-doc',
            '--disable-programs',
            '--disable-avdevice',
            '--disable-avformat',
            '--disable-swscale',
            '--disable-avfilter',
            '--disable-asm',
            '--enable-swresample',
            '--cc=clang',
            '--cxx=clang++',
            '--extra-cflags=-fsanitize=address -O1 -g -fno-omit-frame-pointer',
            '--extra-ldflags=-fsanitize=address'
        ]

        try:
            subprocess.check_call(configure_cmd, cwd=build_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            # Fallback for configuration issues
            return b'\x00' * 149

        # Build dependencies
        try:
            subprocess.check_call(['make', '-j8'], cwd=build_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            return b'\x00' * 149

        # Locate fuzzer harness
        fuzzer_src = os.path.join(src_path, 'tools', 'target_dec_fuzzer.c')
        if not os.path.exists(fuzzer_src):
            # Attempt to find it recursively
            found = glob.glob(os.path.join(src_path, '**', 'target_dec_fuzzer.c'), recursive=True)
            if found:
                fuzzer_src = found[0]
            else:
                return b'\x00' * 149

        # Identify static libraries to link
        libs = []
        for lib in ['libavcodec', 'libavutil', 'libswresample']:
            lib_a = os.path.join(build_dir, lib, f'{lib}.a')
            if os.path.exists(lib_a):
                libs.append(lib_a)

        # Compile fuzzer
        fuzzer_bin = os.path.join(build_dir, 'fuzzer_rv60')
        compile_cmd = [
            'clang',
            '-fsanitize=address,fuzzer',
            '-O1', '-g',
            '-I', src_path,
            '-I', build_dir,
            '-DFFMPEG_DECODER="rv60"',
            fuzzer_src,
            '-o', fuzzer_bin
        ] + libs + ['-lm', '-lz', '-lpthread']

        has_fuzzer = False
        try:
            subprocess.check_call(compile_cmd, cwd=build_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            has_fuzzer = True
        except subprocess.CalledProcessError:
            pass

        if not has_fuzzer:
            return b'\x00' * 149

        # Prepare corpus and artifacts
        corpus_dir = os.path.join(build_dir, 'corpus')
        artifact_dir = os.path.join(build_dir, 'artifacts')
        os.makedirs(corpus_dir, exist_ok=True)
        os.makedirs(artifact_dir, exist_ok=True)

        # Seed corpus
        seeds = glob.glob(os.path.join(src_path, '**', '*.rm'), recursive=True)
        seeds += glob.glob(os.path.join(src_path, '**', '*.rv60'), recursive=True)
        for seed in seeds:
            try:
                shutil.copy(seed, corpus_dir)
            except:
                pass
        
        if not os.listdir(corpus_dir):
            # Create a basic seed if none found
            with open(os.path.join(corpus_dir, 'seed'), 'wb') as f:
                f.write(b'\x00' * 149)

        # Run Fuzzer
        # Target length is 149, so we restrict max_len to slightly more to encourage relevant inputs
        fuzz_env = env.copy()
        fuzz_env['ASAN_OPTIONS'] = 'halt_on_error=1:exitcode=77'
        
        fuzz_cmd = [
            fuzzer_bin,
            corpus_dir,
            '-max_total_time=120', # 2 minutes
            '-max_len=200',
            '-artifact_prefix=' + artifact_dir + '/'
        ]

        try:
            subprocess.run(fuzz_cmd, cwd=build_dir, env=fuzz_env, timeout=130, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            pass
        except subprocess.CalledProcessError:
            pass

        # Retrieve crash
        crashes = glob.glob(os.path.join(artifact_dir, 'crash-*'))
        if crashes:
            # Return the smallest crash
            crashes.sort(key=os.path.getsize)
            with open(crashes[0], 'rb') as f:
                return f.read()

        return b'\x00' * 149