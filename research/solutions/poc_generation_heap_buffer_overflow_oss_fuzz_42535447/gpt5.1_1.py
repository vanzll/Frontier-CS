import os
import tarfile
import tempfile
import subprocess
import random
import time


class Solution:
    def _extract_tarball(self, src_path: str) -> str:
        tmpdir = tempfile.mkdtemp(prefix="poc_gen_")
        with tarfile.open(src_path, "r:*") as tar:
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            for member in tar.getmembers():
                member_path = os.path.join(tmpdir, member.name)
                if not is_within_directory(tmpdir, member_path):
                    continue
            tar.extractall(tmpdir)
        entries = [os.path.join(tmpdir, e) for e in os.listdir(tmpdir)]
        dirs = [p for p in entries if os.path.isdir(p)]
        if len(dirs) == 1:
            return dirs[0]
        return tmpdir

    def _find_files(self, root: str):
        harness_file = None
        gm_file = None
        any_fuzzer = None
        for dirpath, _, files in os.walk(root):
            for name in files:
                ext = os.path.splitext(name)[1]
                if ext not in (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh"):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except Exception:
                    continue
                if "decodeGainmapMetadata" in text and ext in (".c", ".cc", ".cpp", ".cxx"):
                    if gm_file is None:
                        gm_file = path
                if "LLVMFuzzerTestOneInput" in text:
                    if any_fuzzer is None:
                        any_fuzzer = path
                    if "decodeGainmapMetadata" in text:
                        if harness_file is None:
                            harness_file = path
        if harness_file is None:
            harness_file = any_fuzzer
        return harness_file, gm_file

    def _write_main(self, root: str) -> str:
        main_path = os.path.join(root, "standalone_fuzz_main.cc")
        src = r"""
#include <cstdint>
#include <cstdio>
#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

int main(int argc, char** argv) {
  if (argc < 2) {
    return 1;
  }
  const char* path = argv[1];
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    return 1;
  }
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());
  if (data.empty()) {
    return 0;
  }
  LLVMFuzzerTestOneInput(reinterpret_cast<const uint8_t*>(data.data()),
                         data.size());
  return 0;
}
"""
        with open(main_path, "w", encoding="utf-8") as f:
            f.write(src)
        return main_path

    def _collect_include_dirs(self, root: str, harness_file: str, gm_file: str | None):
        inc_dirs = set()
        inc_dirs.add(root)
        if harness_file:
            inc_dirs.add(os.path.dirname(harness_file))
        if gm_file:
            inc_dirs.add(os.path.dirname(gm_file))
        for dirpath, dirs, _ in os.walk(root):
            base = os.path.basename(dirpath).lower()
            if base in ("include", "includes", "inc"):
                inc_dirs.add(dirpath)
        return sorted(inc_dirs)

    def _compile_with_cmd(self, compiler: str, cmd: list[str]) -> str | None:
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            return None
        if res.returncode == 0:
            return cmd[-1]
        return None

    def _compile_harness(self, root: str, harness_file: str, gm_file: str | None) -> str | None:
        if harness_file is None:
            return None
        main_path = self._write_main(root)
        compilers = ["clang++", "g++"]
        for compiler in compilers:
            # minimal sources
            sources = [main_path, harness_file]
            if gm_file and gm_file != harness_file:
                sources.append(gm_file)
            inc_dirs = self._collect_include_dirs(root, harness_file, gm_file)
            out_bin = os.path.join(root, "poc_fuzz_bin")
            cmd = [compiler, "-std=c++17", "-g", "-O1",
                   "-fsanitize=address", "-fno-omit-frame-pointer"]
            for d in inc_dirs:
                cmd.extend(["-I", d])
            cmd.extend(sources)
            cmd.extend(["-o", out_bin])
            bin_path = self._compile_with_cmd(compiler, cmd)
            if bin_path is not None:
                return bin_path

            # fallback: compile many sources
            source_files = [main_path, harness_file]
            seen = set(source_files)
            for dirpath, _, files in os.walk(root):
                for name in files:
                    ext = os.path.splitext(name)[1]
                    if ext not in (".c", ".cc", ".cpp", ".cxx"):
                        continue
                    path = os.path.join(dirpath, name)
                    if path in seen:
                        continue
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            snippet = f.read(4096)
                    except Exception:
                        continue
                    if "LLVMFuzzerTestOneInput" in snippet:
                        continue
                    if " main(" in snippet or "main(" in snippet or " WinMain(" in snippet:
                        continue
                    source_files.append(path)
                    seen.add(path)
            cmd = [compiler, "-std=c++17", "-g", "-O1",
                   "-fsanitize=address", "-fno-omit-frame-pointer"]
            for d in inc_dirs:
                cmd.extend(["-I", d])
            cmd.extend(source_files)
            cmd.extend(["-o", out_bin])
            bin_path = self._compile_with_cmd(compiler, cmd)
            if bin_path is not None:
                return bin_path
        return None

    def _run_fuzz(self, bin_path: str, workdir: str) -> bytes | None:
        max_iters = 4000
        time_limit = 20.0
        start = time.time()
        input_path = os.path.join(workdir, "poc_input.bin")
        for _ in range(max_iters):
            if time.time() - start > time_limit:
                break
            length = random.randint(1, 256)
            data = os.urandom(length)
            try:
                with open(input_path, "wb") as f:
                    f.write(data)
            except Exception:
                continue
            try:
                res = subprocess.run(
                    [bin_path, input_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except Exception:
                continue
            if res.returncode != 0:
                stderr = res.stderr
                if (b"AddressSanitizer" in stderr or
                        b"heap-buffer-overflow" in stderr or
                        b"buffer-overflow" in stderr):
                    return data
        return None

    def solve(self, src_path: str) -> bytes:
        root = self._extract_tarball(src_path)
        harness_file, gm_file = self._find_files(root)
        bin_path = self._compile_harness(root, harness_file, gm_file)
        if bin_path:
            poc = self._run_fuzz(bin_path, root)
            if poc is not None:
                return poc
        return b"A" * 133