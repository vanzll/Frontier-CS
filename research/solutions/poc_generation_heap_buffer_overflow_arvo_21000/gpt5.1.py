import os
import tarfile
import tempfile
import subprocess
import shutil
import stat
import random
import time
import re
import shlex
from pathlib import Path


class Solution:
    def solve(self, src_path: str) -> bytes:
        work_dir = tempfile.mkdtemp(prefix="poc_work_")
        try:
            self._extract_tar(src_path, work_dir)
            binary, interface = self._build_and_locate_harness(work_dir)
            if binary is not None and interface is not None:
                poc = self._find_crashing_input(binary, interface)
                if poc is not None:
                    return poc
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
        # Fallback: fixed-length PoC if everything else fails
        return b"A" * 33

    def _extract_tar(self, src_path: str, out_dir: str) -> None:
        with tarfile.open(src_path, "r:*") as tar:
            tar.extractall(out_dir)

    def _build_and_locate_harness(self, work_dir: str):
        build_script = None
        for root, dirs, files in os.walk(work_dir):
            if "build.sh" in files:
                build_script = os.path.join(root, "build.sh")
                break

        if build_script is not None:
            try:
                env = os.environ.copy()
                cc = shutil.which("clang") or shutil.which("gcc")
                cxx = shutil.which("clang++") or shutil.which("g++")
                if cc:
                    env.setdefault("CC", cc)
                if cxx:
                    env.setdefault("CXX", cxx)
                cmd = f"cd {shlex.quote(os.path.dirname(build_script))} && chmod +x build.sh && ./build.sh"
                subprocess.run(
                    ["bash", "-lc", cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=120,
                    env=env,
                    check=False,
                )
            except Exception:
                pass

        exec_paths = []
        for root, dirs, files in os.walk(work_dir):
            for f in files:
                path = os.path.join(root, f)
                try:
                    if os.path.isfile(path) and not f.endswith(".sh"):
                        st = os.stat(path)
                        if st.st_mode & stat.S_IXUSR and os.access(path, os.X_OK):
                            exec_paths.append(path)
                except Exception:
                    continue

        if not exec_paths:
            return None, None

        def exec_rank(path: str):
            name = os.path.basename(path).lower()
            score = 0
            keywords = ["capwap", "ndpi", "poc", "target", "fuzz", "test"]
            for i, kw in enumerate(keywords):
                if kw in name:
                    score += 10 - i
            return (-score, len(name))

        exec_paths.sort(key=exec_rank)
        binary = exec_paths[0]
        interface = self._infer_interface_from_sources(work_dir, binary)
        return binary, interface

    def _infer_interface_from_sources(self, work_dir: str, binary: str) -> str:
        bin_stem = os.path.splitext(os.path.basename(binary))[0]
        main_sources = []
        for root, dirs, files in os.walk(work_dir):
            for f in files:
                if f.endswith(".c") or f.endswith(".cc") or f.endswith(".cpp"):
                    path = os.path.join(root, f)
                    try:
                        with open(path, "r", errors="ignore") as fp:
                            code = fp.read()
                    except Exception:
                        continue
                    if "main(" in code:
                        main_sources.append((path, code))

        if not main_sources:
            return "stdin"

        def source_rank(item):
            path, code = item
            name = os.path.splitext(os.path.basename(path))[0]
            score = 0
            low = code.lower()
            if "ndpi_search_setup_capwap" in code:
                score += 4
            if "capwap" in low:
                score += 2
            if name == bin_stem:
                score += 3
            return -score

        main_sources.sort(key=source_rank)
        chosen_path, chosen_code = main_sources[0]
        code = chosen_code

        if "stdin" in code or "read(0" in code or "read (0" in code:
            return "stdin"
        if re.search(r"argv\s*\[\s*1\s*\]", code):
            return "file-arg"
        return "stdin"

    def _run_candidate(self, binary: str, interface: str, data: bytes, timeout: float = 0.2):
        try:
            if interface == "file-arg":
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    f.write(data)
                    f.flush()
                    fname = f.name
                try:
                    res = subprocess.run(
                        [binary, fname],
                        input=b"",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=timeout,
                        check=False,
                    )
                finally:
                    try:
                        os.unlink(fname)
                    except Exception:
                        pass
            else:
                res = subprocess.run(
                    [binary],
                    input=data,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    check=False,
                )
        except subprocess.TimeoutExpired:
            return False, b"", False
        except Exception:
            return False, b"", False

        out = res.stdout + res.stderr
        crashed = (
            res.returncode != 0
            and (b"AddressSanitizer" in out or b"heap-buffer-overflow" in out or b"buffer-overflow" in out)
        )
        seen_capwap = b"capwap" in out.lower()
        return crashed, out, seen_capwap

    def _generate_initial_candidate(self, i: int) -> bytes:
        max_len = 64
        if i < 256:
            length = 33
        else:
            if random.random() < 0.7:
                length = 33
            else:
                length = random.randint(1, max_len)
        if i < 16:
            interesting = [
                0x00,
                0xFF,
                0x20,
                0x08,
                0x80,
                0x01,
                0x10,
                0x7F,
                0x40,
                0xF0,
                0x55,
                0xAA,
                0x33,
                0xCC,
                0x99,
                0x77,
            ]
            b = interesting[i]
            return bytes([b]) * length
        else:
            try:
                buf = bytearray(os.urandom(length))
            except Exception:
                buf = bytearray(random.getrandbits(8) for _ in range(length))
            if length >= 8:
                patterns = [
                    (0x20, 0x00),
                    (0x20, 0x80),
                    (0x00, 0x20),
                    (0x80, 0x20),
                ]
                p = random.choice(patterns)
                buf[0] = p[0]
                buf[1] = p[1]
            return bytes(buf)

    def _mutate(self, seed: bytes) -> bytes:
        if not seed:
            return b"\x00"
        ba = bytearray(seed)
        num_mut = random.randint(1, min(4, max(1, len(ba) // 4 + 1)))
        interesting_bytes = [
            0x00,
            0x01,
            0x02,
            0x03,
            0x07,
            0x08,
            0x0F,
            0x10,
            0x1F,
            0x20,
            0x3F,
            0x40,
            0x7F,
            0x80,
            0xFF,
        ]
        for _ in range(num_mut):
            pos = random.randrange(len(ba))
            ba[pos] = random.choice(interesting_bytes)
        # Occasionally adjust length
        if random.random() < 0.3 and len(ba) > 1:
            # truncate
            new_len = random.randint(1, len(ba))
            ba = ba[:new_len]
        elif random.random() < 0.3 and len(ba) < 64:
            # extend
            extra = random.randint(1, 4)
            for _ in range(extra):
                ba.append(random.choice(interesting_bytes))
        return bytes(ba)

    def _find_crashing_input(self, binary: str, interface: str) -> bytes:
        random.seed(0xC0FFEE)
        max_iters = 2000
        start_time = time.time()
        time_budget = 20.0  # seconds
        best_capwap_input = None

        for i in range(max_iters):
            if time.time() - start_time > time_budget:
                break

            if best_capwap_input is None:
                data = self._generate_initial_candidate(i)
            else:
                if random.random() < 0.7:
                    data = self._mutate(best_capwap_input)
                else:
                    data = self._generate_initial_candidate(i)

            crashed, out, seen_capwap = self._run_candidate(binary, interface, data)
            if crashed:
                return data
            if seen_capwap and best_capwap_input is None:
                best_capwap_input = data

        if best_capwap_input is not None:
            for _ in range(2000):
                if time.time() - start_time > time_budget:
                    break
                data = self._mutate(best_capwap_input)
                crashed, out, seen_capwap = self._run_candidate(binary, interface, data)
                if crashed:
                    return data

        return None