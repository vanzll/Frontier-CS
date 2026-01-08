import os
import subprocess
import tempfile
import shutil
import stat
from typing import List, Optional


def _is_elf(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except Exception:
        return False


def _contains_libfuzzer_symbols(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            data = f.read(4096 * 4)  # read first ~16KB, good enough
        return b"LLVMFuzzerTestOneInput" in data or b"libFuzzer" in data
    except Exception:
        return False


def _find_fuzz_binaries(root: str) -> List[str]:
    fuzzers: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        # Skip some large or irrelevant directories
        base = os.path.basename(dirpath)
        if base in (".git", "build", "cmake-build-debug", "cmake-build-release"):
            continue
        for name in filenames:
            lower = name.lower()
            if "fuzz" not in lower:
                continue
            path = os.path.join(dirpath, name)
            try:
                st = os.stat(path)
                if not stat.S_ISREG(st.st_mode):
                    continue
            except Exception:
                continue
            if not _is_elf(path):
                continue
            if not _contains_libfuzzer_symbols(path):
                continue
            fuzzers.append(path)
    return fuzzers


def _try_build(src_path: str, out_dir: str, timeout: int = 900) -> bool:
    build_sh = os.path.join(src_path, "build.sh")
    if not os.path.isfile(build_sh):
        return False
    env = os.environ.copy()
    env.setdefault("OUT", out_dir)
    env.setdefault("CC", "clang")
    env.setdefault("CXX", "clang++")
    extra_flags = "-g -O1"
    env["CFLAGS"] = (env.get("CFLAGS", "") + " " + extra_flags).strip()
    env["CXXFLAGS"] = (env.get("CXXFLAGS", "") + " " + extra_flags).strip()
    env.setdefault("FUZZING_ENGINE", "libfuzzer")
    env.setdefault("SANITIZER", "address")
    env.setdefault("ARCHITECTURE", "x86_64")
    # Provide a default libfuzzer+asan engine if not set
    env.setdefault("LIB_FUZZING_ENGINE", "-fsanitize=fuzzer,address")
    try:
        subprocess.run(
            ["bash", build_sh],
            cwd=src_path,
            env=env,
            timeout=timeout,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _read_first_existing(paths: List[str]) -> Optional[bytes]:
    for p in paths:
        if os.path.isfile(p):
            try:
                with open(p, "rb") as f:
                    return f.read()
            except Exception:
                continue
    return None


def _run_libfuzzer(fuzzer_path: str, max_total_time: int) -> Optional[bytes]:
    work_dir = tempfile.mkdtemp(prefix="pocgen_")
    corpus_dir = os.path.join(work_dir, "corpus")
    artifacts_dir = os.path.join(work_dir, "artifacts")
    os.makedirs(corpus_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    # Seed corpus with empty input
    open(os.path.join(corpus_dir, "seed"), "wb").close()

    fuzzer_dir = os.path.dirname(fuzzer_path)
    before_files = set(os.listdir(fuzzer_dir))
    try:
        st_mode = os.stat(fuzzer_path).st_mode
        os.chmod(fuzzer_path, st_mode | stat.S_IXUSR)
    except Exception:
        pass

    cmd = [
        fuzzer_path,
        "-max_total_time=%d" % max_total_time,
        "-timeout=10",
        "-rss_limit_mb=0",
        "-artifact_prefix=%s/" % artifacts_dir,
        corpus_dir,
    ]

    try:
        subprocess.run(
            cmd,
            cwd=fuzzer_dir,
            timeout=max_total_time + 20,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass

    # 1) Check artifacts_dir for crash files
    candidates: List[str] = []
    try:
        for name in os.listdir(artifacts_dir):
            path = os.path.join(artifacts_dir, name)
            if os.path.isfile(path):
                candidates.append(path)
    except Exception:
        pass

    if candidates:
        candidates.sort()
        data = _read_first_existing(candidates)
        shutil.rmtree(work_dir, ignore_errors=True)
        return data

    # 2) Check fuzzer_dir for new crash-like files
    try:
        after_files = set(os.listdir(fuzzer_dir))
        new_files = after_files - before_files
        crash_like: List[str] = []
        for name in new_files:
            if name.startswith(("crash-", "oom-", "timeout-", "leak-")):
                path = os.path.join(fuzzer_dir, name)
                if os.path.isfile(path):
                    crash_like.append(path)
        if crash_like:
            crash_like.sort()
            data = _read_first_existing(crash_like)
            shutil.rmtree(work_dir, ignore_errors=True)
            return data
    except Exception:
        pass

    shutil.rmtree(work_dir, ignore_errors=True)
    return None


class Solution:
    def solve(self, src_path: str) -> bytes:
        cache_path = os.path.join(src_path, ".poc_cache")
        if os.path.isfile(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    return f.read()
            except Exception:
                pass

        # 1) Look for prebuilt fuzzers
        fuzzers = _find_fuzz_binaries(src_path)

        # 2) If none, attempt to build, then search again (prefer OUT dir)
        if not fuzzers:
            out_dir = os.path.join(src_path, "out")
            os.makedirs(out_dir, exist_ok=True)
            if _try_build(src_path, out_dir):
                fuzzers = _find_fuzz_binaries(out_dir)
                if not fuzzers:
                    fuzzers = _find_fuzz_binaries(src_path)

        # 3) If still none, return a simple fallback input
        if not fuzzers:
            # Fallback: short neutral input
            return b"A" * 16

        # 4) Run libFuzzer-driven fuzzing on discovered fuzzers
        total_fuzz_time = 120  # seconds budget
        per_fuzzer_time = max(20, total_fuzz_time // max(1, len(fuzzers)))

        for fpath in fuzzers:
            poc = _run_libfuzzer(fpath, max_total_time=per_fuzzer_time)
            if poc:
                try:
                    with open(cache_path, "wb") as f:
                        f.write(poc)
                except Exception:
                    pass
                return poc

        # 5) If no crash found, return a fallback deterministic input
        return b"A" * 16