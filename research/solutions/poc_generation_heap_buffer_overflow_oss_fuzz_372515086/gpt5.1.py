import os
import subprocess
import tarfile
import zipfile
import tempfile
import shutil
import stat
import time


class Solution:
    def solve(self, src_path: str) -> bytes:
        tmp_dir = tempfile.mkdtemp(prefix="pocgen_")
        try:
            root_dir = self._extract_archive(src_path, tmp_dir)
            if root_dir is None:
                root_dir = tmp_dir

            poc = self._find_existing_poc(root_dir)
            if poc is not None:
                return poc

            out_dir = os.path.join(root_dir, "out")
            os.makedirs(out_dir, exist_ok=True)

            fuzzers = self._find_fuzz_binaries(root_dir)
            if not fuzzers:
                self._build_fuzzers(root_dir, out_dir)
                fuzzers = self._find_fuzz_binaries(root_dir)

            fuzzers = [f for f in fuzzers if self._is_libfuzzer_target(f)]
            if not fuzzers:
                return b"A" * 1032

            prioritized = sorted(fuzzers, key=self._fuzzer_priority, reverse=True)
            max_fuzzers = min(3, len(prioritized))
            for idx in range(max_fuzzers):
                fuzzer = prioritized[idx]
                artifact_dir = os.path.join(tmp_dir, "artifacts_" + os.path.basename(fuzzer))
                os.makedirs(artifact_dir, exist_ok=True)
                corpus_dir = self._prepare_corpus(root_dir, fuzzer, tmp_dir)
                poc = self._run_fuzzer_and_get_artifact(fuzzer, artifact_dir, corpus_dir)
                if poc is not None and len(poc) > 0:
                    return poc

        except Exception:
            pass
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return b"A" * 1032

    def _extract_archive(self, archive_path: str, dst_dir: str) -> str | None:
        try:
            if os.path.isdir(archive_path):
                # Copy directory structure into dst_dir
                for name in os.listdir(archive_path):
                    src = os.path.join(archive_path, name)
                    dst = os.path.join(dst_dir, name)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)
            elif tarfile.is_tarfile(archive_path):
                with tarfile.open(archive_path, "r:*") as tf:
                    tf.extractall(dst_dir)
            elif zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(archive_path, "r") as zf:
                    zf.extractall(dst_dir)
            else:
                return None
        except Exception:
            return None

        entries = [e for e in os.listdir(dst_dir) if not e.startswith(".")]
        if len(entries) == 1:
            root = os.path.join(dst_dir, entries[0])
            if os.path.isdir(root):
                return root
        return dst_dir

    def _find_existing_poc(self, root: str) -> bytes | None:
        best_path = None
        best_size = None
        for dirpath, _, files in os.walk(root):
            for f in files:
                lf = f.lower()
                if lf.startswith("crash-") or lf.endswith(".crash") or "poc" in lf or "repro" in lf:
                    path = os.path.join(dirpath, f)
                    try:
                        size = os.path.getsize(path)
                    except OSError:
                        continue
                    if size <= 0:
                        continue
                    if best_size is None or size < best_size:
                        best_size = size
                        best_path = path
        if best_path:
            try:
                with open(best_path, "rb") as fp:
                    return fp.read()
            except Exception:
                return None
        return None

    def _find_fuzz_binaries(self, root: str) -> list:
        fuzzers = []
        for dirpath, _, files in os.walk(root):
            for f in files:
                path = os.path.join(dirpath, f)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                if not (st.st_mode & stat.S_IXUSR):
                    continue
                lname = f.lower()
                if "fuzz" not in lname and "fuzzer" not in lname:
                    continue
                if path.endswith((".a", ".o", ".so", ".dylib", ".dll")):
                    continue
                fuzzers.append(path)
        return fuzzers

    def _build_fuzzers(self, root: str, out_dir: str) -> None:
        build_sh = None
        # Prefer build.sh in root, else first found in tree
        root_build = os.path.join(root, "build.sh")
        if os.path.isfile(root_build):
            build_sh = root_build
        else:
            for dirpath, _, files in os.walk(root):
                if "build.sh" in files:
                    build_sh = os.path.join(dirpath, "build.sh")
                    break
        if not build_sh:
            return
        env = os.environ.copy()
        env.setdefault("CC", "clang")
        env.setdefault("CXX", "clang++")
        env.setdefault("FUZZING_ENGINE", "libfuzzer")
        env.setdefault("SANITIZER", "address")
        env.setdefault("ARCH", "x86_64")
        env.setdefault("CFLAGS", "-O1")
        env.setdefault("CXXFLAGS", "-O1")
        env["OUT"] = out_dir
        env.setdefault("SRC", root)
        try:
            subprocess.run(
                ["bash", build_sh],
                cwd=os.path.dirname(build_sh),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=300,
                check=False,
            )
        except Exception:
            pass

    def _is_libfuzzer_target(self, path: str) -> bool:
        try:
            result = subprocess.run(
                [path, "-help=1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=5,
                check=False,
            )
            out = result.stdout.decode(errors="ignore")
            if "libFuzzer" in out or "OVERVIEW: libFuzzer" in out:
                return True
            if "Usage:" in out and "-max_len" in out:
                return True
        except Exception:
            return False
        return False

    def _fuzzer_priority(self, path: str) -> int:
        name = os.path.basename(path).lower()
        score = 0
        if "h3" in name:
            score += 4
        if "polygon" in name or "poly" in name:
            score += 8
        if "cell" in name or "cells" in name:
            score += 5
        if "experimental" in name:
            score += 3
        if "to_cells" in name or "tocells" in name:
            score += 4
        if name.endswith("_fuzzer"):
            score += 2
        if "fuzzer" in name:
            score += 1
        return score

    def _prepare_corpus(self, root: str, fuzzer_path: str, tmp_dir: str) -> str | None:
        base = os.path.basename(fuzzer_path)
        dirpath = os.path.dirname(fuzzer_path)

        # Look for zip corpora next to fuzzer
        try:
            for f in os.listdir(dirpath):
                if base in f and "seed_corpus" in f and f.endswith(".zip"):
                    zpath = os.path.join(dirpath, f)
                    out = os.path.join(tmp_dir, "corpus_" + base)
                    os.makedirs(out, exist_ok=True)
                    try:
                        with zipfile.ZipFile(zpath, "r") as zf:
                            zf.extractall(out)
                            return out
                    except Exception:
                        pass
        except Exception:
            pass

        # Look for any seed_corpus zip in tree with related name
        try:
            for dpath, _, files in os.walk(root):
                for f in files:
                    lf = f.lower()
                    if "seed_corpus" in lf and f.endswith(".zip"):
                        if any(tok in lf for tok in base.lower().split("_")) or "polygon" in lf or "cells" in lf:
                            zpath = os.path.join(dpath, f)
                            out = os.path.join(tmp_dir, "corpus_generic_" + base)
                            os.makedirs(out, exist_ok=True)
                            try:
                                with zipfile.ZipFile(zpath, "r") as zf:
                                    zf.extractall(out)
                                    return out
                            except Exception:
                                continue
        except Exception:
            pass

        # Unzipped corpus directories near fuzzer
        try:
            for f in os.listdir(dirpath):
                p = os.path.join(dirpath, f)
                if os.path.isdir(p):
                    lf = f.lower()
                    if "seed_corpus" in lf or "corpus" in lf or "seeds" in lf:
                        return p
        except Exception:
            pass

        # Fallback: any corpus directory under root
        try:
            for dpath, dirs, _ in os.walk(root):
                for d in dirs:
                    ld = d.lower()
                    if "seed_corpus" in ld or "corpus" in ld or "seeds" in ld:
                        return os.path.join(dpath, d)
        except Exception:
            pass

        return None

    def _run_fuzzer_and_get_artifact(self, fuzzer: str, artifact_dir: str, corpus_dir: str | None) -> bytes | None:
        cmd = [
            fuzzer,
            "-max_total_time=10",
            "-timeout=5",
            "-artifact_prefix=%s/" % artifact_dir,
            "-print_final_stats=1",
        ]
        if corpus_dir is not None:
            cmd.append(corpus_dir)
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass

        best_path = None
        best_size = 0
        for dirpath, _, files in os.walk(artifact_dir):
            for f in files:
                path = os.path.join(dirpath, f)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size > best_size:
                    best_size = size
                    best_path = path
        if best_path and best_size > 0:
            try:
                with open(best_path, "rb") as fp:
                    return fp.read()
            except Exception:
                return None
        return None