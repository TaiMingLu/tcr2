#!/usr/bin/env python3
"""Workspace validator — checks scoring files and structural rules.

Usage:
    python validate.py                          # validate all
    python validate.py --reference-only         # validate reference.json only
    python validate.py --compare                # validate + compare scores against reference
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

WORKSPACE_DIR = Path(__file__).parent
SCORING_DIR = WORKSPACE_DIR / "scoring"


# ── Scoring validation ───────────────────────────────────────────────────────


def validate_reference(data: dict) -> list[str]:
    """Validate reference.json structure. Returns list of errors."""
    errors = []

    if "experiments" not in data:
        errors.append("Missing top-level 'experiments' key")
        return errors

    if not isinstance(data["experiments"], dict):
        errors.append("'experiments' must be a dict")
        return errors

    if len(data["experiments"]) == 0:
        errors.append("'experiments' is empty — add at least one experiment")

    # Validate weights sum to ~1.0
    weights = []
    for exp_name, exp in data["experiments"].items():
        if isinstance(exp, dict) and "weight" in exp:
            w = exp["weight"]
            if isinstance(w, (int, float)):
                weights.append(w)
    if weights:
        total = sum(weights)
        if abs(total - 1.0) > 0.01:
            errors.append(f"Experiment weights sum to {total:.3f}, expected 1.0")

    for exp_name, exp in data["experiments"].items():
        prefix = f"experiments.{exp_name}"

        if not isinstance(exp, dict):
            errors.append(f"{prefix}: must be a dict")
            continue

        # Required fields
        if "weight" not in exp:
            errors.append(f"{prefix}: missing 'weight'")
        elif not isinstance(exp["weight"], (int, float)):
            errors.append(f"{prefix}.weight: must be a number")
        elif not (0 < exp["weight"] <= 1):
            errors.append(f"{prefix}.weight: must be between 0 and 1")
        if "primary_metric" not in exp:
            errors.append(f"{prefix}: missing 'primary_metric'")
        if "metrics" not in exp:
            errors.append(f"{prefix}: missing 'metrics'")
        if "results" not in exp:
            errors.append(f"{prefix}: missing 'results'")

        # Validate metrics
        metrics = exp.get("metrics", {})
        if not isinstance(metrics, dict):
            errors.append(f"{prefix}.metrics: must be a dict")
            metrics = {}

        for metric_name, metric_def in metrics.items():
            if not isinstance(metric_def, dict):
                errors.append(f"{prefix}.metrics.{metric_name}: must be a dict")
                continue
            if "higher_is_better" not in metric_def:
                errors.append(f"{prefix}.metrics.{metric_name}: missing 'higher_is_better'")
            elif not isinstance(metric_def["higher_is_better"], bool):
                errors.append(f"{prefix}.metrics.{metric_name}.higher_is_better: must be a bool")
            if "coefficient" not in metric_def:
                errors.append(f"{prefix}.metrics.{metric_name}: missing 'coefficient'")
            elif not isinstance(metric_def["coefficient"], (int, float)):
                errors.append(f"{prefix}.metrics.{metric_name}.coefficient: must be a number")

        # primary_metric must be in metrics
        primary = exp.get("primary_metric")
        if primary and metrics and primary not in metrics:
            errors.append(f"{prefix}: primary_metric '{primary}' not found in metrics")

        # Validate results
        results = exp.get("results", {})
        if not isinstance(results, dict):
            errors.append(f"{prefix}.results: must be a dict")
            continue

        if len(results) == 0:
            errors.append(f"{prefix}.results: empty — add at least one method")

        has_proposed = False
        for method_name, method in results.items():
            mp = f"{prefix}.results.{method_name}"
            if not isinstance(method, dict):
                errors.append(f"{mp}: must be a dict")
                continue

            # type is required
            if "type" not in method:
                errors.append(f"{mp}: missing 'type' (must be 'baseline' or 'proposed')")
            elif method["type"] not in ("baseline", "proposed"):
                errors.append(f"{mp}: type must be 'baseline' or 'proposed', got '{method['type']}'")
            elif method["type"] == "proposed":
                has_proposed = True

            # Check that metric values are numbers and use known metric names
            for key, val in method.items():
                if key == "type":
                    continue
                if key not in metrics:
                    errors.append(f"{mp}: unknown metric '{key}' (not in metrics)")
                elif not isinstance(val, (int, float)):
                    errors.append(f"{mp}.{key}: must be a number, got {type(val).__name__}")

            # Check primary metric is present
            if primary and primary not in method and method.get("type") != "baseline":
                errors.append(f"{mp}: missing primary metric '{primary}'")

        if not has_proposed:
            errors.append(f"{prefix}.results: no method with type 'proposed' — at least one is expected")

    return errors


def validate_scores(data: dict, reference: dict | None = None) -> list[str]:
    """Validate scores.json structure. Returns list of errors."""
    errors = []

    if not isinstance(data, dict):
        errors.append("scores.json must be a dict")
        return errors

    # Accept both formats: {experiments: {...}} or flat {exp: {method: {...}}}
    if "experiments" in data and isinstance(data["experiments"], dict):
        experiments = data["experiments"]
    else:
        experiments = data

    if len(experiments) == 0:
        errors.append("scores.json is empty")
        return errors

    ref_exps = {}
    if reference and "experiments" in reference:
        ref_exps = reference["experiments"]

    for exp_name, exp_data in experiments.items():
        if exp_name.startswith("_"):
            continue

        if not isinstance(exp_data, dict):
            errors.append(f"{exp_name}: must be a dict")
            continue

        if exp_name not in ref_exps and ref_exps:
            errors.append(f"{exp_name}: not found in reference.json experiments")
            continue

        ref_metrics = ref_exps.get(exp_name, {}).get("metrics", {}) if ref_exps else {}
        ref_results = ref_exps.get(exp_name, {}).get("results", {}) if ref_exps else {}

        # Accept both: {results: {method: {metrics}}} or flat {method: {metrics}}
        if "results" in exp_data and isinstance(exp_data["results"], dict):
            methods = exp_data["results"]
        else:
            methods = exp_data

        for method_name, method_scores in methods.items():
            if method_name.startswith("_"):
                continue
            mp = f"{exp_name}.{method_name}"

            if not isinstance(method_scores, dict):
                errors.append(f"{mp}: must be a dict of metric_name -> value")
                continue

            if ref_results and method_name not in ref_results:
                errors.append(f"{mp}: method not found in reference.json")

            for metric_name, val in method_scores.items():
                if not isinstance(val, (int, float)):
                    errors.append(f"{mp}.{metric_name}: must be a number, got {type(val).__name__}")
                if ref_metrics and metric_name not in ref_metrics:
                    errors.append(f"{mp}.{metric_name}: not a known metric in reference.json")

    return errors


def _extract_scores_experiments(scores: dict) -> dict:
    """Normalize scores.json into {exp_name: {method_name: {metrics}}}."""
    if "experiments" in scores and isinstance(scores["experiments"], dict):
        raw = scores["experiments"]
    else:
        raw = scores
    out = {}
    for exp_name, exp_data in raw.items():
        if exp_name.startswith("_") or not isinstance(exp_data, dict):
            continue
        if "results" in exp_data and isinstance(exp_data["results"], dict):
            out[exp_name] = exp_data["results"]
        else:
            out[exp_name] = exp_data
    return out


def compare_scores(scores: dict, reference: dict) -> None:
    """Print side-by-side comparison of reproduced scores against reference."""
    exps = reference.get("experiments", {})
    score_exps = _extract_scores_experiments(scores)

    for exp_name, exp_methods in score_exps.items():
        exp_ref = exps.get(exp_name)
        if not exp_ref:
            continue

        weight = exp_ref.get("weight", "?")
        print(f"\n=== {exp_name} (weight: {weight}) ===")
        if exp_ref.get("description"):
            print(f"    {exp_ref['description']}")

        primary = exp_ref.get("primary_metric")
        metrics = exp_ref.get("metrics", {})
        ref_results = exp_ref.get("results", {})

        all_metrics = list(metrics.keys())

        print(f"\n    {'Method':<25} {'Source':<10}", end="")
        for m in all_metrics:
            marker = " *" if m == primary else ""
            print(f"{(m + marker):>15}", end="")
        print()
        print(f"    {'-'*25} {'-'*10}", end="")
        for _ in all_metrics:
            print(f"{'─'*15}", end="")
        print()

        all_methods = list(dict.fromkeys(list(ref_results.keys()) + list(exp_methods.keys())))
        for method_name in all_methods:
            ref_method = ref_results.get(method_name, {})
            score_method = exp_methods.get(method_name)
            tag = f" [{ref_method['type']}]" if ref_method.get("type") else ""

            print(f"    {(method_name + tag):<25} {'paper':<10}", end="")
            for m in all_metrics:
                val = ref_method.get(m)
                print(f"{val:>15.3f}" if val is not None else f"{'—':>15}", end="")
            print()

            if score_method:
                print(f"    {'':<25} {'repro':<10}", end="")
                for m in all_metrics:
                    val = score_method.get(m)
                    print(f"{val:>15.3f}" if val is not None else f"{'—':>15}", end="")
                print()
            print()


# ── Structural checks ────────────────────────────────────────────────────────


def check_import_separation() -> list[str]:
    """Check that eval/ and data/ do not import from method/."""
    errors = []
    pattern = re.compile(r"^\s*(?:from\s+method\b|import\s+method\b)", re.MULTILINE)
    for dirname in ("eval", "data"):
        dirpath = WORKSPACE_DIR / dirname
        if not dirpath.is_dir():
            continue
        for pyfile in dirpath.rglob("*.py"):
            try:
                content = pyfile.read_text()
            except Exception:
                continue
            if pattern.search(content):
                rel = pyfile.relative_to(WORKSPACE_DIR)
                errors.append(f"{rel}: imports from method/")
    return errors


def _resolve_local_import(module: str, from_file: Path | None, level: int) -> Path | None:
    """Resolve a dotted module name to a file inside the workspace, if any.

    Returns None if the import does not resolve to a workspace-local file
    (i.e. it is stdlib, third-party, or unresolvable).
    """
    if level > 0 and from_file is not None:
        anchor = from_file.parent
        for _ in range(level - 1):
            anchor = anchor.parent
        parts = module.split(".") if module else []
        candidate_base = anchor.joinpath(*parts) if parts else anchor
    else:
        if not module:
            return None
        parts = module.split(".")
        candidate_base = WORKSPACE_DIR.joinpath(*parts)

    # Must live inside the workspace
    try:
        candidate_base.resolve().relative_to(WORKSPACE_DIR.resolve())
    except ValueError:
        return None

    py_file = candidate_base.with_suffix(".py")
    if py_file.is_file():
        return py_file
    init_file = candidate_base / "__init__.py"
    if init_file.is_file():
        return init_file
    return None


# Top-level workspace directories that always hold local code / data. An
# import whose first dotted component matches this set is workspace-local by
# convention, even if the directory doesn't exist in the current tree (e.g.
# a fresh clone before download.sh runs).
_WORKSPACE_IMPORT_PREFIXES = frozenset({"data", "method", "eval", "baseline"})


def _expected_import_base(module: str, from_file: Path | None, level: int) -> Path | None:
    """Compute the expected workspace path for an import (existence not required).

    Returns the candidate base path (without .py suffix) if the import looks
    workspace-local, else None. Workspace-local = relative imports anchored
    in from_file OR absolute imports whose first dotted component is a known
    workspace directory.

    Policy: every workspace-local import must resolve to a git-tracked file.
    This includes vendored third-party libraries under ``method/`` — they
    should be committed, not gitignored, so fresh clones work without a
    runtime clone step (up to reasonable size limits).
    """
    if level > 0 and from_file is not None:
        anchor = from_file.parent
        for _ in range(level - 1):
            anchor = anchor.parent
        parts = module.split(".") if module else []
        candidate_base = anchor.joinpath(*parts) if parts else anchor
    else:
        if not module:
            return None
        parts = module.split(".")
        if parts[0] not in _WORKSPACE_IMPORT_PREFIXES:
            return None
        candidate_base = WORKSPACE_DIR.joinpath(*parts)
    try:
        candidate_base.resolve().relative_to(WORKSPACE_DIR.resolve())
    except ValueError:
        return None
    return candidate_base


def check_imports_not_gitignored() -> list[str]:
    """Check that every workspace-local import in eval/, method/, baseline/
    resolves to a file that git *tracks*.

    Two failure modes both break a fresh clone:
      (a) import target exists on disk but is gitignored (won't be committed)
      (b) import target does not exist at all in git (file missing entirely)

    Both fail here — (a) was historically caught by walking the disk and
    cross-checking ``git check-ignore``; (b) was silently skipped because the
    old resolver returned None for missing files. Both now resolve to the
    expected path and are validated against the tracked fileset.
    """
    errors: list[str] = []
    # expected_path -> list of (source_file, lineno) tuples
    import_map: dict[Path, list[tuple[Path, int]]] = {}

    for dirname in ("eval", "method", "baseline"):
        dirpath = WORKSPACE_DIR / dirname
        if not dirpath.is_dir():
            continue
        for pyfile in dirpath.rglob("*.py"):
            try:
                source = pyfile.read_text()
            except Exception:
                continue
            try:
                tree = ast.parse(source, filename=str(pyfile))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        base = _expected_import_base(alias.name, None, 0)
                        if base is not None:
                            import_map.setdefault(base, []).append((pyfile, node.lineno))
                elif isinstance(node, ast.ImportFrom):
                    if node.module is None and node.level == 0:
                        continue
                    base = _expected_import_base(node.module or "", pyfile, node.level)
                    if base is not None:
                        import_map.setdefault(base, []).append((pyfile, node.lineno))

    if not import_map:
        return errors

    # Ask git for the full tracked fileset once, then match each expected path
    # against it. A ".py" file or a "/__init__.py" package either exists in the
    # tree or it doesn't — anything else is a fresh-clone failure.
    try:
        tracked_result = subprocess.run(
            ["git", "-C", str(WORKSPACE_DIR), "ls-files"],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return errors
    tracked = {line.strip() for line in tracked_result.stdout.splitlines() if line.strip()}

    import glob as _glob

    for base, callsites in sorted(import_map.items(), key=lambda kv: str(kv[0])):
        rel_base = base.relative_to(WORKSPACE_DIR)
        py_rel = str(rel_base) + ".py"
        pkg_rel = str(rel_base) + "/__init__.py"
        if py_rel in tracked or pkg_rel in tracked:
            continue  # import target is tracked — fresh clone is fine
        # Compiled extensions (.so, .pyd, .cpython-*.so) are built at install
        # time and are never committed to git. Skip if one exists on disk or
        # if the parent directory contains a build system file that would
        # produce one (setup.py, CMakeLists.txt, meson.build, pyproject.toml
        # with a compiled backend, or pybind11 sources).
        module_name = base.name
        parent_dir = base.parent
        ext_globs = [
            str(parent_dir / f"{module_name}.*.so"),
            str(parent_dir / f"{module_name}.*.pyd"),
            str(parent_dir / f"{module_name}.so"),
            str(parent_dir / f"{module_name}.pyd"),
        ]
        if any(_glob.glob(g) for g in ext_globs):
            continue  # compiled extension on disk — built at install time
        build_markers = ("setup.py", "CMakeLists.txt", "meson.build", "pyproject.toml",
                         "Cargo.toml")
        # Walk up from the module's parent to the workspace root looking for
        # a build system file that would produce this compiled extension.
        search = parent_dir
        _found_build = False
        while True:
            if any((search / m).exists() for m in build_markers):
                _found_build = True
                break
            if search == WORKSPACE_DIR or search == search.parent:
                break
            search = search.parent
        if _found_build:
            continue
        for source_file, lineno in callsites:
            rel_source = source_file.relative_to(WORKSPACE_DIR)
            on_disk = (
                (WORKSPACE_DIR / py_rel).is_file()
                or (WORKSPACE_DIR / pkg_rel).is_file()
            )
            reason = (
                f"exists locally but is gitignored"
                if on_disk
                else f"not present in git at all"
            )
            errors.append(
                f"{rel_source}:{lineno}: imports {rel_base} ({py_rel} or "
                f"{pkg_rel}), which is {reason} — fresh clone will hit ImportError"
            )
    return errors


def check_train_test_independent() -> list[str]:
    """Check that eval/train/ and eval/test/ do not import from each other.

    The two slices must be independently runnable. A shortcut agent that reads
    eval/train/ should not be able to learn anything about eval/test/'s shape,
    metrics, or data through a sibling import.
    """
    errors: list[str] = []
    train_dir = WORKSPACE_DIR / "eval" / "train"
    test_dir = WORKSPACE_DIR / "eval" / "test"

    def _scan(src_dir: Path, forbidden_dir: Path, forbidden_label: str) -> None:
        if not src_dir.is_dir() or not forbidden_dir.is_dir():
            return
        for pyfile in src_dir.rglob("*.py"):
            try:
                source = pyfile.read_text()
            except Exception:
                continue
            try:
                tree = ast.parse(source, filename=str(pyfile))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                resolved = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        r = _resolve_local_import(alias.name, None, 0)
                        if r is not None:
                            try:
                                r.resolve().relative_to(forbidden_dir.resolve())
                                resolved = r
                                break
                            except ValueError:
                                pass
                elif isinstance(node, ast.ImportFrom):
                    if node.module is None and node.level == 0:
                        continue
                    r = _resolve_local_import(node.module or "", pyfile, node.level)
                    if r is not None:
                        try:
                            r.resolve().relative_to(forbidden_dir.resolve())
                            resolved = r
                        except ValueError:
                            pass
                if resolved is not None:
                    rel_src = pyfile.relative_to(WORKSPACE_DIR)
                    rel_dst = resolved.relative_to(WORKSPACE_DIR)
                    errors.append(
                        f"{rel_src}:{node.lineno}: imports {rel_dst} "
                        f"(eval/train and eval/{forbidden_label} must be independent)"
                    )

    _scan(train_dir, test_dir, "test")
    _scan(test_dir, train_dir, "train")
    return errors


_SHARED_PATTERN = re.compile(r"(?<![A-Za-z0-9_/])/?shared/(?:datasets|models|hf_cache)\b")


def check_no_shared_references() -> list[str]:
    """Check that committed code does NOT reference shared/ paths.

    `shared/` is a local cache on the swarm machine and does not exist
    on a fresh clone. Any reference in committed code (method/, eval/,
    baseline/, scripts/ including download.sh) breaks reproducibility.
    """
    errors: list[str] = []
    for dirname in ("method", "eval", "baseline", "scripts"):
        dirpath = WORKSPACE_DIR / dirname
        if not dirpath.is_dir():
            continue
        for fpath in dirpath.rglob("*"):
            if not fpath.is_file():
                continue
            if fpath.suffix not in (".py", ".sh"):
                continue
            try:
                lines = fpath.read_text().splitlines()
            except Exception:
                continue
            rel = fpath.relative_to(WORKSPACE_DIR)
            for lineno, line in enumerate(lines, 1):
                if _SHARED_PATTERN.search(line):
                    errors.append(
                        f"{rel}:{lineno}: references shared/ "
                        f"(won't exist on fresh clone — use data/downloads/ instead)"
                    )
    return errors


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Validate workspace")
    parser.add_argument("--reference-only", action="store_true",
                        help="Only validate reference.json")
    parser.add_argument("--compare", action="store_true",
                        help="Compare scores against reference")
    args = parser.parse_args()

    ok = True

    # 1. Scoring: reference.json
    print("── Scoring ──")
    ref_path = SCORING_DIR / "reference.json"
    if not ref_path.exists():
        print("  ✗ scoring/reference.json not found")
        sys.exit(1)

    ref_data = json.loads(ref_path.read_text())
    ref_errors = validate_reference(ref_data)

    if ref_errors:
        print(f"  reference.json — {len(ref_errors)} error(s):")
        for e in ref_errors:
            print(f"    ✗ {e}")
        ok = False
    else:
        n_exp = len(ref_data.get("experiments", {}))
        n_methods = sum(len(e.get("results", {}))
                        for e in ref_data.get("experiments", {}).values())
        print(f"  reference.json — OK ({n_exp} experiments, {n_methods} methods)")

    # 2. Scoring: scores.json (and scores_train.json / scores_test.json if
    # present — those are produced by the designer iteration; earlier
    # iterations don't create them and are not checked for them).
    if not args.reference_only:
        for filename in ("scores.json", "scores_train.json", "scores_test.json"):
            scores_path = SCORING_DIR / filename
            if not scores_path.exists():
                # scores.json absence is surfaced; the designer-produced
                # split files are silently skipped when missing so earlier
                # iterations don't get spurious noise.
                if filename == "scores.json":
                    print(f"  {filename} — not found (skipping)")
                continue

            scores_data = json.loads(scores_path.read_text())
            scores_errors = validate_scores(scores_data, ref_data)

            if scores_errors:
                print(f"  {filename} — {len(scores_errors)} error(s):")
                for e in scores_errors:
                    print(f"    ✗ {e}")
                ok = False
            else:
                n_exp = len(scores_data)
                n_methods = sum(len(m) for m in scores_data.values()
                                if isinstance(m, dict))
                print(f"  {filename} — OK ({n_exp} experiments, {n_methods} methods)")

            if args.compare and not scores_errors and filename == "scores.json":
                compare_scores(scores_data, ref_data)

        # 2b. Designer-produced shell entry points: if the designer wrote a
        # scores_train.json / scores_test.json, a matching evaluate_train.sh /
        # evaluate_test.sh must also exist alongside scripts/evaluate.sh. Only
        # checked when the corresponding score file is present, so pre-designer
        # iterations don't get flagged.
        scripts_dir = WORKSPACE_DIR / "scripts"
        for score_name, script_name in (
            ("scores_train.json", "evaluate_train.sh"),
            ("scores_test.json", "evaluate_test.sh"),
        ):
            if not (SCORING_DIR / score_name).exists():
                continue
            script_path = scripts_dir / script_name
            if not script_path.exists():
                print(
                    f"  ✗ scripts/{script_name} missing "
                    f"(required when scoring/{score_name} is present)"
                )
                ok = False
            else:
                print(f"  scripts/{script_name} — OK")

    # 3. Import separation
    print("\n── Structure ──")
    sep_errors = check_import_separation()
    if sep_errors:
        print(f"  import separation — {len(sep_errors)} error(s):")
        for e in sep_errors:
            print(f"    ✗ {e}")
        ok = False
    else:
        print("  import separation — OK (eval/ and data/ do not import from method/)")

    # 4. Imports don't land on gitignored files (would break on fresh clone)
    ignored_errors = check_imports_not_gitignored()
    if ignored_errors:
        print(f"  gitignored imports — {len(ignored_errors)} error(s):")
        for e in ignored_errors:
            print(f"    ✗ {e}")
        ok = False
    else:
        print("  gitignored imports — OK (all eval/method/baseline imports resolve to tracked files)")

    # 5. eval/train and eval/test are independent (no cross-imports)
    indep_errors = check_train_test_independent()
    if indep_errors:
        print(f"  eval/train ↔ eval/test independence — {len(indep_errors)} error(s):")
        for e in indep_errors:
            print(f"    ✗ {e}")
        ok = False
    else:
        print("  eval/train ↔ eval/test independence — OK")

    # 6. No shared/ references in committed code (would break on fresh clone)
    shared_errors = check_no_shared_references()
    if shared_errors:
        print(f"  shared/ references — {len(shared_errors)} error(s):")
        for e in shared_errors:
            print(f"    ✗ {e}")
        ok = False
    else:
        print("  shared/ references — OK (no shared/ paths in committed code)")

    # Summary
    print()
    if ok:
        print("All checks passed.")
    else:
        print("FAILED — fix the errors above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
