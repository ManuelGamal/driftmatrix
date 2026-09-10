"""CI regression gate (spec 12). Non-zero exit on regression."""
from __future__ import annotations
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from .config import NOISE_FLOOR_NATS


@dataclass(frozen=True)
class GateResult:
    passed: bool
    violations: list


def write_baseline(summary: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(summary, sort_keys=True, indent=2),
                          encoding="utf-8")


def check_against_baseline(summary: dict, baseline_path, tolerance: float = 0.05):
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    violations = []
    for key, base_val in baseline.items():
        if key not in summary:
            violations.append(f"missing metric: {key}")
            continue
        cur = float(summary[key])
        allowed = abs(float(base_val)) * tolerance
        if abs(cur - float(base_val)) > allowed:
            violations.append(
                f"{key}: {cur} deviates from baseline {base_val} "
                f"beyond tolerance {tolerance}")
    floor = summary.get("noise_floor")
    if floor is not None and float(floor) >= NOISE_FLOOR_NATS:
        violations.append(
            f"noise_floor {floor} >= spec threshold {NOISE_FLOOR_NATS} nats")
    return GateResult(not violations, violations)


def main() -> int:
    summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    result = check_against_baseline(summary, Path(sys.argv[2]))
    for v in result.violations:
        print(f"GATE VIOLATION: {v}", file=sys.stderr)
    print("GATE PASSED" if result.passed else "GATE FAILED")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
