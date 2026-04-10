#!/usr/bin/env python3

import argparse
import math
import re
import statistics
import sys
from pathlib import Path


LOSS_RE = re.compile(r"step\s+(\d+):\s+train loss\s+([0-9.]+),\s+val loss\s+([0-9.]+)")


def parse_group(spec: str) -> tuple[str, list[Path]]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"invalid --group '{spec}': expected LABEL=LOG[,LOG...]"
        )
    label, raw_paths = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"invalid --group '{spec}': empty label")
    paths = [Path(part.strip()) for part in raw_paths.split(",") if part.strip()]
    if not paths:
        raise argparse.ArgumentTypeError(
            f"invalid --group '{spec}': expected at least one log path"
        )
    return label, paths


def load_curve(path: Path) -> dict[int, float]:
    curve: dict[int, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = LOSS_RE.search(line)
            if match:
                step = int(match.group(1))
                val_loss = float(match.group(3))
                curve[step] = val_loss
    if not curve:
        raise ValueError(f"no train/val loss lines found in {path}")
    return curve


def mean_curve(curves: list[dict[int, float]]) -> dict[int, float]:
    common_steps = set(curves[0])
    for curve in curves[1:]:
        common_steps &= set(curve)
    if not common_steps:
        raise ValueError("no common eval steps across logs")
    return {
        step: statistics.fmean(curve[step] for curve in curves)
        for step in sorted(common_steps)
    }


def first_crossing(curve: dict[int, float], threshold: float) -> int | None:
    for step in sorted(curve):
        if curve[step] <= threshold:
            return step
    return None


def format_step(step: int | None) -> str:
    return "not reached" if step is None else str(step)


def print_group_summary(
    label: str,
    group_curves: list[dict[int, float]],
    averaged_curve: dict[int, float],
    thresholds: list[float],
) -> None:
    final_step = max(averaged_curve)
    final_mean = averaged_curve[final_step]
    per_seed_final = [curve[final_step] for curve in group_curves]

    print(f"{label}:")
    print(
        f"  final mean val loss at step {final_step}: {final_mean:.4f}"
        f" (seed range {min(per_seed_final):.4f}-{max(per_seed_final):.4f})"
    )
    for threshold in thresholds:
        mean_step = first_crossing(averaged_curve, threshold)
        seed_steps = [first_crossing(curve, threshold) for curve in group_curves]
        seed_step_text = ", ".join(format_step(step) for step in seed_steps)
        print(
            f"  threshold <= {threshold:.4f}: mean curve {format_step(mean_step)}"
            f" | seeds [{seed_step_text}]"
        )


def winner_labels(pairs: list[tuple[str, float]], *, lower_is_better: bool = True) -> list[str]:
    if not pairs:
        return []
    key_fn = min if lower_is_better else max
    best_value = key_fn(value for _, value in pairs)
    return [label for label, value in pairs if math.isclose(value, best_value, abs_tol=1e-9)]


def print_dual_score_summary(
    labels: list[str],
    mean_curves: dict[str, dict[int, float]],
    thresholds: list[float],
) -> None:
    final_step = min(max(curve) for curve in mean_curves.values())
    endpoint_pairs = [(label, mean_curves[label][final_step]) for label in labels]
    endpoint_winners = winner_labels(endpoint_pairs)

    print("Dual-Score Summary:")
    print(
        f"  endpoint @ step {final_step}: "
        f"{', '.join(endpoint_winners)}"
        f" ({min(value for _, value in endpoint_pairs):.4f})"
    )

    for threshold in thresholds:
        threshold_pairs = []
        for label in labels:
            step = first_crossing(mean_curves[label], threshold)
            if step is not None:
                threshold_pairs.append((label, float(step)))
        if not threshold_pairs:
            print(f"  threshold <= {threshold:.4f}: no group reached it")
            continue
        threshold_winners = winner_labels(threshold_pairs)
        best_step = int(min(value for _, value in threshold_pairs))
        print(
            f"  threshold <= {threshold:.4f}: "
            f"{', '.join(threshold_winners)} first at step {best_step}"
        )

    print()


def compare_groups(
    labels: list[str],
    mean_curves: dict[str, dict[int, float]],
    thresholds: list[float],
) -> None:
    if len(labels) < 2:
        return

    base = labels[0]
    print("\nComparisons:")
    for other in labels[1:]:
        base_curve = mean_curves[base]
        other_curve = mean_curves[other]
        common_steps = sorted(set(base_curve) & set(other_curve))
        if not common_steps:
            continue

        final_step = common_steps[-1]
        final_delta = other_curve[final_step] - base_curve[final_step]
        if math.isclose(final_delta, 0.0, abs_tol=1e-9):
            final_text = "ties"
        elif final_delta < 0:
            final_text = f"beats {base} by {-final_delta:.4f}"
        else:
            final_text = f"trails {base} by {final_delta:.4f}"

        print(f"  {other} vs {base} at step {final_step}: {final_text}")
        for threshold in thresholds:
            base_step = first_crossing(base_curve, threshold)
            other_step = first_crossing(other_curve, threshold)
            if base_step is None or other_step is None:
                delta_text = "comparison unavailable"
            else:
                delta = other_step - base_step
                if delta == 0:
                    delta_text = "same crossing step"
                elif delta < 0:
                    delta_text = f"gets there {abs(delta)} steps earlier"
                else:
                    delta_text = f"gets there {delta} steps later"
            print(
                f"    threshold <= {threshold:.4f}: {other} {delta_text}"
                f" ({format_step(other_step)} vs {format_step(base_step)})"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare mean validation-loss threshold crossings across log groups."
    )
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        type=parse_group,
        metavar="LABEL=LOG[,LOG...]",
        help="Benchmark group label plus one or more comma-separated log paths.",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        required=True,
        help="Validation-loss threshold to score. Repeat for multiple thresholds.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    labels: list[str] = []
    mean_curves: dict[str, dict[int, float]] = {}
    grouped_curves: dict[str, list[dict[int, float]]] = {}

    for label, paths in args.group:
        labels.append(label)
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            parser.error(f"group '{label}' has missing logs: {', '.join(missing)}")
        curves = [load_curve(path) for path in paths]
        grouped_curves[label] = curves
        mean_curves[label] = mean_curve(curves)

    thresholds = sorted(set(args.threshold), reverse=True)

    print("Dual-Score Benchmark")
    print("====================")
    print(f"Thresholds: {', '.join(f'{threshold:.4f}' for threshold in thresholds)}")
    print()

    for label in labels:
        print_group_summary(label, grouped_curves[label], mean_curves[label], thresholds)
        print()

    print_dual_score_summary(labels, mean_curves, thresholds)
    compare_groups(labels, mean_curves, thresholds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
