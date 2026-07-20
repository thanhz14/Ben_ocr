#!/usr/bin/env python
"""
Benchmark OCR/document-parsing results per label, following the OmniDocBench approach.

Metrics by category
-------------------
  text    : normalized edit distance (edit_similarity)
  formula : normalized edit distance + CDM  (from metric/formula.py → metric/cdm_metric.py)
  table   : normalized edit distance + TEDS (from metric/table.py  → metric/table_metric.py)

Usage
-----
  python benchmark_main.py [--gt ground_truth.json] [--pred prediction.json] [--output result.json]
"""
import argparse
import json
from collections import defaultdict

from rapidfuzz.distance import Levenshtein

from metric.fomula import FormulaMetric
from metric.table import TableMetric


# ---------------------------------------------------------------------------
# Label → category mapping
# Add new label names here as the dataset evolves.
# ---------------------------------------------------------------------------
FORMULA_LABELS: set[str] = {
    "formula", "equation", "inline_formula", "display_formula", "math",
}
TABLE_LABELS: set[str] = {
    "table",
}


def get_category(label: str) -> str:
    """Return 'text', 'formula', or 'table' for a given block label."""
    norm = label.lower().strip()
    if norm in TABLE_LABELS:
        return "table"
    if norm in FORMULA_LABELS:
        return "formula"
    return "text"


# ---------------------------------------------------------------------------
# Block matching
# ---------------------------------------------------------------------------

def _block_text(block: dict) -> str:
    return (block.get("text") or "").strip()


def match_blocks_by_similarity(
    gt_blocks: list[dict],
    pred_blocks: list[dict],
) -> list[tuple[dict, dict]]:
    """
    Greedy 1-to-1 matching: each GT block is paired with the unmatched
    prediction block that has the highest normalized text similarity.

    Returns a list of (gt_block, pred_block) pairs.
    Unmatched GT blocks are excluded from returned pairs.
    """
    if not gt_blocks or not pred_blocks:
        return []

    matched_pred: set[int] = set()
    pairs: list[tuple[dict, dict]] = []

    for gt in gt_blocks:
        gt_text = _block_text(gt)
        best_idx = -1
        best_sim = -1.0

        for i, pred in enumerate(pred_blocks):
            if i in matched_pred:
                continue
            sim = Levenshtein.normalized_similarity(gt_text, _block_text(pred))
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx >= 0:
            pairs.append((gt, pred_blocks[best_idx]))
            matched_pred.add(best_idx)

    return pairs


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(
    gt_path: str,
    pred_path: str,
    output_path: str | None = None,
) -> dict:
    """
    Load GT and prediction files, match blocks, compute metrics per label,
    and return a results dict.
    """
    with open(gt_path, encoding="utf-8") as f:
        gt_data = json.load(f)
    with open(pred_path, encoding="utf-8") as f:
        pred_data = json.load(f)

    gt_blocks: list[dict] = gt_data.get("blocks", [])
    pred_blocks: list[dict] = pred_data.get("blocks", [])

    # --- separate blocks by category ---
    gt_by_cat: dict[str, list[dict]] = defaultdict(list)
    for b in gt_blocks:
        cat = get_category(b.get("label", "text"))
        gt_by_cat[cat].append(b)

    pred_by_cat: dict[str, list[dict]] = defaultdict(list)
    for b in pred_blocks:
        # prediction may use "type" or "label" for the class name
        label = b.get("type") or b.get("label") or "text"
        cat = get_category(label)
        pred_by_cat[cat].append(b)

    # accumulate per-label metric lists
    # structure: label → metric_name → [values]
    per_label: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    formula_metric = FormulaMetric()
    table_metric = TableMetric()

    # ---- TEXT ---------------------------------------------------------------
    text_pairs = match_blocks_by_similarity(gt_by_cat["text"], pred_by_cat["text"])
    for gt, pred in text_pairs:
        label = gt.get("label", "text")
        gt_text = _block_text(gt)
        pred_text = _block_text(pred)
        sim = Levenshtein.normalized_similarity(gt_text, pred_text)
        per_label[label]["edit_similarity"].append(sim)

    # ---- FORMULA ------------------------------------------------------------
    formula_pairs = match_blocks_by_similarity(
        gt_by_cat["formula"], pred_by_cat["formula"]
    )
    for idx, (gt, pred) in enumerate(formula_pairs):
        label = gt.get("label", "formula")
        gt_text = _block_text(gt)
        pred_text = _block_text(pred)

        sim = Levenshtein.normalized_similarity(gt_text, pred_text)
        per_label[label]["edit_similarity"].append(sim)

        if gt_text and pred_text:
            sample_id = f"{gt.get('id', idx)}"
            cdm_result = formula_metric.cdm_score(gt_text, pred_text, sample_id)
            per_label[label]["cdm_f1"].append(cdm_result.get("F1_score", 0.0))

    # ---- TABLE --------------------------------------------------------------
    table_pairs = match_blocks_by_similarity(
        gt_by_cat["table"], pred_by_cat["table"]
    )
    for gt, pred in table_pairs:
        label = gt.get("label", "table")
        gt_text = _block_text(gt)
        pred_text = _block_text(pred)

        sim = Levenshtein.normalized_similarity(gt_text, pred_text)
        per_label[label]["edit_similarity"].append(sim)

        if gt_text and pred_text:
            teds = table_metric.teds_score(gt_text, pred_text)
            per_label[label]["teds"].append(teds if teds is not None else 0.0)

    # --- aggregate: mean per label ---
    results: dict[str, dict] = {}
    for label, metrics in per_label.items():
        count = len(metrics.get("edit_similarity", []))
        agg: dict = {"count": count, "category": get_category(label)}
        for metric_name, values in metrics.items():
            if values:
                agg[metric_name] = sum(values) / len(values)
        results[label] = agg

    _print_results(results)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

    return results


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_results(results: dict) -> None:
    METRIC_COLS = ["edit_similarity", "cdm_f1", "teds"]

    # determine which metric columns are present
    active_cols = [m for m in METRIC_COLS if any(m in v for v in results.values())]

    col_w = 16
    label_w = 22
    header_parts = [f"{'label':<{label_w}}", f"{'category':<10}", f"{'count':>6}"]
    for col in active_cols:
        header_parts.append(f"{col:>{col_w}}")
    header = "  ".join(header_parts)

    sep = "-" * len(header)
    print("\n" + sep)
    print("BENCHMARK RESULTS PER LABEL")
    print(sep)
    print(header)
    print(sep)

    category_order = {"text": 0, "formula": 1, "table": 2}
    for label in sorted(results, key=lambda l: (category_order.get(results[l]["category"], 99), l)):
        row = results[label]
        parts = [
            f"{label:<{label_w}}",
            f"{row['category']:<10}",
            f"{row['count']:>6}",
        ]
        for col in active_cols:
            val = row.get(col)
            parts.append(f"{val:>{col_w}.4f}" if val is not None else f"{'N/A':>{col_w}}")
        print("  ".join(parts))

    # category summary
    cat_acc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in results.values():
        cat = row["category"]
        for m in active_cols:
            if m in row:
                cat_acc[cat][m].append(row[m])

    print(sep)
    print("CATEGORY AVERAGES")
    print(sep)
    for cat in ["text", "formula", "table"]:
        if cat not in cat_acc:
            continue
        parts = [f"{cat:<{label_w}}", f"{'':10}", f"{'':>6}"]
        for col in active_cols:
            vals = cat_acc[cat].get(col, [])
            avg = sum(vals) / len(vals) if vals else None
            parts.append(f"{avg:>{col_w}.4f}" if avg is not None else f"{'N/A':>{col_w}}")
        print("  ".join(parts))
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark OCR/document-parsing per label (OmniDocBench style)"
    )
    parser.add_argument(
        "--gt",
        default="ground_truth.json",
        help="Path to ground-truth JSON (default: ground_truth.json)",
    )
    parser.add_argument(
        "--pred",
        default="prediction.json",
        help="Path to prediction JSON (default: prediction.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save JSON results",
    )
    args = parser.parse_args()
    evaluate(args.gt, args.pred, args.output)


if __name__ == "__main__":
    main()
