import json
from collections import defaultdict

import numpy as np
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment


GT_PATH = "ground_truth.json"
PRED_PATH = "prediction.json"


LABEL_MAP = {
    "doc_title": "title",
    "paragraph_title": "paragraph_title",
    "section_title": "paragraph_title",
    "subsection_title": "paragraph_title",
    "text": "text",
    "abstract": "abstract",
    "figure_title": "figure_caption",
    "figure_caption": "figure_caption",
    "table_title": "table_caption",
    "table_caption": "table_caption",
    "image": "image",
    "footnote": "footnote",
    "footer": "footer",
    "aside_text": "aside_text"
}


def load_blocks(path):
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data["blocks"]

    return data


def normalize_label(block):
    label = block.get("label")
    if label is None:
        label = block.get("type")

    return LABEL_MAP.get(label, label)


def group_by_label(blocks):
    groups = defaultdict(list)

    for block in blocks:
        label = normalize_label(block)

        if label is None:
            continue

        groups[label].append(block)

    return groups


def similarity(a, b):
    return Levenshtein.normalized_similarity(a, b)


def evaluate_label(label, gt_blocks, pred_blocks):

    print("=" * 120)
    print(f"LABEL : {label}")
    print("=" * 120)

    n = len(gt_blocks)
    m = len(pred_blocks)

    if n == 0 and m == 0:
        return None

    if n == 0:
        print("No Ground Truth.\n")
        return {
            "gt": 0,
            "pred": m,
            "matched": 0,
            "score": 0
        }

    if m == 0:
        print("No Prediction.\n")
        return {
            "gt": n,
            "pred": 0,
            "matched": 0,
            "score": 0
        }

    cost = np.ones((n, m))

    for i in range(n):
        for j in range(m):
            cost[i, j] = 1 - similarity(
                gt_blocks[i]["text"],
                pred_blocks[j]["text"]
            )

    rows, cols = linear_sum_assignment(cost)

    total_score = 0

    for r, c in zip(rows, cols):

        score = 1 - cost[r, c]
        total_score += score

        print(f"GT   [{r}]")
        print(gt_blocks[r]["text"][:120])

        print()

        print(f"PRED [{c}]")
        print(pred_blocks[c]["text"][:120])

        print()
        print(f"Similarity : {score:.4f}")
        print("-" * 120)

    final_score = total_score / max(n, m)

    return {
        "gt": n,
        "pred": m,
        "matched": len(rows),
        "score": final_score
    }


def main():

    gt = load_blocks(GT_PATH)
    pred = load_blocks(PRED_PATH)

    gt_groups = group_by_label(gt)
    pred_groups = group_by_label(pred)

    labels = sorted(
        set(gt_groups.keys()) |
        set(pred_groups.keys())
    )

    results = []

    for label in labels:

        result = evaluate_label(
            label,
            gt_groups.get(label, []),
            pred_groups.get(label, [])
        )

        if result is None:
            continue

        results.append((label, result))

    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    overall = []

    for label, r in results:

        overall.append(r["score"])

        print(
            f"{label:20}"
            f" GT={r['gt']:4}"
            f" Pred={r['pred']:4}"
            f" Match={r['matched']:4}"
            f" Score={r['score']:.4f}"
        )

    print("=" * 80)
    print(f"Overall Score : {np.mean(overall):.4f}")


if __name__ == "__main__":
    main()