import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import evaluate_page, match_blocks, normalize_label  # noqa: E402


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def paper_number(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("gt_"))
    except ValueError as error:
        raise ValueError(f"Unexpected GT filename: {path.name}") from error


def calculate_agreement(annotator_1_dir: Path, annotator_2_dir: Path) -> dict:
    files_1 = {paper_number(path): path for path in annotator_1_dir.glob("gt_*.json")}
    files_2 = {paper_number(path): path for path in annotator_2_dir.glob("gt_*.json")}
    if not files_1:
        raise ValueError(f"No gt_*.json files found in {annotator_1_dir}")
    if not files_2:
        raise ValueError(f"No gt_*.json files found in {annotator_2_dir}")

    totals = defaultdict(
        lambda: {
            "annotator_1_blocks": 0,
            "annotator_2_blocks": 0,
            "matched": 0,
            "iou_scores": [],
            "content_similarities": [],
            "cdm_scores": [],
            "teds_scores": [],
        }
    )
    papers_1 = set(files_1)
    papers_2 = set(files_2)
    common_papers = sorted(papers_1 & papers_2)
    evaluated_pages = 0
    page_mismatches = []

    for position, paper_id in enumerate(common_papers, start=1):
        document_1 = load_json(files_1[paper_id])
        document_2 = load_json(files_2[paper_id])
        pages_1 = {page["page_index"]: page for page in document_1.get("pages", [])}
        pages_2 = {page["page_index"]: page for page in document_2.get("pages", [])}
        indices_1 = set(pages_1)
        indices_2 = set(pages_2)
        if indices_1 != indices_2:
            page_mismatches.append(
                {
                    "paper_id": paper_id,
                    "only_annotator_1": sorted(indices_1 - indices_2),
                    "only_annotator_2": sorted(indices_2 - indices_1),
                }
            )

        for page_index in sorted(indices_1 | indices_2):
            evaluated_pages += 1
            blocks_1 = pages_1.get(page_index, {}).get("blocks", [])
            blocks_2 = pages_2.get(page_index, {}).get("blocks", [])
            labels = {
                normalize_label(block.get("label", "Text"))
                for block in blocks_1 + blocks_2
            }

            for label in labels:
                label_blocks_1 = [
                    block
                    for block in blocks_1
                    if normalize_label(block.get("label", "Text")) == label
                ]
                label_blocks_2 = [
                    block
                    for block in blocks_2
                    if normalize_label(block.get("label", "Text")) == label
                ]
                metric = totals[label]
                metric["annotator_1_blocks"] += len(label_blocks_1)
                metric["annotator_2_blocks"] += len(label_blocks_2)
                metric["matched"] += len(match_blocks(label_blocks_1, label_blocks_2))

                page_result = evaluate_page(
                    {"blocks": label_blocks_1}, {"blocks": label_blocks_2}
                ).get(label, {})
                metric["iou_scores"].extend(page_result.get("iou_scores", []))
                metric["content_similarities"].extend(
                    page_result.get("norm_edit_similarities", [])
                )
                metric["cdm_scores"].extend(page_result.get("cdm_scores", []))
                metric["teds_scores"].extend(page_result.get("teds_scores", []))

        print(f"[{position}/{len(common_papers)}] Paper {paper_id} done")

    by_label = {}
    all_ious = []
    for label, metric in sorted(totals.items()):
        count_1 = metric["annotator_1_blocks"]
        count_2 = metric["annotator_2_blocks"]
        matched = metric["matched"]
        precision = matched / count_2 if count_2 else 0.0
        recall = matched / count_1 if count_1 else 0.0
        f1 = 2 * matched / (count_1 + count_2) if count_1 + count_2 else 0.0
        iou_scores = metric["iou_scores"]
        all_ious.extend(iou_scores)

        result = {
            "annotator_1_blocks": count_1,
            "annotator_2_blocks": count_2,
            "matched_blocks": matched,
            "precision_treating_annotator_1_as_reference": precision,
            "recall_treating_annotator_1_as_reference": recall,
            "detection_f1": f1,
            "mean_iou": mean(iou_scores) if iou_scores else 0.0,
        }
        if metric["content_similarities"]:
            result["normalized_edit_similarity"] = mean(
                metric["content_similarities"]
            )
        if metric["cdm_scores"]:
            result["cdm"] = mean(metric["cdm_scores"])
        if metric["teds_scores"]:
            result["teds"] = mean(metric["teds_scores"])
        by_label[label] = result

    detection_labels = list(by_label.values())
    recognition_scores = [
        by_label.get("Text", {}).get("normalized_edit_similarity", 0.0),
        by_label.get("Formula", {}).get("cdm", 0.0),
        by_label.get("Table", {}).get("teds", 0.0),
    ]
    return {
        "scope": {
            "annotator_1_dir": str(annotator_1_dir),
            "annotator_2_dir": str(annotator_2_dir),
            "common_papers": len(common_papers),
            "evaluated_pages": evaluated_pages,
            "papers_only_annotator_1": sorted(papers_1 - papers_2),
            "papers_only_annotator_2": sorted(papers_2 - papers_1),
            "page_mismatches": page_mismatches,
        },
        "by_label": by_label,
        "summary": {
            "detection_macro_f1": mean(
                result["detection_f1"] for result in detection_labels
            ) if detection_labels else 0.0,
            "mean_iou": mean(all_ious) if all_ious else 0.0,
            "recognition_overall": mean(recognition_scores),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate agreement between two OCR annotators."
    )
    parser.add_argument(
        "--annotator-1", type=Path, default=Path("data/gt_agreement_10pct")
    )
    parser.add_argument(
        "--annotator-2", type=Path, default=Path("data/gt_agreement_10pct_hu2")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("result/evaluation/annotation_agreement.json"),
    )
    args = parser.parse_args()

    os.environ.setdefault("CDM_SAVE_VIS", "0")
    result = calculate_agreement(args.annotator_1, args.annotator_2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")

    summary = result["summary"]
    print(f"Detection macro F1: {summary['detection_macro_f1']:.4f}")
    print(f"Mean IoU: {summary['mean_iou']:.4f}")
    print(f"Recognition overall: {summary['recognition_overall']:.4f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
