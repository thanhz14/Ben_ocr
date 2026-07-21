import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import sys

# Import metrics from metric folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'metric'))
from cdm_metric import CDM
from table_metric import TEDS
from rapidfuzz.distance import Levenshtein


@dataclass
class BlockMetrics:
    """Metrics for a single block"""
    id: str
    label: str
    iou: float
    edit_distance: float = None
    edit_similarity: float = None
    cdm_score: float = None
    teds_score: float = None


@dataclass
class PageMetrics:
    """Aggregated metrics for a page"""
    page_index: int
    label: str
    matched_blocks: int
    total_gt_blocks: int
    precision: float
    recall: float
    f1_score: float
    norm_edit_similarity_avg: float = None
    cdm_avg: float = None
    teds_avg: float = None


@dataclass
class PaperMetrics:
    """Aggregated metrics for a paper"""
    paper_name: str
    label: str
    page_metrics: List[PageMetrics]
    precision: float
    recall: float
    f1_score: float
    norm_edit_similarity_avg: float = None
    cdm_avg: float = None
    teds_avg: float = None


# -----------------------------
# LABEL NORMALIZATION
# -----------------------------
def normalize_label(label: str) -> str:
    s = (label or "Text").strip().lower()
    if s in {"text", "txt", "paragraph", "title", "caption"}:
        return "Text"
    if s in {"formula", "equation", "math", "latex"}:
        return "Formula"
    if s in {"table", "tab"}:
        return "Table"
    return label if label else "Text"


def calculate_iou(box1: Dict, box2: Dict) -> float:
    """
    Calculate IoU between two boxes in format:
    {x, y, width, height} (normalized or absolute)
    """
    x1_min, y1_min = box1['x'], box1['y']
    x1_max = x1_min + box1['width']
    y1_max = y1_min + box1['height']

    x2_min, y2_min = box2['x'], box2['y']
    x2_max = x2_min + box2['width']
    y2_max = y2_min + box2['height']

    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    box1_area = box1['width'] * box1['height']
    box2_area = box2['width'] * box2['height']
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def match_blocks(gt_blocks: List[Dict], pred_blocks: List[Dict], iou_threshold: float = 0.5) -> List[Tuple]:
    """
    Match GT blocks with predicted blocks using highest IoU (greedy)
    Return list of (gt_idx, pred_idx, iou)
    """
    matches = []
    matched_pred = set()

    iou_matrix = []
    for gt_block in gt_blocks:
        row = []
        for pred_block in pred_blocks:
            iou = calculate_iou(gt_block['bounding_box'], pred_block['bounding_box'])
            row.append(iou)
        iou_matrix.append(row)

    for gt_idx in range(len(gt_blocks)):
        best_pred_idx = -1
        best_iou = iou_threshold

        for pred_idx in range(len(pred_blocks)):
            if pred_idx not in matched_pred and iou_matrix[gt_idx][pred_idx] > best_iou:
                best_iou = iou_matrix[gt_idx][pred_idx]
                best_pred_idx = pred_idx

        if best_pred_idx != -1:
            matches.append((gt_idx, best_pred_idx, best_iou))
            matched_pred.add(best_pred_idx)

    return matches


def calculate_edit_distance(gt_text: str, pred_text: str) -> Tuple[float, float]:
    """
    Returns:
      distance: raw edit distance (0..+inf)
      similarity: normalized similarity (0..1)
    """
    distance = Levenshtein.distance(gt_text, pred_text)
    similarity = Levenshtein.normalized_similarity(gt_text, pred_text)
    return distance, similarity


def calculate_cdm_score(gt_formula: str, pred_formula: str) -> float:
    try:
        cal_cdm = CDM(output_root='result/evaluation/CDM')
        metrics = cal_cdm.evaluate(gt_formula, pred_formula, 'eval_formula')
        return float(metrics.get('F1_score', 0.0))
    except Exception as e:
        print(f'⚠️  CDM calculation error: {e}')
        return 0.0


def calculate_teds_score(gt_html: str, pred_html: str, structure_only: bool = False) -> float:
    try:
        teds = TEDS(structure_only=structure_only)
        score = teds.evaluate(pred_html, gt_html)
        return float(score) if score is not None else 0.0
    except Exception as e:
        print(f'⚠️  TEDS calculation error: {e}')
        return 0.0


def calculate_object_detection_metrics(
    gt_blocks: List[Dict],
    pred_blocks: List[Dict],
    iou_threshold: float = 0.5
) -> Tuple[float, float, float]:
    matches = match_blocks(gt_blocks, pred_blocks, iou_threshold)

    tp = len(matches)
    fp = len(pred_blocks) - tp
    fn = len(gt_blocks) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def load_json_file(file_path: str) -> Dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_page(gt_page: Dict, pred_page: Dict) -> Dict:
    gt_blocks = gt_page.get('blocks', [])
    pred_blocks = pred_page.get('blocks', [])

    if not gt_blocks or not pred_blocks:
        return {}

    matches = match_blocks(gt_blocks, pred_blocks)

    metrics_by_label = defaultdict(lambda: {
        'matched': 0,
        'total_gt': 0,
        'total_pred': 0,
        # NOTE: lưu normalized similarity [0,1], không phải raw distance
        'norm_edit_similarities': [],
        'cdm_scores': [],
        'teds_scores': [],
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0
    })

    # Count by normalized label
    for block in gt_blocks:
        label = normalize_label(block.get('label', 'Text'))
        metrics_by_label[label]['total_gt'] += 1

    for block in pred_blocks:
        label = normalize_label(block.get('label', 'Text'))
        metrics_by_label[label]['total_pred'] += 1

    # Process matched pairs
    for gt_idx, pred_idx, _ in matches:
        gt_block = gt_blocks[gt_idx]
        pred_block = pred_blocks[pred_idx]

        gt_label = normalize_label(gt_block.get('label', 'Text'))
        pred_label = normalize_label(pred_block.get('label', 'Text'))

        # chỉ tính khi label khớp
        if gt_label != pred_label:
            continue

        metrics_by_label[gt_label]['matched'] += 1

        gt_content = gt_block.get('content') or gt_block.get('block_content', '')
        pred_content = pred_block.get('content') or pred_block.get('block_content', '')

        if not gt_content or not pred_content:
            continue

        # Text: norm edit similarity
        if gt_label == 'Text':
            try:
                _, similarity = calculate_edit_distance(gt_content, pred_content)
                metrics_by_label[gt_label]['norm_edit_similarities'].append(float(similarity))
            except Exception as e:
                print(f'⚠️  Error calculating edit similarity for Text: {e}')

        # Formula: norm edit similarity + CDM
        elif gt_label == 'Formula':
            try:
                _, similarity = calculate_edit_distance(gt_content, pred_content)
                metrics_by_label[gt_label]['norm_edit_similarities'].append(float(similarity))

                cdm_score = calculate_cdm_score(gt_content, pred_content)
                metrics_by_label[gt_label]['cdm_scores'].append(float(cdm_score))
            except Exception as e:
                print(f'⚠️  Error calculating metrics for Formula: {e}')

        # Table: norm edit similarity + TEDS
        elif gt_label == 'Table':
            try:
                _, similarity = calculate_edit_distance(gt_content, pred_content)
                metrics_by_label[gt_label]['norm_edit_similarities'].append(float(similarity))

                teds_score = calculate_teds_score(gt_content, pred_content)
                metrics_by_label[gt_label]['teds_scores'].append(float(teds_score))
            except Exception as e:
                print(f'⚠️  Error calculating metrics for Table: {e}')

    # Detection metrics by label
    labels = list(metrics_by_label.keys())
    for label in labels:
        gt_blocks_label = [b for b in gt_blocks if normalize_label(b.get('label', 'Text')) == label]
        pred_blocks_label = [b for b in pred_blocks if normalize_label(b.get('label', 'Text')) == label]

        if gt_blocks_label or pred_blocks_label:
            precision, recall, f1 = calculate_object_detection_metrics(gt_blocks_label, pred_blocks_label)
            metrics_by_label[label]['precision'] = float(precision)
            metrics_by_label[label]['recall'] = float(recall)
            metrics_by_label[label]['f1'] = float(f1)

        if metrics_by_label[label]['norm_edit_similarities']:
            metrics_by_label[label]['norm_edit_similarity_avg'] = float(
                np.mean(metrics_by_label[label]['norm_edit_similarities'])
            )
        if metrics_by_label[label]['cdm_scores']:
            metrics_by_label[label]['cdm_avg'] = float(np.mean(metrics_by_label[label]['cdm_scores']))
        if metrics_by_label[label]['teds_scores']:
            metrics_by_label[label]['teds_avg'] = float(np.mean(metrics_by_label[label]['teds_scores']))

    return dict(metrics_by_label)


def evaluate_paper(gt_file: str, pred_file: str, paper_name: str) -> Dict:
    gt_data = load_json_file(gt_file)
    pred_data = load_json_file(pred_file)

    gt_pages = {p['page_index']: p for p in gt_data['pages']}
    pred_pages = {p['page_index']: p for p in pred_data['pages']}

    page_results = defaultdict(list)

    for page_idx in sorted(set(list(gt_pages.keys()) + list(pred_pages.keys()))):
        gt_page = gt_pages.get(page_idx, {'blocks': []})
        pred_page = pred_pages.get(page_idx, {'blocks': []})

        page_metrics = evaluate_page(gt_page, pred_page)
        for label, metrics in page_metrics.items():
            page_results[label].append(metrics)

    paper_results = {
        'paper_name': paper_name,
        'pages': len(gt_pages),
        'labels': {}
    }

    for label, page_metrics_list in page_results.items():
        if not page_metrics_list:
            continue

        avg_metrics = {
            'precision': float(np.mean([m['precision'] for m in page_metrics_list])),
            'recall': float(np.mean([m['recall'] for m in page_metrics_list])),
            'f1': float(np.mean([m['f1'] for m in page_metrics_list])),
            'total_gt': int(sum([m['total_gt'] for m in page_metrics_list])),
            'matched': int(sum([m['matched'] for m in page_metrics_list])),
        }

        all_norm_sims = []
        for m in page_metrics_list:
            all_norm_sims.extend(m.get('norm_edit_similarities', []))
        if all_norm_sims:
            avg_metrics['norm_edit_similarity_avg'] = float(np.mean(all_norm_sims))

        all_cdm = []
        for m in page_metrics_list:
            all_cdm.extend(m.get('cdm_scores', []))
        if all_cdm:
            avg_metrics['cdm_avg'] = float(np.mean(all_cdm))

        all_teds = []
        for m in page_metrics_list:
            all_teds.extend(m.get('teds_scores', []))
        if all_teds:
            avg_metrics['teds_avg'] = float(np.mean(all_teds))

        paper_results['labels'][label] = avg_metrics

    return paper_results


def calculate_overall_score(paper_results: Dict) -> float:
    """
    Overall = trung bình các component available:
      - Text norm_edit_similarity * 100
      - Table TEDS * 100
      - Formula CDM * 100
    """
    components = []

    if 'Text' in paper_results['labels']:
        text_metrics = paper_results['labels']['Text']
        if 'norm_edit_similarity_avg' in text_metrics:
            components.append(text_metrics['norm_edit_similarity_avg'] * 100.0)

    if 'Table' in paper_results['labels']:
        table_metrics = paper_results['labels']['Table']
        if 'teds_avg' in table_metrics:
            components.append(table_metrics['teds_avg'] * 100.0)

    if 'Formula' in paper_results['labels']:
        formula_metrics = paper_results['labels']['Formula']
        if 'cdm_avg' in formula_metrics:
            components.append(formula_metrics['cdm_avg'] * 100.0)

    if not components:
        return 0.0
    return float(sum(components) / len(components))


def print_results(all_results: Dict):
    print('\n' + '=' * 100)
    print('EVALUATION RESULTS')
    print('=' * 100 + '\n')

    for paper_name, paper_results in all_results.items():
        print(f'📄 Paper: {paper_name}')
        print(f'   Pages: {paper_results["pages"]}\n')

        for label, metrics in paper_results['labels'].items():
            print(f'   {label}:')
            print(f'      Precision: {metrics["precision"]:.4f}')
            print(f'      Recall: {metrics["recall"]:.4f}')
            print(f'      F1: {metrics["f1"]:.4f}')
            print(f'      Matched/Total: {metrics["matched"]}/{metrics["total_gt"]}')

            if 'norm_edit_similarity_avg' in metrics:
                print(f'      Norm Edit Similarity Avg: {metrics["norm_edit_similarity_avg"]:.4f}')
            if 'cdm_avg' in metrics:
                print(f'      CDM Avg: {metrics["cdm_avg"]:.4f}')
            if 'teds_avg' in metrics:
                print(f'      TEDS Avg: {metrics["teds_avg"]:.4f}')
            print()

        overall_score = calculate_overall_score(paper_results)
        print(f'   🎯 Overall Score: {overall_score:.4f}\n')


def main():
    gt_dir = 'data/gt'
    pred_dir = 'data/pred'

    os.makedirs('result/evaluation', exist_ok=True)

    gt_path = Path(gt_dir)
    gt_files = sorted(gt_path.glob('gt_paper_*.json'))

    if not gt_files:
        print(f'⚠️  No GT files found in {gt_dir}')
        return

    print(f'📂 Found {len(gt_files)} papers to evaluate\n')

    all_results = {}

    for gt_file in gt_files:
        paper_name = gt_file.stem.replace('gt_', '')  # e.g., paper_0
        pred_file = Path(pred_dir) / f'pred_{paper_name}.json'

        if not pred_file.exists():
            print(f'⚠️  Prediction file not found: {pred_file}')
            continue

        try:
            print(f'⏳ Evaluating {paper_name}...')
            results = evaluate_paper(str(gt_file), str(pred_file), paper_name)
            all_results[paper_name] = results
            print('   ✅ Done')
        except Exception as e:
            print(f'   ❌ Error: {e}')

    print_results(all_results)

    output_file = 'result/evaluation/evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f'✅ Results saved to {output_file}')


if __name__ == '__main__':
    main()