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
    iou_avg: float = None
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
    iou_avg: float = None
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
    if s in {"table", "tab", "*"}:
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
        # IoU của các cặp block được ghép và có cùng nhãn.
        'iou_scores': [],
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
    for gt_idx, pred_idx, matched_iou in matches:
        gt_block = gt_blocks[gt_idx]
        pred_block = pred_blocks[pred_idx]

        gt_label = normalize_label(gt_block.get('label', 'Text'))
        pred_label = normalize_label(pred_block.get('label', 'Text'))

        # chỉ tính khi label khớp
        if gt_label != pred_label:
            continue

        metrics_by_label[gt_label]['matched'] += 1
        metrics_by_label[gt_label]['iou_scores'].append(float(matched_iou))

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
        if metrics_by_label[label]['iou_scores']:
            metrics_by_label[label]['iou_avg'] = float(
                np.mean(metrics_by_label[label]['iou_scores'])
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

        # Trung bình theo block (micro average), không phải trung bình các trang.
        all_ious = []
        for m in page_metrics_list:
            all_ious.extend(m.get('iou_scores', []))
        if all_ious:
            avg_metrics['iou_avg'] = float(np.mean(all_ious))
            # Giữ các IoU block để tính trung bình toàn dataset chính xác.
            avg_metrics['iou_scores'] = all_ious

        all_norm_sims = []
        for m in page_metrics_list:
            all_norm_sims.extend(m.get('norm_edit_similarities', []))
        if all_norm_sims:
            avg_metrics['norm_edit_similarity_avg'] = float(np.mean(all_norm_sims))
            avg_metrics['norm_edit_similarity_count'] = len(all_norm_sims)

        all_cdm = []
        for m in page_metrics_list:
            all_cdm.extend(m.get('cdm_scores', []))
        if all_cdm:
            avg_metrics['cdm_avg'] = float(np.mean(all_cdm))
            avg_metrics['cdm_count'] = len(all_cdm)

        all_teds = []
        for m in page_metrics_list:
            all_teds.extend(m.get('teds_scores', []))
        if all_teds:
            avg_metrics['teds_avg'] = float(np.mean(all_teds))
            avg_metrics['teds_count'] = len(all_teds)

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


def weighted_metric_average(
    all_results: Dict,
    label: str,
    average_key: str,
    count_key: str,
) -> float:
    """Tính micro-average theo số block thực sự có điểm."""
    weighted_sum = 0.0
    total_count = 0

    for paper_result in all_results.values():
        metrics = paper_result['labels'].get(label, {})
        if average_key not in metrics:
            continue

        count = int(metrics.get(count_key, 0))
        weighted_sum += float(metrics[average_key]) * count
        total_count += count

    return weighted_sum / total_count if total_count else 0.0


def main():
    gt_dir = 'data/gt'
    pred_dir = 'data/deepseek'
    output_file = 'result/evaluation/evaluation_results.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    gt_files = sorted(Path(gt_dir).glob('gt_*.json'))

    if not gt_files:
        print(f'⚠️  No GT files found in {gt_dir}')
        return

    # Accumulate raw counts/scores over the whole dataset. No paper-level
    # averaging is used, so a small paper cannot have the same weight as a
    # large paper.
    totals = defaultdict(lambda: {
        'total_gt': 0,
        'total_pred': 0,
        'matched': 0,
        'iou_scores': [],
        'norm_edit_similarities': [],
        'cdm_scores': [],
        'teds_scores': [],
    })
    evaluated_papers = 0
    evaluated_pages = 0
    missing_prediction_files = []
    errors = []

    for paper_pos, gt_file in enumerate(gt_files, start=1):
        paper_num = gt_file.stem.replace('gt_', '').replace('paper_', '')
        pred_file = Path(pred_dir) / f'pred_{paper_num}.json'
        try:
            gt_data = load_json_file(str(gt_file))
            if pred_file.exists():
                pred_data = load_json_file(str(pred_file))
                evaluated_papers += 1
            else:
                pred_data = {'pages': []}
                missing_prediction_files.append(str(pred_file))

            gt_pages = {p['page_index']: p for p in gt_data.get('pages', [])}
            pred_pages = {p['page_index']: p for p in pred_data.get('pages', [])}
            page_indices = sorted(set(gt_pages) | set(pred_pages))

            for page_pos, page_idx in enumerate(page_indices, start=1):
                evaluated_pages += 1
                gt_page = gt_pages.get(page_idx, {'blocks': []})
                pred_page = pred_pages.get(page_idx, {'blocks': []})
                gt_blocks = gt_page.get('blocks', [])
                pred_blocks = pred_page.get('blocks', [])

                labels = {
                    normalize_label(b.get('label', 'Text'))
                    for b in gt_blocks + pred_blocks
                }
                for label in sorted(labels):
                    progress = (
                        f'[{paper_pos}/{len(gt_files)}] paper {paper_num} | '
                        f'page {page_pos}/{len(page_indices)} (index {page_idx}) | '
                        f'{label}'
                    )
                    print(f'\r{progress:<100}', end='', file=sys.stderr, flush=True)
                    gt_label = [b for b in gt_blocks
                                if normalize_label(b.get('label', 'Text')) == label]
                    pred_label = [b for b in pred_blocks
                                  if normalize_label(b.get('label', 'Text')) == label]
                    totals[label]['total_gt'] += len(gt_label)
                    totals[label]['total_pred'] += len(pred_label)
                    totals[label]['matched'] += len(match_blocks(gt_label, pred_label))
                    # Recognition scores and IoUs intentionally include matched
                    # blocks only. Misses are represented by P/R/F1.
                    label_result = evaluate_page(
                        {'blocks': gt_label}, {'blocks': pred_label}
                    ).get(label, {})
                    totals[label]['iou_scores'].extend(label_result.get('iou_scores', []))
                    totals[label]['norm_edit_similarities'].extend(
                        label_result.get('norm_edit_similarities', [])
                    )
                    totals[label]['cdm_scores'].extend(label_result.get('cdm_scores', []))
                    totals[label]['teds_scores'].extend(label_result.get('teds_scores', []))
        except Exception as e:
            errors.append({'file': str(gt_file), 'error': f'{type(e).__name__}: {e}'})

    # Finish the in-place progress line before printing final metrics.
    print(file=sys.stderr, flush=True)

    by_label = {}
    all_iou_scores = []
    for label, metric in sorted(totals.items()):
        tp = int(metric['matched'])
        total_gt = int(metric['total_gt'])
        total_pred = int(metric['total_pred'])
        precision = tp / total_pred if total_pred else 0.0
        recall = tp / total_gt if total_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        iou_scores = metric['iou_scores']
        all_iou_scores.extend(iou_scores)

        label_result = {
            'total_gt': total_gt,
            'total_pred': total_pred,
            'matched': tp,
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'miou': float(np.mean(iou_scores)) if iou_scores else 0.0,
            'matched_with_content': len(metric['norm_edit_similarities']),
        }
        if metric['norm_edit_similarities']:
            label_result['ned'] = float(np.mean(metric['norm_edit_similarities']))
        if metric['cdm_scores']:
            label_result['cdm'] = float(np.mean(metric['cdm_scores']))
            label_result['cdm_count'] = len(metric['cdm_scores'])
        if metric['teds_scores']:
            label_result['teds'] = float(np.mean(metric['teds_scores']))
            label_result['teds_count'] = len(metric['teds_scores'])
        by_label[label] = label_result

    ned = by_label.get('Text', {}).get('ned', 0.0)
    cdm = by_label.get('Formula', {}).get('cdm', 0.0)
    teds = by_label.get('Table', {}).get('teds', 0.0)
    overall = float(np.mean([ned, cdm, teds]))
    miou = float(np.mean(all_iou_scores)) if all_iou_scores else 0.0
    detection_labels = [m for m in by_label.values()
                        if m['total_gt'] or m['total_pred']]
    detection_macro = {
        key: float(np.mean([m[key] for m in detection_labels]))
        if detection_labels else 0.0
        for key in ('precision', 'recall', 'f1', 'miou')
    }

    output_data = {
        'evaluation_scope': {
            'gt_dir': gt_dir,
            'pred_dir': pred_dir,
            'gt_papers': len(gt_files),
            'evaluated_papers': evaluated_papers,
            'evaluated_pages': evaluated_pages,
            'missing_prediction_files': missing_prediction_files,
            'errors': errors,
        },
        'by_label': by_label,
        'dataset_summary': {
            'ned': ned,
            'cdm': cdm,
            'teds': teds,
            'overall': overall,
            'miou': miou,
            'matched_blocks_with_iou': len(all_iou_scores),
            'detection_macro_by_label': detection_macro,
        },
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f'NED: {ned:.4f}')
    print(f'CDM: {cdm:.4f}')
    print(f'TEDS: {teds:.4f}')
    print(f'Overall: {overall:.4f}')
    print(f'mIoU: {miou:.4f}')

if __name__ == '__main__':
    main()
