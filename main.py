import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
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
    edit_distance_avg: float = None
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
    edit_distance_avg: float = None
    cdm_avg: float = None
    teds_avg: float = None


def calculate_iou(box1: Dict, box2: Dict) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: {x, y, width, height} in [0,1] range
        box2: {x, y, width, height} in [0,1] range
    
    Returns:
        float: IoU score [0,1]
    """
    # Convert to [x1, y1, x2, y2] format
    x1_min, y1_min = box1['x'], box1['y']
    x1_max = x1_min + box1['width']
    y1_max = y1_min + box1['height']
    
    x2_min, y2_min = box2['x'], box2['y']
    x2_max = x2_min + box2['width']
    y2_max = y2_min + box2['height']
    
    # Calculate intersection
    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    # Calculate union
    box1_area = box1['width'] * box1['height']
    box2_area = box2['width'] * box2['height']
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def match_blocks(gt_blocks: List[Dict], pred_blocks: List[Dict], iou_threshold: float = 0.5) -> List[Tuple]:
    """
    Match GT blocks with predicted blocks using highest IoU
    
    Args:
        gt_blocks: List of GT blocks
        pred_blocks: List of predicted blocks
        iou_threshold: Minimum IoU to consider as match
    
    Returns:
        List of (gt_idx, pred_idx, iou) tuples
    """
    matches = []
    matched_gt = set()
    matched_pred = set()
    
    # Calculate IoU matrix
    iou_matrix = []
    for gt_idx, gt_block in enumerate(gt_blocks):
        row = []
        for pred_idx, pred_block in enumerate(pred_blocks):
            iou = calculate_iou(gt_block['bounding_box'], pred_block['bounding_box'])
            row.append(iou)
        iou_matrix.append(row)
    
    # Greedy matching: find highest IoU pairs
    for gt_idx in range(len(gt_blocks)):
        best_pred_idx = -1
        best_iou = iou_threshold
        
        for pred_idx in range(len(pred_blocks)):
            if pred_idx not in matched_pred and iou_matrix[gt_idx][pred_idx] > best_iou:
                best_iou = iou_matrix[gt_idx][pred_idx]
                best_pred_idx = pred_idx
        
        if best_pred_idx != -1:
            matches.append((gt_idx, best_pred_idx, best_iou))
            matched_gt.add(gt_idx)
            matched_pred.add(best_pred_idx)
    
    return matches


def calculate_edit_distance(gt_text: str, pred_text: str) -> Tuple[float, float]:
    """
    Calculate edit distance and similarity
    
    Args:
        gt_text: Ground truth text
        pred_text: Predicted text
    
    Returns:
        Tuple: (edit_distance, edit_similarity)
    """
    distance = Levenshtein.distance(gt_text, pred_text)
    similarity = Levenshtein.normalized_similarity(gt_text, pred_text)
    return distance, similarity


def calculate_cdm_score(gt_formula: str, pred_formula: str) -> float:
    """
    Calculate CDM score for formula
    
    Args:
        gt_formula: Ground truth formula
        pred_formula: Predicted formula
    
    Returns:
        float: CDM F1 score
    """
    try:
        cal_cdm = CDM(output_root='result/evaluation/CDM')
        metrics = cal_cdm.evaluate(gt_formula, pred_formula, 'eval_formula')
        return metrics.get('F1_score', 0.0)
    except Exception as e:
        print(f'⚠️  CDM calculation error: {e}')
        return 0.0


def calculate_teds_score(gt_html: str, pred_html: str, structure_only: bool = False) -> float:
    """
    Calculate TEDS score for table
    
    Args:
        gt_html: Ground truth table HTML
        pred_html: Predicted table HTML
        structure_only: Only compare structure
    
    Returns:
        float: TEDS score
    """
    try:
        teds = TEDS(structure_only=structure_only)
        score = teds.evaluate(pred_html, gt_html)
        return score if score is not None else 0.0
    except Exception as e:
        print(f'⚠️  TEDS calculation error: {e}')
        return 0.0


def calculate_object_detection_metrics(gt_blocks: List[Dict], pred_blocks: List[Dict], 
                                       iou_threshold: float = 0.5) -> Tuple[float, float, float]:
    """
    Calculate object detection metrics (Precision, Recall, F1)
    
    Args:
        gt_blocks: Ground truth blocks
        pred_blocks: Predicted blocks
        iou_threshold: IoU threshold for positive match
    
    Returns:
        Tuple: (precision, recall, f1_score)
    """
    matches = match_blocks(gt_blocks, pred_blocks, iou_threshold)
    
    tp = len(matches)
    fp = len(pred_blocks) - tp
    fn = len(gt_blocks) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def load_json_file(file_path: str) -> Dict:
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_page(gt_page: Dict, pred_page: Dict, debug: bool = False) -> Dict:
    """
    Evaluate a single page
    
    Args:
        gt_page: GT page data
        pred_page: Predicted page data
        debug: Print debug info
    
    Returns:
        dict: Metrics for each label
    """
    gt_blocks = gt_page.get('blocks', [])
    pred_blocks = pred_page.get('blocks', [])
    
    if not gt_blocks or not pred_blocks:
        return {}
    
    if debug:
        print(f'  Debug: GT blocks: {len(gt_blocks)}, Pred blocks: {len(pred_blocks)}')
        for i, b in enumerate(gt_blocks[:2]):
            print(f'    GT[{i}]: label={b.get("label")}, bbox={b.get("bounding_box")}')
        for i, b in enumerate(pred_blocks[:2]):
            print(f'    Pred[{i}]: label={b.get("label")}, bbox={b.get("bounding_box")}')
    
    # Match blocks with lower threshold
    matches = match_blocks(gt_blocks, pred_blocks, iou_threshold=0.1)
    
    if debug:
        print(f'  Debug: Matches found: {len(matches)}')
        for gt_idx, pred_idx, iou in matches[:3]:
            gt_label = gt_blocks[gt_idx].get('label')
            pred_label = pred_blocks[pred_idx].get('label')
            print(f'    Match: GT[{gt_idx}]({gt_label}) - Pred[{pred_idx}]({pred_label}), IoU={iou:.4f}')
    
    # Organize blocks by label
    metrics_by_label = defaultdict(lambda: {
        'matched': 0,
        'total_gt': 0,
        'total_pred': 0,
        'edit_distances': [],
        'cdm_scores': [],
        'teds_scores': [],
        'precision': 0,
        'recall': 0,
        'f1': 0
    })
    
    # Initialize label counts
    for block in gt_blocks:
        label = block.get('label', 'Text')
        metrics_by_label[label]['total_gt'] += 1
    
    for block in pred_blocks:
        label = block.get('label', 'Text')
        metrics_by_label[label]['total_pred'] += 1
    
    # Process matches
    matched_gt_set = set()
    matched_pred_set = set()
    
    for gt_idx, pred_idx, iou in matches:
        gt_block = gt_blocks[gt_idx]
        pred_block = pred_blocks[pred_idx]
        
        gt_label = gt_block.get('label', 'Text')
        pred_label = pred_block.get('label', 'Text')
        
        # Check label match
        if gt_label != pred_label:
            continue
        
        metrics_by_label[gt_label]['matched'] += 1
        matched_gt_set.add(gt_idx)
        matched_pred_set.add(pred_idx)
        
        # Get content safely
        gt_content = gt_block.get('content') or gt_block.get('block_content', '')
        pred_content = pred_block.get('content') or pred_block.get('block_content', '')
        
        if not gt_content or not pred_content:
            continue
        
        # Calculate content-based metrics
        if gt_label == 'Text':
            try:
                distance, similarity = calculate_edit_distance(gt_content, pred_content)
                metrics_by_label[gt_label]['edit_distances'].append(distance)
            except Exception as e:
                print(f'⚠️  Error calculating edit distance for Text: {e}')
        
        elif gt_label == 'Formula':
            try:
                distance, similarity = calculate_edit_distance(gt_content, pred_content)
                metrics_by_label[gt_label]['edit_distances'].append(distance)
                
                cdm_score = calculate_cdm_score(gt_content, pred_content)
                metrics_by_label[gt_label]['cdm_scores'].append(cdm_score)
            except Exception as e:
                print(f'⚠️  Error calculating metrics for Formula: {e}')
        
        elif gt_label == 'Table':
            try:
                distance, similarity = calculate_edit_distance(gt_content, pred_content)
                metrics_by_label[gt_label]['edit_distances'].append(distance)
                
                teds_score = calculate_teds_score(gt_content, pred_content)
                metrics_by_label[gt_label]['teds_scores'].append(teds_score)
            except Exception as e:
                print(f'⚠️  Error calculating metrics for Table: {e}')
    
    # Calculate metrics for each label
    for label, metrics in metrics_by_label.items():
        gt_blocks_label = [b for b in gt_blocks if b.get('label', 'Text') == label]
        pred_blocks_label = [b for b in pred_blocks if b.get('label', 'Text') == label]
        
        if gt_blocks_label or pred_blocks_label:
            precision, recall, f1 = calculate_object_detection_metrics(
                gt_blocks_label, pred_blocks_label, iou_threshold=0.1
            )
            
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
        
        # Calculate averages
        if metrics['edit_distances']:
            metrics['edit_distance_avg'] = np.mean(metrics['edit_distances'])
        if metrics['cdm_scores']:
            metrics['cdm_avg'] = np.mean(metrics['cdm_scores'])
        if metrics['teds_scores']:
            metrics['teds_avg'] = np.mean(metrics['teds_scores'])
    
    return dict(metrics_by_label)


def main():
    """Main evaluation function"""
    gt_dir = 'data/gt'
    pred_dir = 'data/pred'
    
    # Create output directory
    os.makedirs('result/evaluation', exist_ok=True)
    
    # Get all GT files
    gt_path = Path(gt_dir)
    gt_files = sorted(gt_path.glob('gt_paper_*.json'))
    
    if not gt_files:
        print(f'⚠️  No GT files found in {gt_dir}')
        return
    
    print(f'📂 Found {len(gt_files)} papers to evaluate')
    print()
    
    all_results = {}
    
    # Evaluate each paper
    for gt_file in gt_files:
        paper_name = gt_file.stem.replace('gt_', '')  # e.g., 'paper_0'
        pred_file = Path(pred_dir) / f'pred_{paper_name}.json'
        
        if not pred_file.exists():
            print(f'⚠️  Prediction file not found: {pred_file}')
            continue
        
        try:
            print(f'⏳ Evaluating {paper_name}...')
            
            # Debug: Load and print first page
            gt_data = load_json_file(str(gt_file))
            pred_data = load_json_file(str(pred_file))
            
            if gt_data['pages'] and pred_data['pages']:
                print(f'  Page 0 Debug:')
                gt_page_0 = gt_data['pages'][0]
                pred_page_0 = pred_data['pages'][0]
                
                evaluate_page(gt_page_0, pred_page_0, debug=True)
            
            results = evaluate_paper(str(gt_file), str(pred_file), paper_name)
            all_results[paper_name] = results
            print(f'   ✅ Done')
        except Exception as e:
            print(f'   ❌ Error: {e}')
            import traceback
            traceback.print_exc()
    
    # Print results
    print_results(all_results)
    
    # Save results to file
    output_file = 'result/evaluation/evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f'✅ Results saved to {output_file}')


def evaluate_paper(gt_file: str, pred_file: str, paper_name: str) -> Dict:
    """
    Evaluate a complete paper
    
    Args:
        gt_file: Path to GT JSON
        pred_file: Path to prediction JSON
        paper_name: Paper name
    
    Returns:
        dict: Complete evaluation results
    """
    gt_data = load_json_file(gt_file)
    pred_data = load_json_file(pred_file)
    
    gt_pages = {p['page_index']: p for p in gt_data['pages']}
    pred_pages = {p['page_index']: p for p in pred_data['pages']}
    
    # Evaluate each page
    page_results = defaultdict(list)  # ✅ Fix: defaultdict(list) thay vì defaultdict(lambda: defaultdict(list))
    
    for page_idx in sorted(set(list(gt_pages.keys()) + list(pred_pages.keys()))):
        gt_page = gt_pages.get(page_idx, {'blocks': []})
        pred_page = pred_pages.get(page_idx, {'blocks': []})
        
        page_metrics = evaluate_page(gt_page, pred_page)
        
        for label, metrics in page_metrics.items():
            page_results[label].append(metrics)
    
    # Aggregate metrics by label
    paper_results = {
        'paper_name': paper_name,
        'pages': len(gt_pages),
        'labels': {}
    }
    
    for label, page_metrics_list in page_results.items():
        if not page_metrics_list:
            continue
        
        # Calculate averages across pages
        avg_metrics = {
            'precision': np.mean([m['precision'] for m in page_metrics_list]),
            'recall': np.mean([m['recall'] for m in page_metrics_list]),
            'f1': np.mean([m['f1'] for m in page_metrics_list]),
            'total_gt': sum([m['total_gt'] for m in page_metrics_list]),
            'matched': sum([m['matched'] for m in page_metrics_list]),
        }
        
        if any([m.get('edit_distances') for m in page_metrics_list]):
            all_distances = []
            for m in page_metrics_list:
                all_distances.extend(m.get('edit_distances', []))
            avg_metrics['edit_distance_avg'] = np.mean(all_distances) if all_distances else None
        
        if any([m.get('cdm_scores') for m in page_metrics_list]):
            all_cdm = []
            for m in page_metrics_list:
                all_cdm.extend(m.get('cdm_scores', []))
            avg_metrics['cdm_avg'] = np.mean(all_cdm) if all_cdm else None
        
        if any([m.get('teds_scores') for m in page_metrics_list]):
            all_teds = []
            for m in page_metrics_list:
                all_teds.extend(m.get('teds_scores', []))
            avg_metrics['teds_avg'] = np.mean(all_teds) if all_teds else None
        
        paper_results['labels'][label] = avg_metrics
    
    return paper_results


def calculate_overall_score(paper_results: Dict) -> float:
    """
    Calculate overall score
    Overall = [(1 - Text Edit Distance) * 100 + Table TEDS + Formula CDM] / 3
    
    Args:
        paper_results: Paper evaluation results
    
    Returns:
        float: Overall score
    """
    components = []
    
    # Text Edit Distance component
    if 'Text' in paper_results['labels']:
        text_metrics = paper_results['labels']['Text']
        if 'edit_distance_avg' in text_metrics and text_metrics['edit_distance_avg'] is not None:
            # Normalize edit distance to [0, 1]
            max_distance = 100  # Assume max distance is 100
            normalized_distance = min(text_metrics['edit_distance_avg'] / max_distance, 1.0)
            text_score = (1 - normalized_distance) * 100
            components.append(text_score)
    
    # Table TEDS component
    if 'Table' in paper_results['labels']:
        table_metrics = paper_results['labels']['Table']
        if 'teds_avg' in table_metrics and table_metrics['teds_avg'] is not None:
            components.append(table_metrics['teds_avg'] * 100)
    
    # Formula CDM component
    if 'Formula' in paper_results['labels']:
        formula_metrics = paper_results['labels']['Formula']
        if 'cdm_avg' in formula_metrics and formula_metrics['cdm_avg'] is not None:
            components.append(formula_metrics['cdm_avg'] * 100)
    
    if not components:
        return 0.0
    
    # Average the available components
    overall = sum(components) / len(components)
    return overall


def print_results(all_results: Dict):
    """Print evaluation results"""
    print('\n' + '='*100)
    print('EVALUATION RESULTS')
    print('='*100 + '\n')
    
    for paper_name, paper_results in all_results.items():
        print(f'📄 Paper: {paper_name}')
        print(f'   Pages: {paper_results["pages"]}')
        print()
        
        for label, metrics in paper_results['labels'].items():
            print(f'   {label}:')
            print(f'      Precision: {metrics["precision"]:.4f}')
            print(f'      Recall: {metrics["recall"]:.4f}')
            print(f'      F1: {metrics["f1"]:.4f}')
            print(f'      Matched/Total: {metrics["matched"]}/{metrics["total_gt"]}')
            
            if 'edit_distance_avg' in metrics and metrics['edit_distance_avg'] is not None:
                print(f'      Edit Distance Avg: {metrics["edit_distance_avg"]:.4f}')
            
            if 'cdm_avg' in metrics and metrics['cdm_avg'] is not None:
                print(f'      CDM Avg: {metrics["cdm_avg"]:.4f}')
            
            if 'teds_avg' in metrics and metrics['teds_avg'] is not None:
                print(f'      TEDS Avg: {metrics["teds_avg"]:.4f}')
            
            print()
        
        overall_score = calculate_overall_score(paper_results)
        print(f'   🎯 Overall Score: {overall_score:.4f}')
        print()


def main():
    """Main evaluation function"""
    gt_dir = 'data/gt'
    pred_dir = 'data/pred'
    
    # Create output directory
    os.makedirs('result/evaluation', exist_ok=True)
    
    # Get all GT files
    gt_path = Path(gt_dir)
    gt_files = sorted(gt_path.glob('gt_paper_*.json'))
    
    if not gt_files:
        print(f'⚠️  No GT files found in {gt_dir}')
        return
    
    print(f'📂 Found {len(gt_files)} papers to evaluate')
    print()
    
    all_results = {}
    
    # Evaluate each paper
    for gt_file in gt_files:
        paper_name = gt_file.stem.replace('gt_', '')  # e.g., 'paper_0'
        pred_file = Path(pred_dir) / f'pred_{paper_name}.json'
        
        if not pred_file.exists():
            print(f'⚠️  Prediction file not found: {pred_file}')
            continue
        
        try:
            print(f'⏳ Evaluating {paper_name}...')
            results = evaluate_paper(str(gt_file), str(pred_file), paper_name)
            all_results[paper_name] = results
            print(f'   ✅ Done')
        except Exception as e:
            print(f'   ❌ Error: {e}')
    
    # Print results
    print_results(all_results)
    
    # Save results to file
    output_file = 'result/evaluation/evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f'✅ Results saved to {output_file}')


if __name__ == '__main__':
    main()