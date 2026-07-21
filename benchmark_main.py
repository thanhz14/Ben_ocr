import argparse
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional

# Bắt buộc dùng metric từ 2 file này theo yêu cầu
from metric.fomula import *  # noqa
from metric.table import *   # noqa


LABEL_TEXT = "text"
LABEL_FORMULA = "formula"
LABEL_TABLE = "table"
TARGET_LABELS = [LABEL_TEXT, LABEL_FORMULA, LABEL_TABLE]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def norm_str(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


def normalized_edit_distance(a: str, b: str) -> float:
    a = norm_str(a)
    b = norm_str(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    dist = dp[m]
    return max(0.0, 1.0 - dist / max(n, m))


# ---------- JSON schema helpers ----------
def get_page_list(doc: Any) -> List[Dict]:
    """
    Hỗ trợ nhiều format:
    - {"pages":[...]}
    - [{"blocks":[...]}...]
    - {"paper":[{"blocks":[...]}...]} ...
    """
    if isinstance(doc, dict):
        if "pages" in doc and isinstance(doc["pages"], list):
            return doc["pages"]

        # fallback tìm list pages-like
        for k, v in doc.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                # nếu phần tử có block-like hoặc page-like key thì nhận
                sample = v[0]
                if any(x in sample for x in ["blocks", "items", "elements", "layouts", "page", "page_id", "lines"]):
                    return v
        return []
    elif isinstance(doc, list):
        return doc
    return []


def get_blocks(page: Dict) -> List[Dict]:
    for k in ["blocks", "items", "elements", "layouts", "lines", "objects", "annotations"]:
        if k in page and isinstance(page[k], list):
            return page[k]
    # fallback: nếu page có trực tiếp danh sách block trong 1 key bất kỳ
    for _, v in page.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            return v
    return []


def try_map_label(raw: Any) -> str:
    s = norm_str(raw).lower()
    # map mềm cho các alias phổ biến
    if s in {"text", "txt", "paragraph", "caption", "title"}:
        return LABEL_TEXT
    if s in {"formula", "equation", "math", "latex"}:
        return LABEL_FORMULA
    if s in {"table", "tab"}:
        return LABEL_TABLE

    # số id thường gặp (tuỳ dataset, sửa nếu cần)
    # ví dụ 1=text, 2=formula, 3=table
    if s in {"1"}:
        return LABEL_TEXT
    if s in {"2"}:
        return LABEL_FORMULA
    if s in {"3"}:
        return LABEL_TABLE

    return ""


def get_label(block: Dict) -> str:
    cand_keys = [
        "label", "type", "category", "class", "block_type", "kind",
        "label_name", "category_name", "name", "cls",
        "label_id", "category_id", "class_id", "type_id"
    ]
    for k in cand_keys:
        if k in block:
            mapped = try_map_label(block[k])
            if mapped:
                return mapped
    return ""


def get_text_content(block: Dict) -> str:
    cand_keys = [
        "text", "content", "value", "latex", "tex", "html", "markdown",
        "md", "transcript", "ocr", "cell_text", "table_html"
    ]
    for k in cand_keys:
        if k in block and block[k] is not None:
            if isinstance(block[k], (str, int, float)):
                return norm_str(block[k])
            if isinstance(block[k], list):
                # join list text
                return " ".join(norm_str(x) for x in block[k] if x is not None)
            if isinstance(block[k], dict):
                return json.dumps(block[k], ensure_ascii=False)
    return ""


def get_bbox(block: Dict) -> Optional[List[float]]:
    # [x1,y1,x2,y2]
    for k in ["bbox", "box", "rect", "xyxy"]:
        if k in block and isinstance(block[k], list) and len(block[k]) == 4:
            x1, y1, x2, y2 = block[k]
            x1, x2 = min(float(x1), float(x2)), max(float(x1), float(x2))
            y1, y2 = min(float(y1), float(y2)), max(float(y1), float(y2))
            return [x1, y1, x2, y2]

    # [x,y,w,h]
    for k in ["xywh", "bbox_xywh"]:
        if k in block and isinstance(block[k], list) and len(block[k]) == 4:
            x, y, w, h = [float(t) for t in block[k]]
            return [x, y, x + max(0.0, w), y + max(0.0, h)]

    # nested dict
    for k in ["position", "bounding_box", "bb"]:
        if k in block and isinstance(block[k], dict):
            d = block[k]
            if all(t in d for t in ["x1", "y1", "x2", "y2"]):
                x1, y1, x2, y2 = float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])
                return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            if all(t in d for t in ["x", "y", "w", "h"]):
                x, y, w, h = float(d["x"]), float(d["y"]), float(d["w"]), float(d["h"])
                return [x, y, x + max(0.0, w), y + max(0.0, h)]

    # polygon/points
    for k in ["polygon", "points", "poly", "vertices"]:
        if k in block and isinstance(block[k], list):
            xs, ys = [], []
            for p in block[k]:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    xs.append(float(p[0]))
                    ys.append(float(p[1]))
                elif isinstance(p, dict) and "x" in p and "y" in p:
                    xs.append(float(p["x"]))
                    ys.append(float(p["y"]))
            if xs and ys:
                return [min(xs), min(ys), max(xs), max(ys)]

    return None


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match_by_max_iou(gt_blocks: List[Dict], pred_blocks: List[Dict]) -> List[Tuple[int, int, float]]:
    cand = []
    for i, g in enumerate(gt_blocks):
        gb = get_bbox(g)
        if gb is None:
            continue
        for j, p in enumerate(pred_blocks):
            pb = get_bbox(p)
            if pb is None:
                continue
            cand.append((iou_xyxy(gb, pb), i, j))
    cand.sort(key=lambda x: x[0], reverse=True)

    used_g, used_p = set(), set()
    out = []
    for v, i, j in cand:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        out.append((i, j, v))
    return out


# ---------- Metric wrappers ----------
def compute_formula_metrics(gt_formula: str, pred_formula: str) -> Tuple[float, float]:
    ned = normalized_edit_distance(gt_formula, pred_formula)
    cdm = 0.0

    # fallback mềm theo tên hàm có thể có trong metric/fomula.py
    for fname in ["cdm_f1", "compute_cdm_f1", "formula_cdm_f1", "f1_cdm"]:
        if fname in globals() and callable(globals()[fname]):
            try:
                cdm = float(globals()[fname](gt_formula, pred_formula))
                break
            except Exception:
                pass

    return ned, max(0.0, min(1.0, cdm))


def compute_table_metrics(gt_table: str, pred_table: str) -> Tuple[float, float]:
    ned = normalized_edit_distance(gt_table, pred_table)
    teds = 0.0

    # fallback mềm theo tên hàm có thể có trong metric/table.py
    for fname in ["teds", "compute_teds", "table_teds", "cal_teds"]:
        if fname in globals() and callable(globals()[fname]):
            try:
                teds = float(globals()[fname](gt_table, pred_table))
                break
            except Exception:
                pass

    return ned, max(0.0, min(1.0, teds))


def safe_avg(s: float, c: int) -> float:
    return s / c if c > 0 else 0.0


def inspect_schema_once(gt_pages: List[Dict], pred_pages: List[Dict], sample_blocks: int = 3):
    print("\n[DEBUG] ===== Schema inspection =====")
    if not gt_pages:
        print("[DEBUG] GT pages empty")
    else:
        print(f"[DEBUG] GT first page keys: {list(gt_pages[0].keys())}")
        gbs = get_blocks(gt_pages[0])
        print(f"[DEBUG] GT first page blocks: {len(gbs)}")
        for i, b in enumerate(gbs[:sample_blocks]):
            print(f"[DEBUG] GT block[{i}] keys: {list(b.keys())}")

    if not pred_pages:
        print("[DEBUG] Pred pages empty")
    else:
        print(f"[DEBUG] Pred first page keys: {list(pred_pages[0].keys())}")
        pbs = get_blocks(pred_pages[0])
        print(f"[DEBUG] Pred first page blocks: {len(pbs)}")
        for i, b in enumerate(pbs[:sample_blocks]):
            print(f"[DEBUG] Pred block[{i}] keys: {list(b.keys())}")
    print("[DEBUG] ==============================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, type=str)
    ap.add_argument("--pred", required=True, type=str)
    ap.add_argument("--out", default="benchmark_result.json", type=str)
    ap.add_argument("--debug", action="store_true", help="In debug schema và thống kê chi tiết")
    ap.add_argument("--iou-thr", default=0.0, type=float, help="Ngưỡng IoU tối thiểu để nhận match")
    args = ap.parse_args()

    gt_doc = load_json(args.gt)
    pred_doc = load_json(args.pred)

    gt_pages = get_page_list(gt_doc)
    pred_pages = get_page_list(pred_doc)
    n_pages = min(len(gt_pages), len(pred_pages))
    if n_pages <= 0:
        raise ValueError("Không tìm thấy pages hợp lệ ở gt/pred.")

    if args.debug:
        inspect_schema_once(gt_pages, pred_pages, sample_blocks=5)

    matrix_sum = {
        LABEL_TEXT: defaultdict(float),
        LABEL_FORMULA: defaultdict(float),
        LABEL_TABLE: defaultdict(float),
    }
    label_counts = {k: 0 for k in TARGET_LABELS}
    page_reports = []

    # debug counters
    dbg = Counter()

    for pi in range(n_pages):
        g_blocks = get_blocks(gt_pages[pi])
        p_blocks = get_blocks(pred_pages[pi])

        dbg["gt_blocks_total"] += len(g_blocks)
        dbg["pred_blocks_total"] += len(p_blocks)

        # precompute label availability
        for b in g_blocks:
            if get_label(b):
                dbg["gt_blocks_has_label"] += 1
            if get_bbox(b) is not None:
                dbg["gt_blocks_has_bbox"] += 1
            if get_text_content(b):
                dbg["gt_blocks_has_text"] += 1

        for b in p_blocks:
            if get_label(b):
                dbg["pred_blocks_has_label"] += 1
            if get_bbox(b) is not None:
                dbg["pred_blocks_has_bbox"] += 1
            if get_text_content(b):
                dbg["pred_blocks_has_text"] += 1

        matches = greedy_match_by_max_iou(g_blocks, p_blocks)

        if args.iou_thr > 0:
            matches = [m for m in matches if m[2] >= args.iou_thr]

        dbg["matches_total"] += len(matches)

        page_stat = {
            "page_index": pi,
            "num_gt_blocks": len(g_blocks),
            "num_pred_blocks": len(p_blocks),
            "num_matches": len(matches),
            "metrics": {
                LABEL_TEXT: {"norm_editdistance_sum": 0.0, "count": 0},
                LABEL_FORMULA: {"norm_editdistance_sum": 0.0, "cdm_f1_sum": 0.0, "count": 0},
                LABEL_TABLE: {"norm_editdistance_sum": 0.0, "teds_sum": 0.0, "count": 0},
            }
        }

        for gi, pj, iou_val in matches:
            g = g_blocks[gi]
            p = p_blocks[pj]

            gl = get_label(g)
            pl = get_label(p)

            if gl not in TARGET_LABELS:
                dbg["skip_non_target_label"] += 1
                continue

            gt_text = get_text_content(g)
            pred_text = get_text_content(p) if pl == gl else ""

            if gl == LABEL_TEXT:
                ned = normalized_edit_distance(gt_text, pred_text)
                matrix_sum[LABEL_TEXT]["norm_editdistance_sum"] += ned
                matrix_sum[LABEL_TEXT]["count"] += 1
                label_counts[LABEL_TEXT] += 1

                page_stat["metrics"][LABEL_TEXT]["norm_editdistance_sum"] += ned
                page_stat["metrics"][LABEL_TEXT]["count"] += 1

            elif gl == LABEL_FORMULA:
                ned, cdm = compute_formula_metrics(gt_text, pred_text)
                matrix_sum[LABEL_FORMULA]["norm_editdistance_sum"] += ned
                matrix_sum[LABEL_FORMULA]["cdm_f1_sum"] += cdm
                matrix_sum[LABEL_FORMULA]["count"] += 1
                label_counts[LABEL_FORMULA] += 1

                page_stat["metrics"][LABEL_FORMULA]["norm_editdistance_sum"] += ned
                page_stat["metrics"][LABEL_FORMULA]["cdm_f1_sum"] += cdm
                page_stat["metrics"][LABEL_FORMULA]["count"] += 1

            elif gl == LABEL_TABLE:
                ned, teds = compute_table_metrics(gt_text, pred_text)
                matrix_sum[LABEL_TABLE]["norm_editdistance_sum"] += ned
                matrix_sum[LABEL_TABLE]["teds_sum"] += teds
                matrix_sum[LABEL_TABLE]["count"] += 1
                label_counts[LABEL_TABLE] += 1

                page_stat["metrics"][LABEL_TABLE]["norm_editdistance_sum"] += ned
                page_stat["metrics"][LABEL_TABLE]["teds_sum"] += teds
                page_stat["metrics"][LABEL_TABLE]["count"] += 1

            dbg["used_match"] += 1
            if iou_val > 0:
                dbg["used_match_iou_gt_0"] += 1

        page_reports.append(page_stat)

    # matrix average
    matrix_avg = {
        LABEL_TEXT: {
            "norm_editdistance": safe_avg(
                matrix_sum[LABEL_TEXT]["norm_editdistance_sum"],
                int(matrix_sum[LABEL_TEXT]["count"])
            )
        },
        LABEL_FORMULA: {
            "norm_editdistance": safe_avg(
                matrix_sum[LABEL_FORMULA]["norm_editdistance_sum"],
                int(matrix_sum[LABEL_FORMULA]["count"])
            ),
            "cdm_f1": safe_avg(
                matrix_sum[LABEL_FORMULA]["cdm_f1_sum"],
                int(matrix_sum[LABEL_FORMULA]["count"])
            )
        },
        LABEL_TABLE: {
            "norm_editdistance": safe_avg(
                matrix_sum[LABEL_TABLE]["norm_editdistance_sum"],
                int(matrix_sum[LABEL_TABLE]["count"])
            ),
            "teds": safe_avg(
                matrix_sum[LABEL_TABLE]["teds_sum"],
                int(matrix_sum[LABEL_TABLE]["count"])
            )
        }
    }

    # final score / num_pages
    page_scores = []
    for pr in page_reports:
        vals = []
        tc = pr["metrics"][LABEL_TEXT]["count"]
        fc = pr["metrics"][LABEL_FORMULA]["count"]
        bc = pr["metrics"][LABEL_TABLE]["count"]

        if tc > 0:
            vals.append(pr["metrics"][LABEL_TEXT]["norm_editdistance_sum"] / tc)
        if fc > 0:
            vals.append(pr["metrics"][LABEL_FORMULA]["norm_editdistance_sum"] / fc)
            vals.append(pr["metrics"][LABEL_FORMULA]["cdm_f1_sum"] / fc)
        if bc > 0:
            vals.append(pr["metrics"][LABEL_TABLE]["norm_editdistance_sum"] / bc)
            vals.append(pr["metrics"][LABEL_TABLE]["teds_sum"] / bc)

        page_scores.append(sum(vals) / len(vals) if vals else 0.0)

    final_score = sum(page_scores) / n_pages if n_pages > 0 else 0.0

    result = {
        "num_pages_used": n_pages,
        "label_counts": label_counts,
        "matrix_sum": {
            LABEL_TEXT: dict(matrix_sum[LABEL_TEXT]),
            LABEL_FORMULA: dict(matrix_sum[LABEL_FORMULA]),
            LABEL_TABLE: dict(matrix_sum[LABEL_TABLE]),
        },
        "matrix_avg": matrix_avg,
        "final_score_div_num_pages": final_score,
        "page_scores": page_scores,
        "page_reports": page_reports,
    }

    if args.debug:
        result["debug_stats"] = dict(dbg)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "num_pages_used": n_pages,
        "label_counts": label_counts,
        "matrix_avg": matrix_avg,
        "final_score_div_num_pages": final_score,
        "debug_stats": dict(dbg) if args.debug else None
    }, ensure_ascii=False, indent=2))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()