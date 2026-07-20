from pathlib import Path

from parser.labelstudio import load_labelstudio
from parser.paddle import load_paddle

from iou import LayoutEvaluator


GT_ROOT = Path("data/gt")
PRED_ROOT = Path("data/pred")

evaluator = LayoutEvaluator()

dataset_tp = 0
dataset_fp = 0
dataset_fn = 0

dataset_iou = 0
dataset_match = 0
for gt_file in GT_ROOT.glob("*.json"):

    paper_name = gt_file.stem

    print("=" * 60)
    print(paper_name)

    gt_pages = load_labelstudio(gt_file)

    pred_folder = PRED_ROOT / paper_name

    TP = FP = FN = 0
    total_iou = 0
    matched = 0

    for page_name, gt in gt_pages.items():

        pred_file = pred_folder / f"{page_name}.json"

        if not pred_file.exists():

            print(page_name, "missing prediction")

            continue

        pred = load_paddle(pred_file)

        result = evaluator.evaluate(gt, pred)
        for d in result["details"]:

            if d["status"] == "WrongLabel":

                print()

                print("Wrong Label")

                print("GT")

                print(d["gt"])

                print("Prediction")

                print(d["pred"])

                print("IoU", d["iou"])

            elif d["status"] == "Miss":

                print()

                print("Missing GT")

                print(d["gt"])

            elif d["status"] == "FalseDetection":

                print()

                print("False Detection")

                print(d["pred"])
        TP += result["TP"]
        FP += result["FP"]
        FN += result["FN"]

        total_iou += result["MeanIoU"] * result["TP"]
        matched += result["TP"]

        print(
            page_name,
            result["F1"],
            result["MeanIoU"]
        )

    precision = TP / (TP + FP) if TP + FP else 0

    recall = TP / (TP + FN) if TP + FN else 0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    mean_iou = total_iou / matched if matched else 0

    print()
    dataset_tp += TP
    dataset_fp += FP
    dataset_fn += FN

    dataset_iou += total_iou
    dataset_match += matched
    print("Paper Result")

    print("IoU      :", mean_iou)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1       :", f1)
dataset_precision = dataset_tp / (dataset_tp + dataset_fp)

dataset_recall = dataset_tp / (dataset_tp + dataset_fn)

dataset_f1 = (
    2 * dataset_precision * dataset_recall /
    (dataset_precision + dataset_recall)
)

dataset_iou = dataset_iou / dataset_match

print()
print("="*60)
print("DATASET RESULT")
print("="*60)

print("IoU      :", dataset_iou)
print("Precision:", dataset_precision)
print("Recall   :", dataset_recall)
print("F1       :", dataset_f1)