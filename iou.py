from utils import iou


class LayoutEvaluator:

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

    def evaluate(self, gt, pred):

        matched_pred = set()

        TP = 0
        FP = 0
        FN = 0

        total_iou = 0

        details = []

        for g in gt:

            best_iou = 0
            best_idx = -1

            for idx, p in enumerate(pred):

                if idx in matched_pred:
                    continue

                score = iou(g["bbox"], p["bbox"])

                if score > best_iou:
                    best_iou = score
                    best_idx = idx

            if best_iou >= self.iou_threshold:

                matched_pred.add(best_idx)

                p = pred[best_idx]

                if p["label"] == g["label"]:

                    TP += 1
                    total_iou += best_iou

                    details.append({
                        "status": "TP",
                        "gt": g,
                        "pred": p,
                        "iou": best_iou
                    })

                else:

                    FP += 1
                    FN += 1

                    details.append({
                        "status": "WrongLabel",
                        "gt": g,
                        "pred": p,
                        "iou": best_iou
                    })

            else:

                FN += 1

                details.append({
                    "status": "Miss",
                    "gt": g,
                    "pred": None,
                    "iou": 0
                })

        for idx, p in enumerate(pred):

            if idx not in matched_pred:

                FP += 1

                details.append({
                    "status": "FalseDetection",
                    "gt": None,
                    "pred": p,
                    "iou": 0
                })

        precision = TP / (TP + FP) if TP + FP else 0
        recall = TP / (TP + FN) if TP + FN else 0

        f1 = (
            2 * precision * recall /
            (precision + recall)
            if precision + recall else 0
        )

        mean_iou = total_iou / TP if TP else 0

        return {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "MeanIoU": mean_iou,
            "details": details
        }