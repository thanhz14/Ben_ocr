from rapidfuzz.distance import Levenshtein

from metric.cdm_metric import CDM


class FormulaMetric:
    """
    Formula Recognition Metrics

    Metrics
    -------
    - Edit Distance
    - Edit Similarity
    - CDM (Compiled Document Metric)
    """

    def __init__(self, output_root="./result"):
        self.cdm = CDM(output_root)

    def edit_distance(self, gt: str, pred: str) -> int:
        """
        Levenshtein Edit Distance.
        """
        return Levenshtein.distance(gt, pred)

    def edit_similarity(self, gt: str, pred: str) -> float:
        """
        Normalized Edit Similarity.
        """
        return Levenshtein.normalized_similarity(gt, pred)

    def cdm_score(self, gt: str, pred: str, sample_id: str = "0") -> dict:
        """
        Evaluate formula using OmniDocBench CDM.
        """

        return self.cdm.evaluate(
            gt_latex=gt,
            pred_latex=pred,
            img_id=sample_id,
        )

    def evaluate(
        self,
        gt: str,
        pred: str,
        sample_id: str = "0",
    ) -> dict:

        cdm_result = self.cdm_score(gt, pred, sample_id)

        return {
            "edit_distance": self.edit_distance(gt, pred),
            "edit_similarity": self.edit_similarity(gt, pred),
            "cdm_recall": cdm_result["recall"],
            "cdm_precision": cdm_result["precision"],
            "cdm_f1": cdm_result["F1_score"],
            "cdm_tp": cdm_result["tp"],
            "cdm_gt_tokens": cdm_result["gt_tokens"],
            "cdm_pred_tokens": cdm_result["pred_tokens"],
        }


if __name__ == "__main__":

    metric = FormulaMetric()

    gt = r"\frac{a+b}{c}"
    pred = r"\frac{a+b}{d}"

    result = metric.evaluate(
        gt,
        pred,
        sample_id="demo_formula",
    )

    for k, v in result.items():
        print(f"{k}: {v}")