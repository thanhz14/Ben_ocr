from rapidfuzz.distance import Levenshtein

from .table_metric import TEDS


class TableMetric:

    def __init__(self):
        self.teds = TEDS()

    def edit_distance(self, gt: str, pred: str):
        return Levenshtein.distance(gt, pred)

    def edit_similarity(self, gt: str, pred: str):
        return Levenshtein.normalized_similarity(gt, pred)

    def teds_score(self, gt_html: str, pred_html: str):
        return self.teds.evaluate(gt_html, pred_html)

    def evaluate(self, gt_html: str, pred_html: str):
        return {
            "edit_distance": self.edit_distance(gt_html, pred_html),
            "edit_similarity": self.edit_similarity(gt_html, pred_html),
            "teds": self.teds_score(gt_html, pred_html),
        }


if __name__ == "__main__":

    gt = """
    <table>
        <tr><td>A</td><td>B</td></tr>
    </table>
    """

    pred = """
    <table>
        <tr><td>A</td><td>C</td></tr>
    </table>
    """

    metric = TableMetric()

    result = metric.evaluate(gt, pred)

    for k, v in result.items():
        print(f"{k}: {v}")