# Ben OCR Benchmark

Công cụ chuẩn hóa dữ liệu và đánh giá kết quả OCR tài liệu theo từng loại
block: `Text`, `Formula`, `Table` và `Image`.

## Chỉ số đánh giá

- Detection: Precision, Recall, F1 và IoU với ngưỡng match `IoU > 0.5`.
- Text: normalized Levenshtein similarity (được hiển thị là NED, càng cao càng tốt).
- Formula: CDM (Character Detection Matching).
- Table: TEDS (Tree Edit Distance-based Similarity).
- Overall: `(NED + CDM + TEDS) / 3`.

NED, CDM và TEDS chỉ được tính trên các block match đúng label. Block không
match được phản ánh qua Precision, Recall và F1.

## Cấu trúc chính

```text
data/raw_gt/       Ground truth gốc từ Label Studio
data/gt/           Ground truth đã chuẩn hóa
data/raw_pred/     Prediction thô theo paper/page
data/pred/         Prediction đã chuẩn hóa
data/deepseek/     Prediction đang được main.py đánh giá
parser/gt.py       Chuẩn hóa GT và chuyển bảng LaTeX sang HTML
parser/pred.py     Chuẩn hóa prediction
metric/            NED, TEDS và CDM
main.py            Match block và đánh giá toàn dataset
```

## Cài đặt

Khuyến nghị Linux và Python 3.10 trở lên:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

CDM còn cần TeX Live, ImageMagick và Ghostscript để render công thức. Có thể
tắt lưu ảnh trực quan CDM để giảm dung lượng và thời gian:

```bash
export CDM_SAVE_VIS=0
```

## Chuẩn hóa dữ liệu

Chuẩn hóa toàn bộ ground truth:

```bash
python -m parser.gt
```

Kết quả được ghi thành `data/gt/gt_<id>.json`. Parser hỗ trợ `tabular`,
`tabularx`, `multirow` và `multicolumn`.

Parser cũng nhận trực tiếp một file export tổng từ Label Studio, không cần tách
thành `data/raw_gt/*.json` trước:

```bash
python -m parser.gt raw_gt.json --output-dir data/gt_annotator_2
```

Chuẩn hóa prediction:

```bash
python -m parser.pred
```

Kết quả được ghi thành `data/pred/pred_<id>.json`.

## Chạy đánh giá

Trong `main.py`, chỉnh `pred_dir` nếu muốn đánh giá bộ prediction khác:

```python
gt_dir = "data/gt"
pred_dir = "data/deepseek"
```

Sau đó chạy:

```bash
python main.py
```

Tiến độ được hiển thị theo paper, page và label. Terminal in năm điểm tổng:

```text
NED: ...
CDM: ...
TEDS: ...
Overall: ...
mIoU: ...
```

Kết quả đầy đủ theo label, gồm Precision, Recall, F1 và các chỉ số recognition,
được lưu tại `result/evaluation/evaluation_results.json`.

## Thử riêng metric Table

```bash
python -m metric.table
```

## Ghi chú

- Matching block dùng greedy matching theo IoU, thực hiện riêng theo label.
- Mỗi prediction chỉ được match với tối đa một ground-truth block.
- Không đưa `venv`, cache Python, file tạm và ảnh debug CDM vào gói nộp bài.
