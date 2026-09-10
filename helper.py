import json
import os
import re
from pathlib import Path

# ==== Cấu hình ====
INPUT_JSON = "raw_gt.json"   # file json gốc (mảng các object)
OUTPUT_DIR = "data/raw_gt"       # thư mục chứa file tách ra


def extract_filename_from_pdf_path(pdf_path: str) -> str:
    """
    Ví dụ:
      upload/274344/2bc80d12-3.pdf -> 3.pdf
      upload/274344/abc-xyz-12.pdf -> 12.pdf
    """
    base = os.path.basename(pdf_path)  # 2bc80d12-3.pdf
    m = re.search(r"-([^.]+)\.pdf$", base, re.IGNORECASE)
    if m:
        return f"{m.group(1)}.pdf"

    # fallback nếu không match pattern "-<id>.pdf"
    return base


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("File JSON phải là một mảng (list) các phần tử.")

    for i, item in enumerate(data, start=1):
        pdf_path = item.get("data", {}).get("pdf", "")
        if not pdf_path:
            out_name = f"item_{i}.json"
        else:
            # tên theo yêu cầu: 3.pdf -> lưu json thành 3.json
            pdf_name = extract_filename_from_pdf_path(pdf_path)   # 3.pdf
            out_name = os.path.splitext(pdf_name)[0] + ".json"    # 3.json

        out_path = os.path.join(OUTPUT_DIR, out_name)

        # tránh ghi đè nếu trùng tên
        if os.path.exists(out_path):
            out_path = os.path.join(
                OUTPUT_DIR,
                f"{os.path.splitext(out_name)[0]}_{i}.json"
            )

        with open(out_path, "w", encoding="utf-8") as out_f:
            json.dump(item, out_f, ensure_ascii=False, indent=2)

        print(f"Đã tạo: {out_path}")


if __name__ == "__main__":
    main()