import json
from pathlib import Path


def load_labelstudio(path):

    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)

    pages = {}

    for task in data:

        stem = Path(task["file_upload"]).stem

        # Ví dụ: "76b63cfe-page_001"

        page = stem.split("-")[-1]      # "page_001"

        page_num = int(page.split("_")[1]) - 1

        page_name = f"page_{page_num}"

        objects = []

        results = task["annotations"][0]["result"]

        for r in results:

            value = r["value"]

            W = r["original_width"]
            H = r["original_height"]

            x = value["x"] / 100
            y = value["y"] / 100
            w = value["width"] / 100
            h = value["height"] / 100

            objects.append({

                "label": value["rectanglelabels"][0],

                "bbox": [
                    x,
                    y,
                    x + w,
                    y + h
                ]

            })

        pages[page_name] = objects

    return pages