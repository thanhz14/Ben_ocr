import json

LABEL_MAP = {
    "figure_title": "figure_caption",
}

IGNORE_CLASSES = {
    "abstract",
    "paragraph_title"
}


def load_paddle(path):

    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)

    W = data["width"]
    H = data["height"]

    objects = []

    for idx, b in enumerate(data["parsing_res_list"]):

        label = b.get("block_label")

        if label is None:
            continue

        if label in IGNORE_CLASSES:
            continue

        label = LABEL_MAP.get(label, label)

        bbox = b.get("block_bbox")

        if bbox is None:
            continue

        objects.append({
            "id": idx,
            "label": label,
            "bbox": [
                bbox[0] / W,
                bbox[1] / H,
                bbox[2] / W,
                bbox[3] / H
            ],
            "content": b.get("block_content", "")
        })

    return objects