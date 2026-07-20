import json


def convert(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {
        "page": data["page_index"],
        "width": data["width"],
        "height": data["height"],
        "blocks": []
    }

    for block in data["parsing_res_list"]:
        result["blocks"].append({
            "id": block["block_id"],
            "type": block["block_label"],
            "text": block["block_content"],
            "bbox": block["block_bbox"],
            "order": block["block_order"]
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    convert("D:\\Acer\\Code\\Capstone\\BEN\\data\\pred\\paper_1\\page_0.json", "pred_benchmark.json")