import json
import os
import re
from collections import defaultdict
from html import unescape
from pathlib import Path

from lxml import etree, html


def normalize_bbox(bbox, page_width=1224, page_height=1584):
    """
    Normalize bounding box from [x1, y1, x2, y2] to [0,1] range
    """
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        return {
            "x": x1 / page_width,
            "y": y1 / page_height,
            "width": (x2 - x1) / page_width,
            "height": (y2 - y1) / page_height,
        }
    return {"x": 0, "y": 0, "width": 0, "height": 0}


def _norm_text(s: str) -> str:
    if s is None:
        return ""
    s = unescape(str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _looks_like_html_table(content: str) -> bool:
    if not content:
        return False
    s = content.lower()
    return "<table" in s and "</table>" in s


def canonicalize_table_html(raw_html: str) -> str:
    """
    Canonicalize table HTML to stable minimal form:
    <table><tr><td>...</td></tr>...</table>
    """
    if not raw_html:
        return "<table></table>"

    parser = html.HTMLParser(remove_comments=True, encoding="utf-8")

    try:
        root = html.fromstring(raw_html, parser=parser)
    except Exception:
        txt = _norm_text(raw_html)
        return f"<table><tr><td>{txt}</td></tr></table>"

    table = None
    if getattr(root, "tag", None) == "table":
        table = root
    else:
        tables = root.xpath("//table")
        if tables:
            table = tables[0]

    if table is None:
        txt = _norm_text(root.text_content() if hasattr(root, "text_content") else raw_html)
        return f"<table><tr><td>{txt}</td></tr></table>"

    out_table = etree.Element("table")
    rows = table.xpath("./tbody/tr") or table.xpath("./tr") or table.xpath(".//tr")

    for tr_node in rows:
        tr = etree.SubElement(out_table, "tr")
        for c in tr_node.xpath("./th|./td"):
            td = etree.SubElement(tr, "td")
            rs = c.attrib.get("rowspan")
            cs = c.attrib.get("colspan")
            if rs and rs.isdigit() and int(rs) > 1:
                td.set("rowspan", str(int(rs)))
            if cs and cs.isdigit() and int(cs) > 1:
                td.set("colspan", str(int(cs)))
            td.text = _norm_text(c.text_content())

    return etree.tostring(out_table, encoding="unicode", method="html", with_tail=False)


def map_label(block_label, block_content=""):
    """
    Map to 4 standard labels: Text, Image, Formula, Table

    Priority:
    1) If content contains HTML table => Table
    2) By block_label keywords
    """
    if _looks_like_html_table(block_content):
        return "Table"

    b = str(block_label).lower().strip()

    if "table" in b:
        return "Table"
    if "formula" in b or "equation" in b or "math" in b:
        return "Formula"
    if "image" in b or "figure" in b or "img" in b:
        return "Image"
    if "chart" in b:
        return "Image"  # chart -> Image (unless content is table, handled above)
    return "Text"


def parse_pred_json(input_file, page_width=1224, page_height=1584):
    """
    Parse prediction JSON from model output
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages_dict = defaultdict(list)

    page_width = data.get("width", page_width)
    page_height = data.get("height", page_height)
    parsing_res_list = data.get("parsing_res_list", [])

    for item in parsing_res_list:
        block_label = item.get("block_label", "text")
        block_content = item.get("block_content", "")
        block_bbox = item.get("block_bbox", [0, 0, 0, 0])
        block_id = item.get("block_id")
        block_order = item.get("block_order")
        page_index = data.get("page_index", 0)+1

        label = map_label(block_label, block_content)

        # Canonicalize table content for stable compare
        if label == "Table":
            block_content = canonicalize_table_html(block_content)

        block = {
            "id": str(block_id),
            "order": block_order,
            "bounding_box": normalize_bbox(block_bbox, page_width, page_height),
            "content": block_content,
            "label": label,
        }

        pages_dict[page_index].append(block)

    pages = [{"page_index": i, "blocks": pages_dict[i]} for i in sorted(pages_dict.keys())]
    return pages


def batch_parse_predictions(input_dir="data/raw_pred", output_dir="data/pred"):
    """
    Parse prediction JSON files from directory structure
    Each subdirectory (paper_0, paper_1, etc.) contains page JSONs
    """
    os.makedirs(output_dir, exist_ok=True)

    input_path = Path(input_dir)
    paper_dirs = sorted([d for d in input_path.iterdir() if d.is_dir() and d.name.startswith("paper_")])

    if not paper_dirs:
        print(f'⚠️  No directories found matching pattern "paper_*" in {input_dir}')
        return

    print(f"📂 Found {len(paper_dirs)} paper directories")

    for paper_dir in paper_dirs:
        paper_name = paper_dir.name
        paper_num = paper_name.replace("paper_", "")
        output_file = os.path.join(output_dir, f"pred_{paper_num}.json")

        print(f"⏳ Processing {paper_name}...")

        json_files = sorted(paper_dir.glob("*.json"))
        if not json_files:
            print(f"   ⚠️  No JSON files found in {paper_dir}")
            continue

        all_pages = []
        for json_file in json_files:
            try:
                pages = parse_pred_json(str(json_file))
                all_pages.extend(pages)
            except Exception as e:
                print(f"   ❌ Error processing {json_file.name}: {e}")

        try:
            result = {"file_name": paper_name, "pages": all_pages}
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"   ✅ Success: {output_file} ({len(all_pages)} pages)")
        except Exception as e:
            print(f"   ❌ Error saving output: {e}")


if __name__ == "__main__":
    batch_parse_predictions(input_dir="data/raw_pred", output_dir="data/pred")
