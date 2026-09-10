import argparse
import json
import os
import re
from collections import defaultdict
from html import unescape
from pathlib import Path

from lxml import etree, html


def normalize_bbox(bbox, page_width=1224, page_height=1584):
    x = bbox.get("x", 0)
    y = bbox.get("y", 0)
    width = bbox.get("width", 0)
    height = bbox.get("height", 0)

    if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
        return {"x": x, "y": y, "width": width, "height": height}

    return {
        "x": x / page_width,
        "y": y / page_height,
        "width": width / page_width,
        "height": height / page_height,
    }


def _norm_text(s: str) -> str:
    if s is None:
        return ""
    s = unescape(str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_latex_cmds(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\\(?:toprule|midrule|bottomrule|hline)\b", "", s)
    s = re.sub(r"\\c(?:mid)?rule(?:\([^)]*\))?\{[^}]*\}", "", s)
    # Unwrap common presentational commands repeatedly so nested wrappers work.
    wrappers = r"textbf|textit|mathrm|mathbf|emph|underline|operatorname|makecell"
    old = None
    while old != s:
        old = s
        s = re.sub(rf"\\(?:{wrappers})\{{([^{{}}]*)\}}", r"\1", s)
    s = s.replace("\\methodshort", "method")
    s = s.replace("\\%", "%")
    s = s.replace("\\&", "&")
    s = s.replace("\\downarrow", "↓")
    s = re.sub(r"(?<!\\)\\\\", " ", s)
    return _norm_text(s)


def _read_group(source: str, pos: int):
    """Read one balanced {...} group and return (content, next_position)."""
    while pos < len(source) and source[pos].isspace():
        pos += 1
    if pos >= len(source) or source[pos] != "{":
        return None, pos
    depth = 0
    start = pos + 1
    for i in range(pos, len(source)):
        if source[i] == "{" and (i == 0 or source[i - 1] != "\\"):
            depth += 1
        elif source[i] == "}" and (i == 0 or source[i - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return source[start:i], i + 1
    return None, pos


def _extract_tabular_body(latex: str) -> str:
    begin = re.search(r"\\begin\{(tabular\*?|tabularx)\}", latex)
    if not begin:
        return latex
    env = begin.group(1)
    pos = begin.end()
    # tabular has a column specification; tabularx has width + specification.
    for _ in range(2 if env == "tabularx" else 1):
        _, next_pos = _read_group(latex, pos)
        if next_pos == pos:
            break
        pos = next_pos
    end = re.search(rf"\\end\{{{re.escape(env)}\}}", latex[pos:])
    return latex[pos:pos + end.start()] if end else latex[pos:]


def _split_top_level(source: str, separator: str):
    r"""Split on & or \\ only outside braces and preserve escaped \&."""
    parts, buf, depth = [], [], 0
    i = 0
    while i < len(source):
        ch = source[i]
        escaped = i > 0 and source[i - 1] == "\\"
        if ch == "{" and not escaped:
            depth += 1
        elif ch == "}" and not escaped:
            depth = max(0, depth - 1)
        if separator == "&" and ch == "&" and not escaped and depth == 0:
            parts.append("".join(buf)); buf = []; i += 1; continue
        if separator == "\\\\" and source.startswith("\\\\", i) and depth == 0:
            parts.append("".join(buf)); buf = []; i += 2; continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def _command_args(cell: str, command: str, count: int):
    match = re.match(rf"\s*\\{command}\s*", cell, flags=re.S)
    if not match:
        return None
    args, pos = [], match.end()
    for _ in range(count):
        while pos < len(cell) and cell[pos].isspace():
            pos += 1
        if pos < len(cell) and cell[pos] == "*":
            args.append("*"); pos += 1; continue
        arg, next_pos = _read_group(cell, pos)
        if arg is None:
            return None
        args.append(arg); pos = next_pos
    return args, cell[pos:]


def _span(value) -> int:
    try:
        return max(1, int(float(str(value).strip())))
    except (TypeError, ValueError):
        return 1


def _parse_latex_cell(cell: str):
    rowspan = colspan = 1
    content = cell.strip()
    # Commands may be nested in either order; unwrap until neither is present.
    for _ in range(2):
        parsed = _command_args(content, "multicolumn", 3)
        if parsed:
            args, tail = parsed
            colspan = _span(args[0])
            content = args[2] + tail
            continue
        parsed = _command_args(content, "multirow", 3)
        if parsed:
            args, tail = parsed
            rowspan = _span(args[0])
            content = args[2] + tail
            continue
        break
    return _strip_latex_cmds(content), rowspan, colspan


def latex_tabular_to_html(latex: str) -> str:
    """Convert tabular/tabularx into the minimal HTML expected by TEDS."""
    if not latex:
        return "<table></table>"
    content = _extract_tabular_body(latex)
    content = re.sub(r"(?<!\\)%[^\n]*", "", content)  # LaTeX comments
    content = re.sub(
        r"\\(?:toprule|midrule|bottomrule|hline)\b|"
        r"\\c(?:mid)?rule(?:\([^)]*\))?\{[^}]*\}",
        "", content,
    )

    out_table = etree.Element("table")
    active_rowspans = {}
    for raw_row in _split_top_level(content, "\\\\"):
        raw_cells = _split_top_level(raw_row.strip(), "&")
        if not any(c.strip() for c in raw_cells):
            continue
        tr = etree.SubElement(out_table, "tr")
        new_rowspans = {}
        column = 0
        for raw_cell in raw_cells:
            # LaTeX commonly writes an empty placeholder beneath multirow.
            if column in active_rowspans and not raw_cell.strip():
                column += 1
                continue
            while column in active_rowspans:
                column += 1
            text, rowspan, colspan = _parse_latex_cell(raw_cell)
            td = etree.SubElement(tr, "td")
            if rowspan > 1:
                td.set("rowspan", str(rowspan))
            if colspan > 1:
                td.set("colspan", str(colspan))
            td.text = text
            if rowspan > 1:
                for occupied in range(column, column + colspan):
                    new_rowspans[occupied] = rowspan - 1
            column += colspan
        active_rowspans = {
            col: remaining - 1 for col, remaining in active_rowspans.items()
            if remaining > 1
        }
        active_rowspans.update(new_rowspans)

    return etree.tostring(out_table, encoding="unicode", method="html", with_tail=False)


def canonicalize_table_html(raw_html: str) -> str:
    if not raw_html:
        return "<table></table>"

    parser = html.HTMLParser(remove_comments=True, encoding="utf-8")

    try:
        root = html.fromstring(raw_html, parser=parser)
    except Exception:
        txt = _norm_text(raw_html)
        return f"<table><tr><td>{txt}</td></tr></table>"

    table = root if getattr(root, "tag", None) == "table" else (root.xpath("//table")[0] if root.xpath("//table") else None)

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


def normalize_table_content(raw: str) -> str:
    s = (raw or "").strip()
    if re.search(r"\\begin\{tabular(?:x|\*)?\}", s):
        return canonicalize_table_html(latex_tabular_to_html(s))
    # already html (or html-like)
    return canonicalize_table_html(s)


def _normalize_tasks(raw, source="input"):
    if isinstance(raw, list):
        return raw
    elif isinstance(raw, dict):
        if "tasks" in raw and isinstance(raw["tasks"], list):
            return raw["tasks"]
        elif "annotations" in raw:
            return [raw]
        else:
            for v in raw.values():
                if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                    return v
            raise ValueError(f"Unsupported JSON structure in {source}")
    raise ValueError(f"Unsupported root type: {type(raw)} in {source}")


def _parse_tasks(data, page_width=1224, page_height=1584):
    pages_dict = defaultdict(list)
    file_name = None

    for task in data:
        file_name = task.get("file_upload", "unknown")
        annotations = task.get("annotations", [])

        for annotation in annotations:
            results = annotation.get("result", [])

            for result in results:
                if result.get("type") != "ocrlabels":
                    continue

                value = result.get("value", {})
                meta = result.get("meta", {})
                page_index = value.get("pageIndex", 1)

                order_list = meta.get("text", [])
                order = order_list[0] if order_list and order_list[0] != "null" else None

                labels = value.get("ocrlabels", [])
                label = labels[0] if labels else None

                content = value.get("ocrtext", "")
                if str(label).strip().lower() == "table":
                    content = normalize_table_content(content)

                bbox = {
                    "x": value.get("x"),
                    "y": value.get("y"),
                    "width": value.get("width"),
                    "height": value.get("height"),
                }

                block = {
                    "id": result.get("id"),
                    "order": order,
                    "bounding_box": normalize_bbox(bbox, page_width, page_height),
                    "block_content": content,
                    "label": label,
                }

                pages_dict[page_index].append(block)

    pages = [{"page_index": i, "blocks": pages_dict[i]} for i in sorted(pages_dict.keys())]
    return {"file_name": file_name, "pages": pages}


def _write_gt(result, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Saved to {output_file}")


def parse_label_studio_json(input_file, output_file=None, page_width=1224, page_height=1584):
    with open(input_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = _normalize_tasks(raw, input_file)
    result = _parse_tasks(data, page_width, page_height)

    if output_file:
        _write_gt(result, output_file)

    return result


def _task_paper_id(task, fallback_index):
    candidates = [
        task.get("data", {}).get("pdf", ""),
        task.get("file_upload", ""),
    ]
    for candidate in candidates:
        match = re.search(r"-(\d+)\.pdf(?:$|[?&])", str(candidate), re.IGNORECASE)
        if match:
            return match.group(1)

    inner_id = task.get("inner_id")
    if inner_id is not None:
        return str(inner_id)
    return str(fallback_index)


def parse_label_studio_export(
    input_file, output_dir="data/gt", page_width=1224, page_height=1584
):
    """Parse one Label Studio export containing multiple tasks directly to GT files."""
    input_file = Path(input_file)
    with input_file.open("r", encoding="utf-8") as f:
        tasks = _normalize_tasks(json.load(f), input_file)

    output_dir = Path(output_dir)
    used_ids = set()
    print(f"Found {len(tasks)} tasks in {input_file}")
    for index, task in enumerate(tasks, start=1):
        paper_id = _task_paper_id(task, index)
        if paper_id in used_ids:
            raise ValueError(f"Duplicate paper ID {paper_id} in {input_file}")
        used_ids.add(paper_id)
        output_file = output_dir / f"gt_{paper_id}.json"
        result = _parse_tasks([task], page_width, page_height)
        _write_gt(result, output_file)

    return len(tasks)


def batch_parse_label_studio(input_dir="raw_gt.json", output_dir="data/gt_temp", prefix=None, page_width=1224, page_height=1584):
    os.makedirs(output_dir, exist_ok=True)

    pattern = "*.json" if not prefix else f"{prefix}*.json"
    input_files = sorted(Path(input_dir).glob(pattern), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)

    if not input_files:
        print(f'⚠️  No files found matching pattern "{pattern}" in {input_dir}')
        return

    print(f"📂 Found {len(input_files)} files to process")
    for input_file in input_files:
        output_file = os.path.join(output_dir, f"gt_{input_file.stem}.json")
        try:
            print(f"⏳ Processing {input_file.name}...")
            parse_label_studio_json(str(input_file), output_file, page_width, page_height)
            print(f"   ✅ Success: {output_file}")
        except Exception as e:
            print(f"   ❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Normalize Label Studio OCR exports into gt_<id>.json files."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("raw_gt.json"),
        help="A Label Studio export JSON file or a directory of per-task JSON files.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/gt"))
    args = parser.parse_args()

    if args.input.is_file():
        parse_label_studio_export(args.input, args.output_dir)
    elif args.input.is_dir():
        batch_parse_label_studio(args.input, args.output_dir)
    else:
        parser.error(f"input does not exist: {args.input}")


if __name__ == "__main__":
    main()
