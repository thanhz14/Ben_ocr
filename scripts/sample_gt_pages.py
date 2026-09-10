import argparse
import json
import re
from pathlib import Path


def paper_number(path: Path) -> int:
    match = re.fullmatch(r"gt_(\d+)\.json", path.name)
    if not match:
        raise ValueError(f"Unexpected ground-truth filename: {path.name}")
    return int(match.group(1))


def read_pages_to_annotate(path: Path) -> dict[int, list[int]]:
    """Read ``- [ ] Paper N: trang ...`` entries from the checklist."""
    selections = {}
    line_pattern = re.compile(
        r"^\s*-\s*\[[ xX]\]\s*Paper\s+(\d+)\s*:\s*trang\s+(.+?)\s*$"
    )

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = line_pattern.match(line)
        if not match:
            continue

        paper_id = int(match.group(1))
        if paper_id in selections:
            raise ValueError(f"duplicate Paper {paper_id} on line {line_number}")

        # Only the comma-separated integers are significant; trailing notes such
        # as "ok" in the checklist are intentionally ignored.
        page_text = match.group(2)
        pages_match = re.match(r"\d+(?:\s*,\s*\d+)*", page_text)
        if not pages_match:
            raise ValueError(f"no page indices found on line {line_number}: {line}")
        page_indices = [int(value) for value in re.findall(r"\d+", pages_match.group())]
        if len(page_indices) != len(set(page_indices)):
            raise ValueError(f"duplicate page index for Paper {paper_id}")
        selections[paper_id] = page_indices

    if not selections:
        raise ValueError(f"no paper/page entries found in {path}")
    return selections


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a GT subset using pages listed in pages_to_annotate.md."
    )
    parser.add_argument("--input", type=Path, default=Path("data/gt"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/gt_agreement_10pct")
    )
    parser.add_argument(
        "--pages-file",
        type=Path,
        help="checklist to read (default: <output>/pages_to_annotate.md)",
    )
    args = parser.parse_args()

    pages_file = args.pages_file or args.output / "pages_to_annotate.md"
    if not pages_file.is_file():
        parser.error(f"pages checklist not found: {pages_file}")
    try:
        requested_pages = read_pages_to_annotate(pages_file)
    except ValueError as error:
        parser.error(str(error))

    sources = sorted(args.input.glob("gt_*.json"), key=paper_number)
    if not sources:
        parser.error(f"no gt_*.json files found in {args.input}")

    documents = {}
    for source in sources:
        with source.open(encoding="utf-8") as file:
            document = json.load(file)
        documents[paper_number(source)] = (source, document)

    unknown_papers = sorted(set(requested_pages) - set(documents))
    if unknown_papers:
        parser.error(f"papers not found in {args.input}: {unknown_papers}")

    total_pages = sum(len(document.get("pages", [])) for _, document in documents.values())
    sample_size = sum(len(indices) for indices in requested_pages.values())

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_papers = []
    label_counts = {}

    for paper_id, page_indices in requested_pages.items():
        source, document = documents[paper_id]
        pages_by_index = {page["page_index"]: page for page in document.get("pages", [])}
        missing_pages = sorted(set(page_indices) - set(pages_by_index))
        if missing_pages:
            parser.error(f"pages not found for Paper {paper_id}: {missing_pages}")
        selected_pages = [pages_by_index[index] for index in page_indices]
        subset = {**document, "pages": selected_pages}
        destination = args.output / source.name
        with destination.open("w", encoding="utf-8") as file:
            json.dump(subset, file, ensure_ascii=False, indent=2)
            file.write("\n")

        for page in selected_pages:
            for block in page.get("blocks", []):
                label = block.get("label", "Unknown")
                label_counts[label] = label_counts.get(label, 0) + 1

        manifest_papers.append(
            {
                "paper_id": paper_id,
                "source_file": str(source),
                "output_file": str(destination),
                "total_pages": len(document.get("pages", [])),
                "selected_page_indices": page_indices,
            }
        )

    manifest = {
        "source_directory": str(args.input),
        "pages_file": str(pages_file),
        "sampling_unit": "page",
        "sampling_method": "explicit page indices from checklist",
        "total_pages": total_pages,
        "selected_pages": sample_size,
        "actual_ratio": sample_size / total_pages,
        "selected_block_counts": dict(sorted(label_counts.items())),
        "papers": manifest_papers,
    }
    with (args.output / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(
        f"Copied {sample_size}/{total_pages} listed pages "
        f"({sample_size / total_pages:.2%}) into {args.output}"
    )
    print(f"Block labels: {manifest['selected_block_counts']}")


if __name__ == "__main__":
    main()
