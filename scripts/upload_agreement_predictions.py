import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests


def get_access_token(base_url: str, refresh_token: str) -> str:
    response = requests.post(
        f"{base_url}/api/token/refresh/",
        json={"refresh": refresh_token},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["access"]


def api_request(session: requests.Session, method: str, url: str, **kwargs):
    response = session.request(method, url, timeout=60, **kwargs)
    if not response.ok:
        detail = response.text[:500]
        raise RuntimeError(f"Label Studio returned {response.status_code}: {detail}")
    return response


def extract_paper_id(task: dict) -> int:
    pdf_url = task.get("data", {}).get("pdf", "")
    parsed = urlparse(pdf_url)
    filepath = parse_qs(parsed.query).get("filepath", [parsed.path])[-1]
    filename = Path(filepath).name
    match = re.search(r"-(\d+)\.pdf$", filename, flags=re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Cannot determine paper ID from task {task.get('id')} PDF: {pdf_url}"
        )
    return int(match.group(1))


def task_map(session: requests.Session, base_url: str, project_id: int) -> dict[int, dict]:
    response = api_request(
        session,
        "GET",
        f"{base_url}/api/tasks/",
        params={"project": project_id, "page_size": 100},
    )
    payload = response.json()
    tasks = payload.get("tasks", payload.get("results", payload))
    mapped = {}
    for task in tasks:
        paper_id = extract_paper_id(task)
        if paper_id in mapped:
            raise ValueError(f"Multiple tasks found for paper {paper_id}")
        mapped[paper_id] = task
    return mapped


def block_to_result(block: dict, page_index: int, paper_id: int) -> dict:
    bbox = block["bounding_box"]
    result = {
        "id": f"pred-{paper_id}-{page_index}-{block['id']}",
        "from_name": "label",
        "to_name": "pdf",
        "type": "ocrlabels",
        "origin": "prediction",
        "value": {
            "x": bbox["x"],
            "y": bbox["y"],
            "width": bbox["width"],
            "height": bbox["height"],
            "rotation": 0,
            "pageIndex": page_index,
            "ocrtext": block.get("content", block.get("block_content", "")),
            "ocrlabels": [block["label"]],
        },
    }
    if block.get("order") is not None:
        result["meta"] = {"text": [str(block["order"])]}
    return result


def load_predictions(
    manifest_path: Path, prediction_dir: Path
) -> tuple[dict, dict[int, list[dict]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    predictions = {}
    for paper in manifest["papers"]:
        paper_id = paper["paper_id"]
        selected = set(paper["selected_page_indices"])
        path = prediction_dir / f"pred_{paper_id}.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        available = {page["page_index"] for page in document["pages"]}
        missing = sorted(selected - available)
        if missing:
            raise ValueError(f"{path} is missing selected pages: {missing}")

        results = []
        for page in document["pages"]:
            if page["page_index"] not in selected:
                continue
            results.extend(
                block_to_result(block, page["page_index"], paper_id)
                for block in page.get("blocks", [])
            )
        predictions[paper_id] = results
    return manifest, predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload pre-annotations for the agreement subset to Label Studio."
    )
    parser.add_argument("--base-url", default="https://app.heartex.com")
    parser.add_argument("--project", type=int, default=280522)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/gt_agreement_10pct/manifest.json"),
    )
    parser.add_argument(
        "--prediction-dir", type=Path, default=Path("data/pred_unlimit")
    )
    parser.add_argument("--model-version", default="agreement_unlimit_10pct_v1")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Create predictions. Without this flag, only show the upload plan.",
    )
    args = parser.parse_args()

    refresh_token = os.environ.get("LABEL_STUDIO_API_KEY")
    if not refresh_token:
        parser.error("set LABEL_STUDIO_API_KEY to your Label Studio refresh token")

    base_url = args.base_url.rstrip("/")
    manifest, predictions = load_predictions(args.manifest, args.prediction_dir)
    access_token = get_access_token(base_url, refresh_token)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {access_token}"})

    tasks = task_map(session, base_url, args.project)
    expected_papers = {paper["paper_id"] for paper in manifest["papers"]}
    missing_tasks = sorted(expected_papers - tasks.keys())
    if missing_tasks:
        raise ValueError(f"No Label Studio tasks found for papers: {missing_tasks}")

    existing_response = api_request(
        session,
        "GET",
        f"{base_url}/api/predictions/",
        params={"project": args.project},
    )
    existing = {
        (prediction.get("task"), prediction.get("model_version"))
        for prediction in existing_response.json()
    }

    total_blocks = sum(len(results) for results in predictions.values())
    print(
        f"Plan: {manifest['selected_pages']} pages, {total_blocks} blocks, "
        f"{len(expected_papers)} tasks, project {args.project}"
    )

    created = skipped = 0
    for paper in manifest["papers"]:
        paper_id = paper["paper_id"]
        task_id = tasks[paper_id]["id"]
        pages = paper["selected_page_indices"]
        results = predictions[paper_id]
        duplicate = (task_id, args.model_version) in existing
        status = "EXISTS" if duplicate else ("UPLOAD" if args.upload else "READY")
        print(
            f"[{status}] paper {paper_id:>2} -> task {task_id}: "
            f"pages {pages}, {len(results)} blocks"
        )
        if duplicate:
            skipped += 1
            continue
        if not args.upload:
            continue

        api_request(
            session,
            "POST",
            f"{base_url}/api/predictions/",
            json={
                "task": task_id,
                "project": args.project,
                "model_version": args.model_version,
                "score": 1.0,
                "result": results,
            },
        )
        created += 1

    if args.upload:
        print(f"Completed: created {created}, skipped existing {skipped}.")
    else:
        print("Dry run only. Add --upload to create predictions.")


if __name__ == "__main__":
    main()
