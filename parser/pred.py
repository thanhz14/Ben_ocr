import json
import os
from collections import defaultdict
from pathlib import Path

def normalize_bbox(bbox, page_width=1224, page_height=1584):
    """
    Normalize bounding box from [x1, y1, x2, y2] to [0,1] range
    
    Args:
        bbox: [x1, y1, x2, y2] format
        page_width: Page width in pixels
        page_height: Page height in pixels
    
    Returns:
        dict: {x, y, width, height} in [0,1] range
    """
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        return {
            'x': x1 / page_width,
            'y': y1 / page_height,
            'width': (x2 - x1) / page_width,
            'height': (y2 - y1) / page_height
        }
    return {'x': 0, 'y': 0, 'width': 0, 'height': 0}


def map_label(block_label):
    """
    Map any block label to 4 standard labels: Text, Image, Formula, Table
    
    Args:
        block_label: Original label from model
    
    Returns:
        str: Standard label (Text, Image, Formula, Table)
    """
    block_label = str(block_label).lower().strip()
    
    # Image labels
    if 'image' in block_label or 'figure' in block_label or 'img' in block_label:
        return 'Image'
    
    # Table labels
    if 'table' in block_label:
        return 'Table'
    
    # Formula labels
    if 'formula' in block_label or 'equation' in block_label or 'math' in block_label:
        return 'Formula'
    
    # Everything else → Text
    return 'Text'


def parse_pred_json(input_file, page_width=1224, page_height=1584):
    """
    Parse prediction JSON from model output
    
    Args:
        input_file: Path to prediction JSON
        page_width: Page width in pixels
        page_height: Page height in pixels
    
    Returns:
        dict: Parsed blocks grouped by page
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pages_dict = defaultdict(list)
    
    # Get page dimensions from input
    page_width = data.get('width', page_width)
    page_height = data.get('height', page_height)
    
    # Get parsing results
    parsing_res_list = data.get('parsing_res_list', [])
    
    for item in parsing_res_list:
        block_label = item.get('block_label', 'text')
        block_content = item.get('block_content', '')
        block_bbox = item.get('block_bbox', [0, 0, 0, 0])
        block_id = item.get('block_id')
        block_order = item.get('block_order')
        page_index = data.get('page_index', 0)
        
        # Map to 4 standard labels
        label = map_label(block_label)
        
        block = {
            'id': str(block_id),
            'order': block_order,
            'bounding_box': normalize_bbox(block_bbox, page_width, page_height),
            'content': block_content,
            'label': label
        }
        
        pages_dict[page_index].append(block)
    
    # Build output structure
    pages = []
    for page_index in sorted(pages_dict.keys()):
        pages.append({
            'page_index': page_index,
            'blocks': pages_dict[page_index]
        })
    
    return pages


def batch_parse_predictions(input_dir='data/raw_pred', output_dir='data/pred'):
    """
    Parse prediction JSON files from directory structure
    Each subdirectory (paper_0, paper_1, etc.) contains page JSONs
    
    Args:
        input_dir: Root directory containing paper_* subdirectories
        output_dir: Output directory to save parsed JSONs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    input_path = Path(input_dir)
    paper_dirs = sorted([d for d in input_path.iterdir() if d.is_dir() and d.name.startswith('paper_')])
    
    if not paper_dirs:
        print(f'⚠️  No directories found matching pattern "paper_*" in {input_dir}')
        return
    
    print(f'📂 Found {len(paper_dirs)} paper directories')
    
    # Process each paper directory
    for paper_dir in paper_dirs:
        paper_name = paper_dir.name  # e.g., 'paper_0'
        output_file = os.path.join(output_dir, f'pred_{paper_name}.json')
        
        print(f'⏳ Processing {paper_name}...')
        
        # Get all JSON files in the paper directory (one per page)
        json_files = sorted(paper_dir.glob('*.json'))
        
        if not json_files:
            print(f'   ⚠️  No JSON files found in {paper_dir}')
            continue
        
        all_pages = []
        
        # Process each page JSON
        for json_file in json_files:
            try:
                pages = parse_pred_json(str(json_file))
                all_pages.extend(pages)
            except Exception as e:
                print(f'   ❌ Error processing {json_file.name}: {e}')
        
        # Save combined output
        try:
            result = {
                'file_name': paper_name,
                'pages': all_pages
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f'   ✅ Success: {output_file} ({len(all_pages)} pages)')
        except Exception as e:
            print(f'   ❌ Error saving output: {e}')


# Usage
if __name__ == '__main__':
    batch_parse_predictions(
        input_dir='data/raw_pred',
        output_dir='data/pred'
    )