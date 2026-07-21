import json
import os
from collections import defaultdict
from pathlib import Path

def normalize_bbox(bbox, page_width=1224, page_height=1584):
    """
    Normalize bounding box from [x, y, width, height] to [0,1] range
    
    Args:
        bbox: {x, y, width, height} format (already normalized or pixel values)
        page_width: Page width in pixels
        page_height: Page height in pixels
    
    Returns:
        dict: {x, y, width, height} in [0,1] range
    """
    x = bbox.get('x', 0)
    y = bbox.get('y', 0)
    width = bbox.get('width', 0)
    height = bbox.get('height', 0)
    
    # If already normalized (values between 0-1), return as is
    if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
        return {'x': x, 'y': y, 'width': width, 'height': height}
    
    # Otherwise normalize from pixel values
    return {
        'x': x / page_width,
        'y': y / page_height,
        'width': width / page_width,
        'height': height / page_height
    }


def parse_label_studio_json(input_file, output_file=None, page_width=1224, page_height=1584):
    """
    Parse Label Studio JSON export into simplified format
    Convert LaTeX tables to HTML for 'table' labels
    
    Args:
        input_file: Path to Label Studio JSON export
        output_file: Path to save simplified JSON (optional)
        page_width: Page width in pixels
        page_height: Page height in pixels
    
    Returns:
        dict: Simplified annotation structure
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Group by page_index
    pages_dict = defaultdict(list)
    file_name = None
    
    for task in data:
        file_name = task.get('file_upload', 'unknown')
        
        # Get annotations
        annotations = task.get('annotations', [])
        for annotation in annotations:
            results = annotation.get('result', [])
            
            for result in results:
                if result.get('type') != 'ocrlabels':
                    continue
                
                value = result.get('value', {})
                meta = result.get('meta', {})
                page_index = value.get('pageIndex', 1) -1
                
                # Get order from meta.text
                order_list = meta.get('text', [])
                order = order_list[0] if order_list and order_list[0] != 'null' else None
                
                # Get label
                labels = value.get('ocrlabels', [])
                label = labels[0] if labels else None
                
                # Get content
                content = value.get('ocrtext', '')
                
                # Normalize bounding box
                bbox = {
                    'x': value.get('x'),
                    'y': value.get('y'),
                    'width': value.get('width'),
                    'height': value.get('height')
                }
                normalized_bbox = normalize_bbox(bbox, page_width, page_height)
                
                block = {
                    'id': result.get('id'),
                    'order': order,
                    'bounding_box': normalized_bbox,
                    'block_content': content,
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
    
    result = {
        'file_name': file_name,
        'pages': pages
    }
    
    # Save to file if output_file specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f'✅ Saved to {output_file}')
    
    return result


def batch_parse_label_studio(input_dir='data/raw_gt', output_dir='data/gt', prefix='paper_', page_width=1224, page_height=1584):
    """
    Parse multiple Label Studio JSON files from input directory
    
    Args:
        input_dir: Directory containing input JSON files (default: data/raw_gt)
        output_dir: Directory to save output JSON files (default: data/gt)
        prefix: Prefix of input files to match (default: paper_)
        page_width: Page width in pixels
        page_height: Page height in pixels
    """
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all matching input files
    input_path = Path(input_dir)
    input_files = sorted(input_path.glob(f'{prefix}*.json'))
    
    if not input_files:
        print(f'⚠️  No files found matching pattern "{prefix}*.json" in {input_dir}')
        return
    
    print(f'📂 Found {len(input_files)} files to process')
    
    # Process each file
    for input_file in input_files:
        file_stem = input_file.stem  # e.g., 'paper_0'
        output_file = os.path.join(output_dir, f'gt_{file_stem}.json')
        
        try:
            print(f'⏳ Processing {input_file.name}...')
            parse_label_studio_json(str(input_file), output_file, page_width, page_height)
            print(f'   ✅ Success: {output_file}')
        except Exception as e:
            print(f'   ❌ Error: {e}')


# Usage
if __name__ == '__main__':
    batch_parse_label_studio(
        input_dir='data/raw_gt',
        output_dir='data/gt',
        prefix='paper_'
    )