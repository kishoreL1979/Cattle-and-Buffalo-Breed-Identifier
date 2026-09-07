import os
import shutil
import json
from pathlib import Path
from PIL import Image

DATASET_ROOT = Path(r"c:\Users\lokik\OneDrive\Desktop\ai-breed-identifier\cnn dataset")
YOLO_ROOT = Path(r"c:\Users\lokik\OneDrive\Desktop\ai-breed-identifier\dataset_yolo")

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

BREED_NAMES = [
    "Gir",
    "Jaffrabadi",
    "Kankrej",
    "Mehsana",
    "Murrah",
    "Red Sindhi",
    "Sahiwal",
    "Surti",
    "Tharparkar"
]

BREED_TO_ID = {
    "gir": 0,
    "jaffrabadi": 1,
    "kankrej": 2,
    "mehsana": 3,
    "murrah": 4,
    "red_sindhi": 5,
    "sahiwal": 6,
    "surti": 7,
    "tharparkar": 8
}

def coco_to_yolo_bbox(coco_bbox, img_width, img_height):
    """
    Converts COCO bbox [x_min, y_min, w, h] to YOLO normalized [x_center, y_center, w, h].
    """
    try:
        x, y, w, h = float(coco_bbox[0]), float(coco_bbox[1]), float(coco_bbox[2]), float(coco_bbox[3])
        img_width = float(img_width)
        img_height = float(img_height)
    except Exception:
        return 0.5, 0.5, 1.0, 1.0

    if img_width <= 0 or img_height <= 0:
        return 0.5, 0.5, 1.0, 1.0
        
    x_center = (x + w / 2.0) / img_width
    y_center = (y + h / 2.0) / img_height
    norm_w = w / img_width
    norm_h = h / img_height
    
    # Clamp values to [0.0, 1.0]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_w = max(0.001, min(1.0, norm_w))
    norm_h = max(0.001, min(1.0, norm_h))
    
    return x_center, y_center, norm_w, norm_h

def prepare_dataset():
    print(f"Preparing YOLO dataset structure at: {YOLO_ROOT}")
    
    # Clean previous directory if exists
    if YOLO_ROOT.exists():
        shutil.rmtree(YOLO_ROOT, ignore_errors=True)

    # Create directories
    splits_map = {'train': 'train', 'valid': 'val', 'test': 'test'}
    for target_split in ['train', 'val', 'test']:
        (YOLO_ROOT / 'images' / target_split).mkdir(parents=True, exist_ok=True)
        (YOLO_ROOT / 'labels' / target_split).mkdir(parents=True, exist_ok=True)
        
    total_images_processed = 0
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    
    for orig_split, yolo_split in splits_map.items():
        orig_split_dir = DATASET_ROOT / orig_split
        if not orig_split_dir.is_dir():
            continue
            
        for breed_dir in sorted(orig_split_dir.iterdir()):
            if not breed_dir.is_dir():
                continue
                
            breed_key = breed_dir.name.lower()
            if breed_key not in BREED_TO_ID:
                continue
                
            class_id = BREED_TO_ID[breed_key]
            
            # Load COCO json if present
            coco_json_path = breed_dir / "_annotations.coco.json"
            coco_bboxes_by_file = {}
            
            if coco_json_path.exists():
                try:
                    with open(coco_json_path, 'r', encoding='utf-8') as f:
                        coco_data = json.load(f)
                        
                    img_id_to_file = {}
                    img_id_to_dim = {}
                    for img_obj in coco_data.get('images', []):
                        fn = Path(img_obj['file_name']).name
                        img_id_to_file[img_obj['id']] = fn
                        img_id_to_dim[img_obj['id']] = (img_obj.get('width'), img_obj.get('height'))
                        
                    for ann in coco_data.get('annotations', []):
                        img_id = ann['image_id']
                        if img_id in img_id_to_file:
                            fn = img_id_to_file[img_id]
                            if fn not in coco_bboxes_by_file:
                                coco_bboxes_by_file[fn] = []
                            coco_bboxes_by_file[fn].append((ann['bbox'], img_id_to_dim.get(img_id)))
                except Exception as e:
                    print(f"Warning reading {coco_json_path}: {e}")

            for img_path in breed_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    fn = img_path.name
                    target_img_name = f"{breed_key}_{fn}"
                    target_img_path = YOLO_ROOT / 'images' / yolo_split / target_img_name
                    target_lbl_path = YOLO_ROOT / 'labels' / yolo_split / f"{Path(target_img_name).stem}.txt"
                    
                    # Copy image
                    shutil.copy2(img_path, target_img_path)
                    
                    # Determine image dimensions for bbox normalization
                    img_w, img_h = 0, 0
                    bboxes_info = coco_bboxes_by_file.get(fn, [])
                    if not bboxes_info:
                        for k, v in coco_bboxes_by_file.items():
                            if k.lower() == fn.lower():
                                bboxes_info = v
                                break
                                
                    if bboxes_info and bboxes_info[0][1] and bboxes_info[0][1][0]:
                        img_w, img_h = bboxes_info[0][1]
                    else:
                        try:
                            with Image.open(img_path) as im:
                                img_w, img_h = im.width, im.height
                        except Exception:
                            img_w, img_h = 224, 224

                    # Write YOLO label lines
                    label_lines = []
                    if bboxes_info:
                        for coco_bbox, _ in bboxes_info:
                            xc, yc, nw, nh = coco_to_yolo_bbox(coco_bbox, img_w, img_h)
                            label_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                    else:
                        # Full image default bounding box
                        label_lines.append(f"{class_id} 0.500000 0.500000 1.000000 1.000000")
                        
                    with open(target_lbl_path, 'w', encoding='utf-8') as f_lbl:
                        f_lbl.write("\n".join(label_lines) + "\n")
                        
                    total_images_processed += 1
                    split_counts[yolo_split] += 1

    # Create data.yaml
    yaml_content = f"""path: {YOLO_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test

names:
"""
    for cid, bname in enumerate(BREED_NAMES):
        yaml_content += f"  {cid}: '{bname}'\n"
        
    data_yaml_path = YOLO_ROOT / "data.yaml"
    with open(data_yaml_path, 'w', encoding='utf-8') as f_yaml:
        f_yaml.write(yaml_content)
        
    print(f"\n[YOLO Dataset Prepared Successfully]")
    print(f"  Total Images Processed: {total_images_processed}")
    print(f"  Train Images: {split_counts['train']}")
    print(f"  Val Images  : {split_counts['val']}")
    print(f"  Test Images : {split_counts['test']}")
    print(f"  Configuration file created at: {data_yaml_path}")

if __name__ == '__main__':
    prepare_dataset()
