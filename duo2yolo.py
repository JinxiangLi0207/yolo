"""
DUO 数据集 COCO 格式转 YOLO 格式脚本

DUO 数据集信息：
- 4 个类别：holothurian, echinus, scallop, starfish
- 标注格式：COCO JSON (instances_train.json, instances_test.json)
- 目标格式：YOLO TXT (每张图片一个同名txt文件)

使用方法：
    python duo2yolo.py
"""

import json
import os
from pathlib import Path


def coco_to_yolo(coco_json_path, output_dir, image_dir):
    """
    将 COCO JSON 格式转换为 YOLO TXT 格式

    Args:
        coco_json_path: COCO JSON 文件路径
        output_dir: 输出标签目录
        image_dir: 图片目录（用于获取图片尺寸）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取 COCO JSON
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # 获取类别信息
    categories = coco['categories']
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    cat_name_to_id = {cat['name']: idx for idx, cat in enumerate(categories)}

    print(f"类别映射：")
    for old_id, name in cat_id_to_name.items():
        print(f"  {name}: {old_id} -> {cat_name_to_id[name]}")

    # 获取图片信息
    images = {img['id']: img for img in coco['images']}

    # 按图片分组标注
    annotations_by_image = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # 转换每张图片的标注
    converted = 0
    for img_id, img_info in images.items():
        # 获取图片信息
        file_name = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']

        # 获取该图片的所有标注
        anns = annotations_by_image.get(img_id, [])

        # 生成 YOLO 标签文件
        txt_name = os.path.splitext(file_name)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_name)

        with open(txt_path, 'w') as f:
            for ann in anns:
                # 获取类别ID（转换为新的连续ID）
                cat_name = cat_id_to_name[ann['category_id']]
                new_cat_id = cat_name_to_id[cat_name]

                # 获取边界框 [x, y, width, height]
                bbox = ann['bbox']
                x, y, w, h = bbox

                # 转换为 YOLO 格式 [x_center, y_center, width, height] (归一化)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                norm_w = w / img_width
                norm_h = h / img_height

                # 写入标签
                f.write(f"{new_cat_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        converted += 1

    print(f"转换完成：{converted} 张图片")
    print(f"输出目录：{output_dir}")


def main():
    # 设置路径
    base_dir = Path("F:/yolov11attention/yolov11-attention/DUO/DUO")

    # 训练集
    train_json = base_dir / "annotations" / "instances_train.json"
    train_images = base_dir / "images" / "train"
    train_labels = base_dir / "labels" / "train"

    # 测试集
    test_json = base_dir / "annotations" / "instances_test.json"
    test_images = base_dir / "images" / "test"
    test_labels = base_dir / "labels" / "test"

    # 转换训练集
    print("=" * 50)
    print("转换训练集...")
    print("=" * 50)
    coco_to_yolo(train_json, train_labels, train_images)

    # 转换测试集
    print("\n" + "=" * 50)
    print("转换测试集...")
    print("=" * 50)
    coco_to_yolo(test_json, test_labels, test_images)

    # 统计信息
    print("\n" + "=" * 50)
    print("统计信息：")
    print("=" * 50)

    # 统计训练集
    train_img_count = len(list(train_images.glob("*.jpg")))
    train_lbl_count = len(list(train_labels.glob("*.txt")))
    print(f"训练集：{train_img_count} 张图片，{train_lbl_count} 个标签")

    # 统计测试集
    test_img_count = len(list(test_images.glob("*.jpg")))
    test_lbl_count = len(list(test_labels.glob("*.txt")))
    print(f"测试集：{test_img_count} 张图片，{test_lbl_count} 个标签")

    # 统计各类别数量
    print("\n类别统计（训练集）：")
    class_count = {0: 0, 1: 0, 2: 0, 3: 0}
    for lbl_file in train_labels.glob("*.txt"):
        with open(lbl_file, 'r') as f:
            for line in f:
                if line.strip():
                    cls_id = int(line.split()[0])
                    class_count[cls_id] += 1

    classes = ['holothurian', 'echinus', 'scallop', 'starfish']
    for cls_id, count in class_count.items():
        print(f"  {classes[cls_id]}: {count}")


if __name__ == "__main__":
    main()
