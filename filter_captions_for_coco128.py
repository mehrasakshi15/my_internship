"""
Filter the official COCO caption annotations down to only the images that
exist in a local COCO128 folder, producing a small captions_coco128.json
that plugs directly into dataset.py / build_vocab_from_coco() from the
Show and Tell PyTorch reproduction.

COCO128 (Ultralytics) images are sampled from COCO train2017, so this script
matches against the 2017 caption annotation files by default.

Usage:
    1) Download ONLY the annotations (not the 13GB+ images) from:
       http://images.cocodataset.org/annotations/annotations_trainval2017.zip
       (~241MB, contains captions_train2017.json + captions_val2017.json
       among others) and unzip it into ./annotations/

    2) Point --coco128_dir at the folder containing your COCO128 .jpg images
       (e.g. coco128/images/train2017)

    3) Run:
       python filter_captions_for_coco128.py \
           --coco128_dir ./coco128/images/train2017 \
           --annotations_dir ./annotations \
           --out ./captions_coco128.json
"""
import argparse
import json
import os


def load_image_ids_from_folder(folder):
    """COCO filenames look like 000000000009.jpg -> image_id 9."""
    ids = set()
    for fname in os.listdir(folder):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        stem = os.path.splitext(fname)[0]
        try:
            ids.add(int(stem))
        except ValueError:
            print(f'Skipping unrecognized filename: {fname}')
    return ids


def filter_annotations(caption_json_paths, wanted_ids):
    images_by_id = {}
    annotations = []
    for path in caption_json_paths:
        if not os.path.exists(path):
            print(f'  (not found, skipping) {path}')
            continue
        print(f'  reading {path} ...')
        with open(path) as f:
            data = json.load(f)
        for im in data['images']:
            if im['id'] in wanted_ids:
                images_by_id[im['id']] = {'id': im['id'], 'file_name': im['file_name']}
        for ann in data['annotations']:
            if ann['image_id'] in wanted_ids:
                annotations.append({'id': ann['id'], 'image_id': ann['image_id'], 'caption': ann['caption']})
    return list(images_by_id.values()), annotations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco128_dir', required=True, help='Folder with your COCO128 .jpg images')
    parser.add_argument('--annotations_dir', required=True,
                         help='Folder where annotations_trainval2017.zip was unzipped')
    parser.add_argument('--out', default='./captions_coco128.json')
    args = parser.parse_args()

    wanted_ids = load_image_ids_from_folder(args.coco128_dir)
    print(f'Found {len(wanted_ids)} images in {args.coco128_dir}')

    candidate_files = [
        os.path.join(args.annotations_dir, 'annotations', 'captions_train2017.json'),
        os.path.join(args.annotations_dir, 'annotations', 'captions_val2017.json'),
        os.path.join(args.annotations_dir, 'captions_train2017.json'),
        os.path.join(args.annotations_dir, 'captions_val2017.json'),
    ]
    print('Searching for caption annotation files:')
    images, annotations = filter_annotations(candidate_files, wanted_ids)

    matched_ids = set(im['id'] for im in images)
    missing = wanted_ids - matched_ids
    print(f'\nMatched {len(images)} / {len(wanted_ids)} images with captions.')
    print(f'Total captions collected: {len(annotations)}')
    if missing:
        print(f'{len(missing)} images had no matching captions (ids: {sorted(missing)[:10]}...)')

    out = {'images': images, 'annotations': annotations}
    with open(args.out, 'w') as f:
        json.dump(out, f)
    print(f'\nWrote {args.out} -- point config.py caption_file_train at this file, '
          f'and image_dir_train at your coco128_dir.')
