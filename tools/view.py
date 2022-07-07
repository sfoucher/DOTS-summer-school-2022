import argparse
import os

import fiftyone as fo
import pandas as pd
import PIL.Image as Image

from typing import List, Optional

parser = argparse.ArgumentParser(prog='view', 
    description='View ground truth and detections on images using FiftyOne app')

parser.add_argument('root', type=str,
    help='path to the images directory (str)')
parser.add_argument('gt', type=str,
    help='path to a csv file containing ground truth (str)')
parser.add_argument('-dets', type=str,
    help='path to a csv file containing a model\'s detections (str). Defaults to None')
parser.add_argument('-colab', type=bool, default=False,
    help='set to True to suse this tool on Google Colaboratory. Defaults to False')

args = parser.parse_args()

def _create_dataset():
    return fo.Dataset.from_images_dir('/content/data/test')

def _get_points_and_labels(img_path: str, df: pd.DataFrame) -> list:
    w, h = Image.open(img_path).size
    img_name = os.path.basename(img_path)
    records = df[df['images'] == img_name].to_dict('records')
    points = [(r['x'] / w, r['y'] / h) for r in records]
    labels = [r['labels'] for r in records]
    return points, labels

def _get_boxes_and_labels(img_path: str, df: pd.DataFrame, score: bool = False) -> list:
    w, h = Image.open(img_path).size
    img_name = os.path.basename(img_path)
    records = df[df['images'] == img_name].to_dict('records')
    boxes = [[r['x_min'] / w, r['y_min'] / h, (r['x_max']-r['x_min']) / w, (r['y_max']-r['y_min']) / h] for r in records]
    labels = [r['labels'] for r in records]
    if score:
        scores = [r['scores'] for r in records]
        return boxes, labels, scores
    else:
        return boxes, labels

def _create_keypoints(points: list, labels: list) -> List[fo.Keypoint]:
    keypoints = []
    for pt, lab in zip(points, labels):
        kp = fo.Keypoint(label=str(lab), points=[pt])
        keypoints.append(kp)
    return keypoints

def _create_boxes(boxes: list, labels: list, scores: Optional[list] = None) -> List[fo.Keypoint]:
    out_boxes = []
    if scores is not None:
        for box, lab, s in zip(boxes, labels, scores):
            box = fo.Detection(label=str(lab), bounding_box=box, confidence=[s])
            out_boxes.append(box)
    else:
        for box, lab in zip(boxes, labels):
            box = fo.Detection(label=str(lab), bounding_box=box)
            out_boxes.append(box)

    return out_boxes

def main():
    gt = pd.read_csv(args.gt)
    dets = None
    if args.dets is not None:
        dets = pd.read_csv(args.dets)

    dataset = _create_dataset()
    for sample in dataset:

        if 'x_min' in list(gt.columns):
            
            gt_boxes, gt_labels = _get_boxes_and_labels(sample.filepath, df=gt)
            sample['ground-truth'] = fo.Detections(detections=_create_boxes(gt_boxes, gt_labels))

            if dets is not None:
                dets_boxes, dets_labels, dets_scores = _get_boxes_and_labels(sample.filepath, df=dets, score=True)
                sample['predictions'] = fo.Detections(detections=_create_boxes(dets_boxes, dets_labels, scores=dets_scores))

        else:

            gt_points, gt_labels = _get_points_and_labels(sample.filepath, df=gt)
            sample['gt'] = fo.Keypoints(keypoints=_create_keypoints(gt_points, gt_labels))

            if dets is not None:
                dets_points, dets_labels = _get_points_and_labels(sample.filepath, df=dets)
                sample['predictions'] = fo.Keypoints(keypoints=_create_keypoints(dets_points, dets_labels))

        sample.save()

    if args.colab:
        session = fo.launch_app(dataset)
    else:
        session = fo.launch_app(dataset)
        session.wait()

if __name__ == '__main__':
    main()