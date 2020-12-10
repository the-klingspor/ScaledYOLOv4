import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader, LoadImages
from utils.general import (
    coco80_to_coco91_class, check_file, check_img_size, compute_loss, non_max_suppression,
    scale_coords, xyxy2xywh, clip_coords, plot_images, xywh2xyxy, box_iou, output_to_target, ap_per_class)
from utils.torch_utils import select_device, time_synchronized
from test import load_classes
from active_learning.scoring import entropy, mutual_info_scores
from active_learning.sampling import top, kmpp, core_set

from models.models import *


def active_sampling(data,
                    weights=None,
                    ema_weights=None,
                    scoring='entropy',
                    sampling='top',
                    aggr='max',
                    imgsz=640,
                    single_cls=False,
                    augment=False,
                    save_dir=''):
    device = select_device(opt.device)
    save_txt = opt.save_txt  # save *.txt labels
    if save_txt:
        out = Path('active/output')
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Remove previous
    for f in glob.glob(str(Path(save_dir) / 'test_batch*.jpg')):
        os.remove(f)

    # Load model
    model = Darknet(opt.cfg).to(device)

    # load model
    try:
        ckpt = torch.load(weights[0], map_location=device)  # load checkpoint
        ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(ckpt['model'], strict=False)
    except:
        load_darknet_weights(model, weights[0])
    imgsz = check_img_size(imgsz, s=32)  # check img_size

    # load ema teacher model
    ema = None
    if ema_weights:
        ema = Darknet(opt.cgf).to(device)

        try:
            ckpt = torch.load(ema_weights[0], map_location=device)  # load checkpoint
            ckpt['ema'] = {k: v for k, v in ckpt['ema'].items() if model.state_dict()[k].numel() == v.numel()}
            ema.load_state_dict(ckpt['ema'], strict=False)
        except:
            load_darknet_weights(ema, ema_weights[0])

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()
        if ema:
            ema.half()

    # Configure
    model.eval()
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    nc = 1 if single_cls else int(data['nc'])  # number of classes

    # Dataloader
    dataset = LoadImages(data, img_size=imgsz)
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    if ema:
        _ = ema(img.half() if half else img) if device.type != 'cpu' else None  # run once

    try:
        names = model.names if hasattr(model, 'names') else model.module.names
    except:
        names = load_classes(opt.names)

    # Lists for all scores and their corresponding paths
    scores = []
    paths = []

    # compute score for every image in the dataset
    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        if scoring == "entropy":
            score = entropy(pred, aggr)
        elif scoring == "mutual_information":
            pred_teacher = ema(img, augment=augment)[0]
            score = mutual_info_scores(pred, pred_teacher, aggr)
        else:
            raise AssertionError("Valid scoring schemes are entropy and mutual"
                                 "information")

        scores.append(score)
        paths.append(path)

    # Apply sampling to order image paths
    if sampling == "top":
        paths = top(scores, paths)
    elif sampling == "kmpp":
        paths = kmpp(scores, paths)
    elif sampling == "core_set":
        paths = core_set(scores, paths)
    else:
        raise AssertionError("Valid sampling methods are top, k-means++ and "
                             "core-set")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='active.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4.pt', help='model.pt path(s)')
    parser.add_argument('--ema_weights', nargs='+', type=str, default=None, help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--scoring', type=str, default='entropy', help='Scoring strategy, i.e. entropy or mutual info')
    parser.add_argument('--sampling', type=str, default='top', help='Sampling strategy, i.e. top or kmpp')
    parser.add_argument('--aggr', type=str, default='max', help='aggregation strategy, i.e. max or avg')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--cfg', type=str, default='cfg/yolov4.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)

    active_sampling(opt.data,
                    opt.weights,
                    opt.ema_weights,
                    opt.scoring,
                    opt.sampling,
                    opt.aggr,
                    opt.img_size,
                    opt.save_json,
                    opt.single_cls,
                    opt.augment)
