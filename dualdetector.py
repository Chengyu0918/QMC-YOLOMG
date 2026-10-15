
import os
import sys
# cwd = os.getcwd().rstrip('test')
# sys.path.append(os.path.join(cwd, './'))
cwd = os.getcwd()


import cv2

import argparse
import time
from pathlib import Path
import numpy as np
from numpy import random

import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.quality import compute_quality_evidence
from utils.torch_utils import select_device


def draw_predictions(img, label, score, box, color=(156, 39, 176), location=None):
    f_face = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = 0.5
    f_thickness, l_thickness = 1, 2
    
    h, w, _ = img.shape
    u1, v1, u2, v2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(img, (u1, v1), (u2, v2), color, l_thickness)
    
    text = '%s: %.2f' % (label, score)
    text_w, text_h = cv2.getTextSize(text, f_face, f_scale, f_thickness)[0]
    text_h += 6
    if v1 - text_h < 0:
        cv2.rectangle(img, (u1, text_h), (u1 + text_w, 0), color, -1)
        cv2.putText(img, text, (u1, text_h - 4), f_face, f_scale, (255, 255, 255), f_thickness, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (u1, v1), (u1 + text_w, v1 - text_h), color, -1)
        cv2.putText(img, text, (u1, v1 - 4), f_face, f_scale, (255, 255, 255), f_thickness, cv2.LINE_AA)
    
    if location is not None:
        text = '(%.1fm, %.1fm)' % (location[0], location[1])
        text_w, text_h = cv2.getTextSize(text, f_face, f_scale, f_thickness)[0]
        text_h += 6
        if v2 + text_h > h:
            cv2.rectangle(img, (u1, h - text_h), (u1 + text_w, h), color, -1)
            cv2.putText(img, text, (u1, h - 4), f_face, f_scale, (255, 255, 255), f_thickness, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (u1, v2), (u1 + text_w, v2 + text_h), color, -1)
            cv2.putText(img, text, (u1, v2 + text_h - 4), f_face, f_scale, (255, 255, 255), f_thickness, cv2.LINE_AA)
    
    return img

    
class Yolov5Detector():
    def __init__(self, weights='', imgsz=640):
        self.device = device = select_device('0')
        self.half = half = device.type != 'cpu' # half precision only supported on CUDA
        
        # Load model
        weights = os.path.join(cwd, weights)
        self.model = model = attempt_load(weights, map_location=device) # load FP32 model
        self.stride = stride = int(model.stride.max()) # model stride
        self.imgsz = imgsz = check_img_size(imgsz, s=stride) # check img_size
        if half:
            model.half() # to FP16
        
        # Get names
        self.names = model.module.names if hasattr(model, 'module') else model.names
        
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())),torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) # run once

    def imgdeal(self,img):
        img = letterbox(img, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float() # uint8 to fp16/32
        img /= 255.0 # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def run(self, img1, img2, conf_thres=0.1, iou_thres=0.4, classes=None):
        img0_shape = (img1.shape[0], img1.shape[1])  # original H, W
        # Padded resize
        img1 = self.imgdeal(img1)
        img2 = self.imgdeal(img2)
        # Registration-quality evidence computed from the difference map
        q = compute_quality_evidence(img2, img1)
        # Inference
        pred = self.model(img1, img2, q, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=False)

        # Process detections
        det = pred[0]

        if len(det):
            # Rescale boxes from imgsz to img0 size
            shape0 = torch.from_numpy(np.array(img0_shape)).to(self.device)
            boxes = scale_coords(img1.shape[2:], det[:, :4], shape0).round().cpu().numpy() # xyxy
            labels = [self.names[int(cls)] for cls in det[:, -1]]
            scores = [float('%.2f' % conf) for conf in det[:, -2]]
            return labels, scores, boxes
        else:
            return [], [], np.array([])


if __name__ == '__main__':
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='./runs/train/qmc_smoke/weights/best.pt',
                        help='QMC-YOLOMG checkpoint (old YOLOMG weights are incompatible)')
    parser.add_argument('--imgsz', type=int, default=640)
    opt = parser.parse_args()

    detector = Yolov5Detector(weights=opt.weights, imgsz=opt.imgsz)
    # Prefer a smoke-dataset pair (RGB + motion mask); fall back to the
    # original ARD100 demo images if present.
    rgb = sorted(glob.glob('./datasets/smoke/images/val/*.jpg'))
    msk = sorted(glob.glob('./datasets/smoke/mask/val/*.jpg'))
    if rgb and msk:
        img1 = cv2.imread(rgb[0])
        img2 = cv2.imread(msk[0])
    else:
        img1 = cv2.imread('./data/Test_images/images/phantom05_0606.jpg')
        img2 = cv2.imread('./data/Test_images/mask/phantom05_0606.jpg')

    t1 = time.time()
    labels, scores, boxes = detector.run(img1, img2, classes=[0])
    t2 = time.time()
    print('time cost:', t2 - t1, '\n')

    print('labels: ', labels)
    print('scores: ', scores)
    print('boxes: ', boxes)

