from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
import shutil


coco = COCO('/Users/titrom/Desktop/hackaido/nogotochki/Nails.v1i.coco-segmentation/train/_annotations.coco.json')
img_dir = '/Users/titrom/Desktop/hackaido/nogotochki/Nails.v1i.coco-segmentation/train'
label_dir = '/Users/titrom/Desktop/hackaido/nogotochki/Nails.v1i.coco-segmentation/train_labels'
if os.path.exists(label_dir):
    shutil.rmtree(label_dir)
os.makedirs(label_dir)

for image_id in range(len(coco.imgs)):
    img = coco.imgs[image_id]

    image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
    cat_ids = coco.getCatIds()
    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)

    mask = coco.annToMask(anns[1])
    for i in range(len(anns)):
        mask += coco.annToMask(anns[i])
        
    binary_mask = mask[mask > 0] = 1
    cv2.imwrite(os.path.join(label_dir, img['file_name']), mask * 255)
