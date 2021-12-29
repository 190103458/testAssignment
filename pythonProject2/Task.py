import random

import PIL
import cv2
import numpy as np
import torch
import torchvision
from os import walk

torch.set_grad_enabled(False)
import matplotlib.pylab as plt

plt.rcParams["axes.grid"] = False
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
from torchvision import transforms as T

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def random_colour_masks(image):
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def draw(img_path, masks, boxes, pred_cls, threshold=0.7, rect_th=3, text_size=1, text_th=3):
    ##TODO
    img = cv2.imread("static/uploads/" + img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
        rgb_mask = random_colour_masks(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, (int(boxes[i][0][0]), int(boxes[i][0][1])), (int(boxes[i][1][0]), int(boxes[i][1][1])),
                      color=(0, 255, 0), thickness=3)
        cv2.putText(img, pred_cls[i], (int(boxes[i][0][0]), int(boxes[i][0][1])), cv2.FONT_HERSHEY_SIMPLEX,
                    text_size,
                    (0, 0, 255), thickness=text_th)
    cv2.imwrite(('/usr/src/app/static/treated/' + img_path), img)


def get_prediction_v2(img, threshold):
    ##loading an image
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    masks = masks[:pred_t + 1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return masks, pred_boxes, pred_class


def create_category_annotation(key, value):
    category = {
        "supercategory": key,
        "id": value,
        "name": key
    }

    return category


def create_annotation_format(box, image_id, category_id, annotation_id):
    annotation = {
        "area": box[1][0] * box[1][1],
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": [int(box[0][0]), int(box[0][1]), int(box[1][0]) - int(box[0][0]), int(box[1][1]) - int(box[0][1])],
        "category_id": category_id,
        "id": annotation_id
    }
    return annotation


def create_image_annotation(file_name, width, height, image_id):
    images = {
        "file_name": file_name,
        "height": height,
        "width": width,
        "id": image_id
    }

    return images


class Task:
    category_ids = dict()
    cat_id = 0
    annotation_id = 0
    image_id = 0
    annotations = []
    images = []
    coco_format = {}
    files = []
    imgs = []

    def __init__(self):
        coco_format = self.get_coco_json_format()

        ## TODO

    def get_coco_json_format(self):
        # Standard COCO format
        self.coco_format = {
            "type": "instance",
            "images": [],
            "categories": [],
            "annotations": []
        }
        return self.coco_format

    def instance_segmentation_api_v3(self, img_names, imgs, threshold=0.5, rect_th=3, text_size=1, text_th=3):
        self.image_id =0
        for img in imgs:
            masks, boxes, pred_cls = get_prediction_v2(img, threshold)
            width, height = img.size
            image = create_image_annotation(img_names[self.image_id], width, height, self.image_id)
            self.images.append(image)

            for i in range(len(masks)):
                if pred_cls[i] in self.category_ids:
                    annotation = create_annotation_format(boxes[i], self.image_id, self.category_ids[pred_cls[i]],
                                                          self.annotation_id)
                    self.annotations.append(annotation)
                    self.annotation_id += 1
                else:
                    self.coco_format["categories"].append(create_category_annotation(pred_cls[i], self.cat_id))
                    self.category_ids[pred_cls[i]] = self.annotation_id
                    self.cat_id += 1
                    annotation = create_annotation_format(boxes[i], self.image_id, self.category_ids[pred_cls[i]],
                                                          self.annotation_id)
                    self.annotations.append(annotation)
                    self.annotation_id += 1
            draw(img_names[self.image_id], masks, boxes, pred_cls)
            self.image_id += 1
        return self.images, self.annotations, self.image_id

    def clear(self):
        self.coco_format["annotations"] = []
        self.coco_format["images"] = []
        self.coco_format["categories"] = []
        self.category_ids.clear()
        self.files = []
        self.imgs = []
        self.images = []
        self.annotations = []
        self.annotation_id = 0
        self.cat_id = 0

    def treat(self):
        self.clear()
        for (dirpath, dirnames, filenames) in walk("static/uploads/"):
            self.files.extend(filenames)
            break

        for i in self.files:
            self.imgs.append(PIL.Image.open("static/uploads/" + i).convert("RGB"))

        self.coco_format["images"], self.coco_format["annotations"], annotation_cnt = self.instance_segmentation_api_v3(
            self.files, self.imgs, threshold=0.7)

        return self.coco_format
