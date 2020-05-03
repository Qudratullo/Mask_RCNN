import os
import cv2

import mrcnn.config
import mrcnn.visualize
import mrcnn
from mrcnn.model import MaskRCNN
import mrcnn.utils


class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.8  # минимальный процент отображения прямоугольника
    NUM_CLASSES = 2


DATASET_FILE = "mask_rcnn_coco.h5"
if not os.path.exists(DATASET_FILE):
    mrcnn.utils.download_trained_weights(DATASET_FILE)

model = MaskRCNN(mode="inference", model_dir="logs", config=MaskRCNNConfig())
# model.load_weights(DATASET_FILE, by_name=True)
model.load_weights(DATASET_FILE, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])


def visualize_detections(image, masks, boxes, class_ids, scores):
    import numpy as np
    bgr_image = image[:, :, ::-1]

    CLASS_NAMES = ['BG', "person", "bicycle", "car", "motorcycle", "bus", "truck"]
    COLORS = mrcnn.visualize.random_colors(len(CLASS_NAMES))

    for i in range(boxes.shape[0]):
        y1, x1, y2, x2 = boxes[i]

        classID = class_ids[i]
        label = CLASS_NAMES[classID]
        font = cv2.FONT_HERSHEY_DUPLEX
        color = [int(c) for c in np.array(COLORS[classID]) * 255]
        text = "{}: {:.3f}".format(label, scores[i])
        size = 0.8
        width = 2

        cv2.rectangle(bgr_image, (x1, y1), (x2, y2), color, width)
        cv2.putText(bgr_image, text, (x1, y1 - 20), font, size, color, width)


IMAGE_DIR = os.path.join(os.getcwd(), "images")
for filename in os.listdir(IMAGE_DIR):
    image = cv2.imread(os.path.join(IMAGE_DIR, filename))
    rgb_image = image[:, :, ::-1]
    detections = model.detect([rgb_image], verbose=1)[0]
    print(detections)

    visualize_detections(image, detections['masks'], detections['rois'], detections['class_ids'], detections['scores'])
