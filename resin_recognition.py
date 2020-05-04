import json
import matplotlib
import tensorflow as tf

from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN
import numpy as np
from numpy import zeros
from numpy import asarray
import colorsys
import argparse
import imutils
import random
import cv2
import os
import time
from matplotlib import pyplot
from matplotlib.patches import Rectangle

from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from mark_cnn_config import MaskRCNNConfig
from resin_dataset import ResinDataset


model_path = 'mask_rcnn.h5'
config = MaskRCNNConfig()
config.display()


def train():

    train_set = ResinDataset()
    train_set.load_dataset("./resin_dataset", is_train=True)
    train_set.prepare()

    test_set = ResinDataset()
    test_set.load_dataset("./resin_dataset", is_train=False)
    test_set.prepare()

    print("Loading Mask R-CNN model...")
    model = modellib.MaskRCNN(mode="training", config=config, model_dir='./')
    model.load_weights('mask_rcnn_coco.h5',
                       by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    model.train(train_set, test_set, learning_rate=2 * config.LEARNING_RATE, epochs=5, layers='heads')
    history = model.keras_model.history.history

    model.keras_model.save_weights(model_path)


def test():
    # Loading the model in the inference mode
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')
    # loading the trained weights o the custom dataset
    model.load_weights(model_path, by_name=True)
    img = load_img("..\\Kangaroo\\kangaroo-master\\kangaroo-master\\images\\00042.jpg")
    img = img_to_array(img)
    result = model.detect([img])


if __name__ == '__main__':
    train()
    test()
