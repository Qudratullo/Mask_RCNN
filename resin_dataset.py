import json

from mrcnn.utils import Dataset
from numpy import zeros
from numpy import asarray

from os import listdir


def distance(x1, y1, x2, y2):
    return (x1 - x2)**2 + (y1 - y2)**2


def is_not_valid(x, r):
    return not 0 <= x < r


def flood_fill(masks, x, y, check, shape_id, color):
    if is_not_valid(x, len(masks[0])) or is_not_valid(y, len(masks)) or not check(x, y):
        return
    masks[x, y, shape_id] = color
    flood_fill(masks, x+1, y, check, shape_id, color)
    flood_fill(masks, x-1, y, check, shape_id, color)
    flood_fill(masks, x, y+1, check, shape_id, color)
    flood_fill(masks, x, y-1, check, shape_id, color)


def fill_polygon(masks, polygon_points, shape_id, class_id):
    random_id = len(polygon_points) // 2
    center_x = int((polygon_points[0][0] + polygon_points[random_id][0]) // 2)
    center_y = int((polygon_points[0][1] + polygon_points[random_id][1]) // 2)
    flood_fill(masks, center_x, center_y,
               lambda x, y: False,
               shape_id, class_id)


def fill_circle(masks, circle_points, shape_id, class_id):
    center_x = int(circle_points[0][0])
    center_y = int(circle_points[0][1])
    flood_fill(masks, center_x, center_y,
               lambda x, y: distance(x, y, circle_points[0][0], circle_points[0][1]) <= distance(circle_points[1][0],
                                                                                                 circle_points[1][1],
                                                                                                 circle_points[0][0],
                                                                                                 circle_points[0][1]),
               shape_id, class_id)


class ResinDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):

        # Add classes. We have only one class to add.
        self.add_class("dataset", 1, "resin")

        # define data locations for images and annotations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'

        # Iterate through all files in the folder to
        # add class, images and annotaions
        for filename in listdir(images_dir):

            # extract image id
            image_id = filename[:-4]

            # skip all images after 150 if we are building the train set
            if is_train and int(image_id) >= 3:
                continue
            # skip all images before 150 if we are building the test/val set
            if not is_train and int(image_id) < 3:
                continue

            # setting image file
            img_path = images_dir + filename

            # setting annotations file
            ann_path = annotations_dir + image_id + '.json'

            # adding images and annotations to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # extract bounding boxes from an annotation file
    def extract_shapes(self, filename):
        # load and parse the file
        with open(filename, "r") as read_file:
            data = json.load(read_file)
            return data["imageWidth"], data["imageHeight"], data["shapes"]

    # load the masks for an image
    """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
     """

    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]

        # define anntation  file location
        path = info['annotation']

        # load JSON
        w, h, shapes = self.extract_shapes(path)

        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(shapes)], dtype='uint8')

        # create masks
        class_ids = list()
        for i in range(len(shapes)):
            shape = shapes[i]
            class_id = self.class_names.index('resin')
            if shape["shape_type"] == "circle":
                fill_circle(masks, shape["points"], i, class_id)
            else:
                fill_polygon(masks, shape["points"], i, class_id)

            # masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(class_id)
        return masks, asarray(class_ids, dtype='int32')
        # load an image reference

    """Return the path of the image."""

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        print(info)
        return info['path']
