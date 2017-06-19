# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

#import datasets
import os
from datasets.imdb import imdb
from datasets.metric_utils import collect_metrics
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import pickle
from fast_rcnn.config import cfg


class dental(imdb):
    def __init__(self, image_set, img_classes, devkit_path=None, ext='.jpg'):
        imdb.__init__(self, image_set)
        self._image_set = image_set
        self._devkit_path = self._get_default_path(devkit_path)
        self._data_path = os.path.join(self._devkit_path, 'VOC')
        self._classes = tuple(['__background__'] + img_classes) # always index 0 for backgroung
        print(self._classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_ext = ext
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        assert os.path.exists(self._devkit_path), \
                'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(
            self._data_path,
            'JPEGImages',
            index + self._image_ext
            )
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOC/ImageSets/Main/train.txt
        image_set_file = os.path.join(
            self._data_path,
            'ImageSets',
            'Main',
            self._image_set + '.txt'
            )
        assert os.path.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self, devkit_path):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        path = 'dental' if devkit_path is None else devkit_path
        return os.path.join(cfg.DATA_DIR, path)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(
            self._data_path,
            'Annotations',
            index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            if x2 - x1 <=0:
                print(x2, x1)
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def evaluate_metrics(self, pred_boxes, iou_threshold=cfg.TEST.IOU_THRESHOLD):

        ground_truth = [item['boxes'] for item in self.gt_roidb()]
        ground_truth = dict(zip(self.image_index, ground_truth))

        # Collect bboxes
        pred_boxes_dict = {}
        for box in pred_boxes:
            image_index = box[0].split('/')[-1].replace(self._image_ext, '')
            box_coord = box[1:5]
            boxes_list = pred_boxes_dict.get(image_index, list())
            boxes_list.append(box_coord)
            pred_boxes_dict[image_index] = boxes_list

        metrics = collect_metrics(
            ground_truth,
            pred_boxes_dict,
            self.image_index,
            cfg.TEST.IOU_THRESHOLD)
        return metrics
