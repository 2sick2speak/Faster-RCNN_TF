# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from .imdb import imdb
from .pascal_voc import pascal_voc
from .dental import dental
from . import factory

import os.path as osp
ROOT_DIR = osp.join(osp.dirname(__file__), '..', '..')
