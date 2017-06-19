# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.pascal_voc
import datasets.dental

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        print(name)
        __sets[name] = (lambda split=split, year=year:
                datasets.pascal_voc(split, year))

# Dental dataset
for split in ['trainval', 'test']:
    name = 'dental_{}'.format(split)
    print(name)
    dental_classes = ['tooth']
    __sets[name] = (lambda split=split:
            datasets.dental(split, dental_classes, 'dental'))

## Periodontitis dataset
for split in ['trainval', 'test']:
    name = 'periodont_{}'.format(split)
    print(name)
    periodont_classes = ['periodontitis']
    __sets[name] = (lambda split=split:
            datasets.dental(split, periodont_classes, 'periodontitis', '.jpeg'))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in  __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()
