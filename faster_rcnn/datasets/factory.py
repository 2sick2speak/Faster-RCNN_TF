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

# Caries dataset
for split in ['trainval', 'test']:
    name = 'caries_{}'.format(split)
    print(name)
    caries_classes = ['caries']
    __sets[name] = (lambda split=split:
            datasets.dental(split, caries_classes, 'caries'))

## Periodontitis dataset
for split in ['trainval', 'test']:
    name = 'periodont_{}'.format(split)
    print(name)
    periodont_classes = ['periodontitis']
    __sets[name] = (lambda split=split:
            datasets.dental(split, periodont_classes, 'periodont'))

## Periodontitis updated dataset
for split in ['trainval', 'test']:
    name = 'periodontitis_{}'.format(split)
    print(name)
    periodontitis_classes = ['periodontitis']
    __sets[name] = (lambda split=split:
            datasets.dental(split, periodontitis_classes, 'periodontitis'))

## Caries + periodontitis dataset
for split in ['trainval', 'test']:
    name = 'cp_{}'.format(split)
    print(name)
    cp_classes = ['periodontitis', 'caries']
    __sets[name] = (lambda split=split:
            datasets.dental(split, cp_classes, 'cp'))

## Caries + periodontitis dataset bitewings
for split in ['trainval', 'test']:
    name = 'bitewing_cp_{}'.format(split)
    print(name)
    cp_classes = ['periodontitis', 'caries']
    __sets[name] = (lambda split=split:
            datasets.dental(split, cp_classes, 'bitewing_cp'))

## Caries + periodontitis dataset bitewings (+probable cases)
for split in ['trainval', 'test']:
    name = 'bitewing_cp_prob_{}'.format(split)
    print(name)
    cp_classes = ['periodontitis', 'caries']
    __sets[name] = (lambda split=split:
            datasets.dental(split, cp_classes, 'bitewing_cp_prob'))

## Caries + periodontitis dataset bitewings augmented
for split in ['trainval', 'test']:
    name = 'bitewing_cp_augm_{}'.format(split)
    print(name)
    cp_classes = ['periodontitis', 'caries']
    __sets[name] = (lambda split=split:
            datasets.dental(split, cp_classes, 'bitewing_cp_augm'))

## Caries + periodontitis dataset bitewings augmented
for split in ['trainval', 'test']:
    name = 'bitewing_cp_prob_augm_{}'.format(split)
    print(name)
    cp_classes = ['periodontitis', 'caries']
    __sets[name] = (lambda split=split:
            datasets.dental(split, cp_classes, 'bitewing_cp_prob_augm'))

## Caries + periodontitis dataset panorama crops
for split in ['trainval', 'test']:
    name = 'panorama_crops_cp_{}'.format(split)
    print(name)
    cp_classes = ['periodontitis', 'caries']
    __sets[name] = (lambda split=split:
            datasets.dental(split, cp_classes, 'panorama_crops_cp'))

## Caries + periodontitis dataset panorama crops + augmented bitewing 320 with prob_*
for split in ['trainval', 'test']:
    name = 'mix_cp_{}'.format(split)
    print(name)
    cp_classes = ['periodontitis', 'caries']
    __sets[name] = (lambda split=split:
            datasets.dental(split, cp_classes, 'mix_cp'))

## Pathologies + treatements mistakes + treatement resuts dataset augmented mix 320, bitewing, caries, periodontitis with prob_*
for split in ['trainval', 'test']:
    name = 'mix320_bitewing_caries_periodontitis_cp_{}'.format(split)
    print(name)
    cp_classes = ['restoration', 'caries', 'endodontic_treat', 'crown', 'implant','overfilling', 'periodontitis', 'underfilling', 'void', 'apical_perforation', 'ledge']
    __sets[name] = (lambda split=split:
            datasets.dental(split, cp_classes, 'mix320_bitewing_caries_periodontitis_cp'))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in  __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()
