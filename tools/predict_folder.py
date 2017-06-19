
import _init_paths
from fast_rcnn.test import return_predictions_web
from fast_rcnn.config import cfg, cfg_from_file
from networks.factory import get_network
from PIL import Image
import tensorflow as tf
import argparse
from tqdm import tqdm
from glob import glob
import numpy as np
import sys
import csv
import os

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Forward pass for folder prediction')
    parser.add_argument('--device', dest='device', help='device to use',
                        default='cpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use',
                        default=0, type=int)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--num_iter', dest='num_iter',
                        help='Iteration_num',
                        default=None, type=str)
    parser.add_argument('--data_path', dest='data_path',
                        help='Path to images',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.device == 'gpu':
        cfg.USE_GPU_NMS = True
        cfg.GPU_ID = args.device_id
    else:
        cfg.USE_GPU_NMS = False

    infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
        if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
    model_name = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
        '_iter_{0}'.format(args.num_iter) + '.ckpt')
    model_path = os.path.join(
        'output',
        cfg.GENERAL.EXP_NAME,
        'models')
    model_full_path = os.path.join(model_path, model_name)
    with tf.Graph().as_default() as validation_graph:
        network = get_network(cfg.TEST.NETWORK_NAME)
        print('Use network `{:s}` in forward pass'.format(cfg.TEST.NETWORK_NAME))
        # start a session
        saver_forward = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
    sess_forward = tf.Session(config=config, graph=validation_graph)
    saver_forward.restore(sess_forward, model_full_path)
    print('Loading model weights from {:s}'.format(model_full_path))

    # Predict boxes
    image_names = sorted(glob(os.path.join(args.data_path, "*")))
    image_boxes = []
    for img_path in tqdm(image_names):
        img = Image.open(img_path)
        # Preprocess image
        if len(img.size) == 2:
            img = img.convert('RGB')
        pred_boxes = return_predictions_web(
            sess_forward,
            network,
            np.asarray(img)
        )
        image_boxes.append(pred_boxes)

    # Convert to list
    data_to_save = []
    for i, img_name in enumerate(image_names):
        boxes = image_boxes[i]
        for j, box in enumerate(boxes):
            data_to_save.append([
                img_name,
                int(box[0]),
                int(box[1]),
                int(box[2]),
                int(box[3]),
                box[4],
                j
            ])
    det_folder = os.path.join(
        'output',
        cfg.GENERAL.EXP_NAME,
        'folder_predict')
    if not os.path.exists(det_folder):
        os.makedirs(det_folder)

    det_file = os.path.join(det_folder, 'image_detections.csv')


    with open(det_file, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['img_name', 'x1', 'y1', 'x2', 'y2', 'score', 'box_id'])
        for line in data_to_save:
            writer.writerow(line)
    print('Predictions completed and saved to folder: {0}'.format(det_file))
