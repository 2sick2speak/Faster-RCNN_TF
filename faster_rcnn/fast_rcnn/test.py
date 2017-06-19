from fast_rcnn.config import cfg, get_output_dir
from utils.timer import Timer
import numpy as np
import cv2
from utils.cython_nms import nms
from utils.blob import im_list_to_blob
import os
import csv
from fast_rcnn.bbox_transform import bbox_transform_inv
from tqdm import tqdm

from datasets.factory import get_imdb
from networks.factory import get_network
import tensorflow as tf

def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob
    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob
    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)
    scales = np.array(scales)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""

    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)

    return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


def im_detect(sess, net, im, boxes=None):
    """Detect object classes in an image given object proposals.
    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    blobs, im_scales = _get_blobs(im, boxes)

    im_blob = blobs['data']
    blobs['im_info'] = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
        dtype=np.float32)

    # forward pass
    feed_dict={net.data: blobs['data'], net.im_info: blobs['im_info'], net.keep_prob: 1.0}

    run_options = None
    run_metadata = None

    cls_score, cls_prob, bbox_pred, rois = sess.run([net.get_output('cls_score'), net.get_output('cls_prob'), net.get_output('bbox_pred'),net.get_output('rois')],
                                                    feed_dict=feed_dict,
                                                    options=run_options,
                                                    run_metadata=run_metadata)

    assert len(im_scales) == 1, "Only single-image batch implemented"
    boxes = rois[:, 1:5] / im_scales[0]


    if cfg.TEST.NO_SOFTMAX:
        # use the raw scores before softmax
        scores = cls_score
    else:
        # use softmax estimated probabilities
        scores = cls_prob

    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im.shape)

    return scores, pred_boxes

def test_net(sess, net, imdb, num_iter=None):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    extra_files_path = '_' + num_iter if num_iter is not None else ''

    # Dir for images

    output = get_output_dir()
    path_to_save = os.path.join(
        output, 'images_with_boxes', 'iteration{0}'.format(extra_files_path))

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    class_colors = {}
    from random import randint
    for cl in imdb._classes[1:]:
        class_colors[cl] = (randint(0, 255), randint(0, 255), randint(0, 255))


    for i in tqdm(range(num_images)):
        # filter out any ground truth boxes
        box_proposals = None

        img_path = imdb.image_path_at(i)
        im = cv2.imread(img_path)
        img_name = img_path.split('/')[-1]

        _t['im_detect'].tic()
        scores, boxes = im_detect(sess, net, im, box_proposals)
        _t['im_detect'].toc()

        _t['misc'].tic()

        # For image saving
        img = im
        all_boxes[0][i] = img_path

        # skip j = 0, because it's the background class
        for j in range(1, imdb.num_classes):
            inds = np.where(scores[:, j] > cfg.TEST.IMAGE_THRESH)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            all_boxes[j][i] = cls_dets # [fclass, image]
            
            # Put rectangle and save images
            for box in cls_dets:
                cv2.rectangle(
                    img,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    class_colors[imdb._classes[j]],
                    2
                )
                textLabel = '{0:.2f}'.format(100*box[4])
                (retval, baseLine) = cv2.getTextSize(
                    textLabel,
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    1
                )
                cv2.putText(
                    img,
                    textLabel,
                    (box[0], box[1]),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )

        cv2.imwrite(os.path.join(path_to_save, img_name), img)

        # Limit to max_per_image detections *over all classes*
        if cfg.TEST.MAX_PER_IMAGE > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, imdb.num_classes)])
            if len(image_scores) > cfg.TEST.MAX_PER_IMAGE:
                image_thresh = np.sort(image_scores)[-cfg.TEST.MAX_PER_IMAGE]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()

    bounding_boxes = os.path.join(output, 'bounding_boxes')
    if not os.path.exists(bounding_boxes):
        os.makedirs(bounding_boxes)

    det_file = os.path.join(bounding_boxes, 'bounding_boxes{0}.csv'.format(extra_files_path))
    image_names = all_boxes[0]
    image_boxes = all_boxes[1]
    #print(len(all_boxes))
    #print(len(all_boxes[2]))
    #print(imdb._classes)

    # TODO Don't Work for more than 1 class classification. FIXME

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

    metrics = imdb.evaluate_metrics(data_to_save)
    metrics_path = os.path.join(output, 'metrics')
    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)

    metrics_filename = os.path.join(metrics_path, 'metrics{0}.csv'.format(extra_files_path))
    metrics.to_csv(metrics_filename, index=False)
    print(metrics.describe())

    with open(det_file, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['img_name', 'x1', 'y1', 'x2', 'y2', 'score', 'box_id'])
        for line in data_to_save:
            writer.writerow(line)

    print('Validation finished for iter num: {0}'.format(num_iter))

    return metrics

def return_predictions_web(sess, net, img):

    # filter out any ground truth boxes
    box_proposals = None
    scores, boxes = im_detect(sess, net, img, box_proposals)

    # FIXME Hardcode for web serving
    num_classes = 2
    thresh=0.05

    all_boxes = [[[] for _ in range(1)]
                 for _ in range(num_classes)]

    # skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > thresh)[0]
        cls_scores = scores[inds, j]
        cls_boxes = boxes[inds, j*4:(j+1)*4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
        keep = nms(cls_dets, cfg.TEST.NMS)
        cls_dets = cls_dets[keep, :]
        all_boxes[j][0] = cls_dets # [fclass, image]

    image_names = all_boxes[0]
    image_boxes = all_boxes[1]

    data_to_save = []
    for i, img_name in enumerate(image_names):
        boxes = image_boxes[i]
        for j, box in enumerate(boxes):
            data_to_save.append([
                int(box[0]),
                int(box[1]),
                int(box[2]),
                int(box[3]),
                float(box[4]),
                j
            ])
    return data_to_save

def validate_model(num_iter):
    """
    Validate model on test set.
    """

    infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
    if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
    model_name = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
        '_iter_{0}'.format(num_iter) + '.ckpt')
    model_path = os.path.join(
        'output',
        cfg.GENERAL.EXP_NAME,
        'models')
    imdb_name = cfg.GENERAL.DATASET_NAME + '_test'
    imdb = get_imdb(imdb_name)
    model_full_path = os.path.join(model_path, model_name)
    with tf.Graph().as_default() as validation_graph:
        network = get_network(cfg.TEST.NETWORK_NAME)
        print('Use network `{:s}` in training'.format(cfg.TEST.NETWORK_NAME))
            # start a session
        saver_forward = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
    sess_forward = tf.Session(config=config, graph=validation_graph)
    saver_forward.restore(sess_forward, model_full_path)
    print('Loading model weights from {:s}'.format(model_full_path))

    metrics = test_net(sess_forward, network, imdb, str(num_iter))
    # TODO Check on new drivers that session is closing
    sess_forward.close()
    return metrics
